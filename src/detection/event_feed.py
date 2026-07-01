"""Event-feed Scout source — drive clipping from a detailed textual match report.

This is the most reliable + cheapest way to know WHEN the key moments happen:
instead of guessing from scoreboard OCR (false positives) or audio spikes, we
ingest a structured play-by-play (minute, team, player, event type) that is
freely published for most matches, and map each event's MATCH CLOCK to a VIDEO
TIMESTAMP via a per-period kick-off offset (the approach used by ClipMaker:
https://github.com/B03GHB4L1/ClipMaker — kick-off timestamp mapping per half/
extra-time/penalties).

Supported inputs (auto-detected by `load_events`):
  * CSV   — flexible headers: minute[,second][,period],team,player[,number],
            type[,importance][,text]
  * JSON  — a list of objects with the same keys (or {"events": [...]})
  * text  — one event per line, e.g. "67' GOAL — Messi (Argentina)" /
            "23: yellow card, Rodri (Spain)"

Match-clock -> video-time:
    video_t = kickoff[period] + (event_minute - period_start_minute)*60 + second
Kick-offs are provided in config (detect.event_feed.kickoffs) or the WebUI, e.g.
    {1: 0, 2: 2740}     # 2nd-half kick-off is at 45:40 of the video file
A single per-half offset can't model in-half stoppages perfectly, so windows are
padded generously and (optionally) snapped to the on-screen clock by the
scoreboard-OCR refiner — see `align_to_ocr` (best-effort, never fatal).

COMPLIANCE: scraping some providers (e.g. WhoScored) may breach their ToS. Prefer
public-domain feeds (openfootball) or licensed APIs (API-Football, TheSportsDB).
Sourcing rights are the operator's responsibility — this module only parses a
feed the operator supplies.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass

from ..utils.io import get_logger
from .scout import EventWindow, _dedupe, _window_bounds

log = get_logger()

# textual event keyword -> our window kind
_KIND_KEYWORDS = {
    "goal": "goal", "scores": "goal", "penalty goal": "goal", "own goal": "goal",
    "save": "save", "saved": "save",
    "shot": "chance", "chance": "chance", "miss": "chance", "post": "chance",
    "bar": "chance", "header": "chance", "free kick": "chance", "offside": "chance",
    "skill": "skill", "dribble": "skill", "nutmeg": "skill", "take-on": "skill",
    "red card": "card", "yellow card": "card", "card": "card", "sent off": "card",
    "var": "card", "penalty": "card",
}
# events we never clip on their own (noise)
_SKIP_KINDS = {"sub", "substitution", "kick-off", "kickoff", "half-time",
               "full-time", "corner", "throw-in", "goal kick"}

_PERIOD_START_MIN = {1: 0, 2: 45, 3: 90, 4: 105}   # ET1 starts at 90', ET2 at 105'


@dataclass
class MatchEvent:
    minute: int
    kind: str
    second: float = 0.0
    period: int = 1
    team: str | None = None
    player: str | None = None
    number: int | None = None
    importance: float | None = None    # None -> derived from kind (xT-style)
    text: str = ""

    def __post_init__(self):
        if self.importance is None:
            self.importance = _default_importance(self.kind)

    def match_seconds(self) -> float:
        """Seconds since this period's nominal start (e.g. 2nd-half 67' -> 22*60)."""
        base = _PERIOD_START_MIN.get(self.period, 0)
        return max(0.0, (self.minute - base) * 60.0 + self.second)


# --------------------------------------------------------------------------- #
def load_events(source: str | list | dict, cfg: dict | None = None) -> list[MatchEvent]:
    """Parse an event feed from a path (.csv/.json/.txt) or an in-memory list/dict."""
    if isinstance(source, (list, dict)):
        rows = source.get("events", []) if isinstance(source, dict) else source
        return _coerce_rows(rows)
    path = str(source)
    low = path.lower()
    try:
        if low.endswith(".csv"):
            with open(path, newline="", encoding="utf-8") as fh:
                return _coerce_rows(list(csv.DictReader(fh)))
        if low.endswith(".json"):
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            rows = data.get("events", []) if isinstance(data, dict) else data
            return _coerce_rows(rows)
        # fall back to line-based text
        with open(path, encoding="utf-8") as fh:
            return _parse_text("\n".join(fh.readlines()))
    except FileNotFoundError:
        log.warning(f"[event_feed] feed not found: {path}")
        return []
    except Exception as exc:  # noqa: BLE001
        log.warning(f"[event_feed] failed to parse {path}: {exc}")
        return []


def load_descriptive_events(source: str | list | dict,
                            cfg: dict | None = None) -> list[MatchEvent]:
    """Parse a MANUALLY-supplied descriptive match log into MatchEvents.

    A superset of `load_events` that additionally understands the two common
    analyst-export JSON shapes, so an operator can paste/upload real data
    instead of relying on API guessing:

      * **StatsBomb** open-data events — a list of event objects with
        ``type.name`` (Shot/Dribble/Goal Keeper/…), ``minute``/``second``,
        ``team.name``, ``player.name``, and outcome sub-objects
        (``shot.outcome.name == 'Goal'``, ``foul_committed.card.name``, …).
      * **SoccerNet** action labels — ``{"annotations": [{"gameTime": "1 - 12:34",
        "label": "Goal", "team": "home"}, …]}`` (gameTime = half - MM:SS).

    Anything else (generic CSV / free-text captions / generic JSON list) is
    delegated to `load_events`. Never raises — returns [] on any failure.
    """
    try:
        # resolve to in-memory data for JSON; delegate CSV/text to load_events
        if isinstance(source, (list, dict)):
            data = source
        else:
            path = str(source)
            if path.lower().endswith(".json"):
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
            else:
                return load_events(source, cfg)

        # SoccerNet: {"annotations": [...]}
        if isinstance(data, dict) and isinstance(data.get("annotations"), list):
            events = _parse_soccernet(data["annotations"])
            log.info(f"[event_feed] SoccerNet log: {len(events)} events "
                     f"({sum(e.kind == 'goal' for e in events)} goals)")
            return events

        rows = data.get("events", []) if isinstance(data, dict) else data
        # StatsBomb: list of {"type": {"name": ...}, ...}
        if _is_statsbomb(rows):
            events = _parse_statsbomb(rows)
            log.info(f"[event_feed] StatsBomb log: {len(events)} events "
                     f"({sum(e.kind == 'goal' for e in events)} goals)")
            return events

        # generic JSON list/dict of rows
        return _coerce_rows(rows if isinstance(rows, list) else [])
    except FileNotFoundError:
        log.warning(f"[event_feed] descriptive log not found: {source}")
        return []
    except Exception as exc:  # noqa: BLE001
        log.warning(f"[event_feed] failed to parse descriptive log: {exc}")
        return []


def _is_statsbomb(rows) -> bool:
    return (isinstance(rows, list) and len(rows) > 0 and isinstance(rows[0], dict)
            and isinstance(rows[0].get("type"), dict)
            and "name" in rows[0]["type"])


def _statsbomb_kind(tname: str, e: dict) -> str | None:
    """Map a StatsBomb event to our clip kind, or None to ignore (pass/carry/…)."""
    t = (tname or "").lower()
    if t == "shot":
        outcome = str((((e.get("shot") or {}).get("outcome")) or {})
                      .get("name", "")).lower()
        return "goal" if outcome == "goal" else "chance"
    if t == "own goal against":
        return "goal"
    card = (((e.get("foul_committed") or {}).get("card"))
            or ((e.get("bad_behaviour") or {}).get("card")) or {}).get("name")
    if card:
        return "card"
    if t == "dribble":
        return "skill"
    if t in ("goal keeper", "goalkeeper"):
        gk = str((((e.get("goalkeeper") or {}).get("type")) or {})
                 .get("name", "")).lower()
        if "save" in gk or "smother" in gk or "punch" in gk:
            return "save"
    return None


def _parse_statsbomb(rows: list) -> list[MatchEvent]:
    out: list[MatchEvent] = []
    for e in rows:
        if not isinstance(e, dict):
            continue
        tname = str(((e.get("type") or {}).get("name")) or "")
        kind = _statsbomb_kind(tname, e)
        if kind is None:
            continue
        minute = _to_int(e.get("minute"))
        if minute is None:
            continue
        second = _to_float(e.get("second")) or 0.0
        period = _to_int(e.get("period")) or (2 if minute >= 45 else 1)
        team = _clean((e.get("team") or {}).get("name"))
        player = _clean((e.get("player") or {}).get("name"))
        label = tname
        if kind == "card":
            label = str((((e.get("foul_committed") or {}).get("card"))
                         or ((e.get("bad_behaviour") or {}).get("card"))
                         or {}).get("name", "Card"))
        out.append(MatchEvent(
            minute=minute, second=second, period=period, kind=kind,
            team=team, player=player,
            text=f"{minute}' {label}" + (f" — {player}" if player else "")))
    return out


_SOCCERNET_GT_RE = re.compile(r"\s*(\d)\s*-\s*(\d{1,3}):(\d{2})")


def _parse_soccernet(annotations: list) -> list[MatchEvent]:
    out: list[MatchEvent] = []
    for a in annotations:
        if not isinstance(a, dict):
            continue
        label = str(a.get("label", ""))
        kind = _map_kind(label)
        if kind is None:
            continue
        m = _SOCCERNET_GT_RE.match(str(a.get("gameTime", "")))
        if not m:
            continue
        period = int(m.group(1)) or 1
        mm, ss = int(m.group(2)), int(m.group(3))
        base = _PERIOD_START_MIN.get(period, 0)
        out.append(MatchEvent(
            minute=base + mm, second=float(ss), period=period, kind=kind,
            team=_clean(a.get("team")),
            text=f"{a.get('gameTime', '')} {label}".strip()))
    return out


def load_from_espn(fixture_id, slug: str = "esp.1", cfg: dict | None = None,
                   timeout: float = 30.0) -> list[MatchEvent]:
    """Fetch goals/cards (+minutes/scorers) from ESPN's PUBLIC, KEYLESS API.

    Compliance-clean (an open API, not a scrape). Best-effort: returns [] on any
    failure so the pipeline degrades to other sources. `slug` is ESPN's league
    code (e.g. esp.1 = La Liga, fifa.world = World Cup). `fixture_id` is the
    ESPN event id for the match.
    """
    try:
        import requests
    except Exception:  # noqa: BLE001
        log.warning("[event_feed] 'requests' unavailable; cannot fetch ESPN")
        return []
    url = (f"https://site.api.espn.com/apis/site/v2/sports/soccer/"
           f"{slug}/summary")
    try:
        r = requests.get(url, params={"event": str(fixture_id)}, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # noqa: BLE001
        log.warning(f"[event_feed] ESPN fetch failed ({slug}/{fixture_id}): {exc}")
        return []
    events = _espn_keyevents(data)
    log.info(f"[event_feed] ESPN {slug}/{fixture_id}: {len(events)} events "
             f"({sum(e.kind == 'goal' for e in events)} goals)")
    return events


def _espn_keyevents(data: dict) -> list[MatchEvent]:
    """Map ESPN summary 'keyEvents' to MatchEvents (goals + cards)."""
    out: list[MatchEvent] = []
    for ke in (data.get("keyEvents") or []):
        ttype = str(((ke.get("type") or {}).get("text")) or "")
        minute = _parse_minute((ke.get("clock") or {}).get("displayValue"))
        if minute is None:
            continue
        period = _to_int((ke.get("period") or {}).get("number"))
        if period is None:
            period = 2 if minute > 45 else 1
        team = ((ke.get("team") or {}).get("displayName")) or None
        parts = ke.get("participants") or [{}]
        player = ((parts[0].get("athlete") or {}).get("displayName")) or None
        text = (ke.get("text") or "").strip()
        if ke.get("scoringPlay") or ttype == "Goal":
            kind = "goal"
        elif "card" in ttype.lower():
            kind = "card"
        else:
            kind = _map_kind(ttype) or _map_kind(text)
        if kind is None:
            continue
        out.append(MatchEvent(
            minute=minute, kind=kind, period=period, team=team, player=player,
            importance=_default_importance(kind),
            text=text or f"{minute}' {kind}"))
    return out


def events_to_windows(events: list[MatchEvent], kickoffs: dict, cfg: dict,
                      duration: float | None = None) -> list[EventWindow]:
    """Map events to verified-by-feed EventWindows using per-period kick-off offsets.

    `kickoffs` maps period -> video-time (seconds) of that period's kick-off.
    Events whose period has no kick-off mapping are skipped (we can't place them).
    """
    sc = cfg.get("detect", {}).get("scout", {})
    ef = cfg.get("detect", {}).get("event_feed", {})
    kickoffs = {int(k): float(v) for k, v in (kickoffs or {}).items()}
    if not kickoffs:
        log.warning("[event_feed] no kick-off mapping; cannot place events on video")
        return []
    min_imp = float(ef.get("min_importance", 0.0))

    windows: list[EventWindow] = []
    placed = skipped = 0
    for ev in events:
        if ev.kind in _SKIP_KINDS or ev.importance < min_imp:
            skipped += 1
            continue
        ko = kickoffs.get(ev.period)
        if ko is None:                       # try nearest known period as fallback
            ko = kickoffs.get(1)
        if ko is None:
            skipped += 1
            continue
        anchor = ko + ev.match_seconds()
        if duration and (anchor < 0 or anchor > duration):
            skipped += 1
            continue
        start, end = _window_bounds(ev.kind, anchor, cfg, sc, duration)
        label = ev.text or f"{ev.minute}' {ev.kind}"
        windows.append(EventWindow(
            kind=ev.kind, anchor_t=anchor, start=start, end=end,
            confidence=float(min(1.0, max(0.3, ev.importance))),
            verified=(ev.kind == "goal"), minute=ev.minute,
            sources=["event_feed"],
            meta={"team": ev.team, "player": ev.player, "number": ev.number,
                  "label": label, "period": ev.period}))
        placed += 1

    windows.sort(key=lambda w: w.anchor_t)
    windows = _dedupe(windows, float(sc.get("merge_gap_seconds", 15.0)))
    # xT/importance ranking: keep only the top-N strongest moments (0 = all)
    top_n = int(ef.get("top_n", 0) or 0)
    if top_n and len(windows) > top_n:
        windows = sorted(windows, key=lambda w: w.confidence, reverse=True)[:top_n]
        windows.sort(key=lambda w: w.anchor_t)
        log.info(f"[event_feed] kept top {top_n} windows by importance")
    log.info(f"[event_feed] {placed} windows placed from feed "
             f"({skipped} skipped), {sum(w.kind == 'goal' for w in windows)} goals")
    return windows


def align_to_ocr(windows: list[EventWindow], ocr_sigs, cfg: dict
                 ) -> list[EventWindow]:
    """Best-effort: snap a feed goal window to the nearest scoreboard-OCR score
    change (which is frame-accurate), correcting kick-off-offset drift. Non-goal
    windows and unmatched goals are left as-is. Never raises."""
    try:
        radius = float(cfg.get("detect", {}).get("event_feed", {})
                       .get("ocr_align_radius_seconds", 40.0))
        used: set[int] = set()
        for w in windows:
            if w.kind != "goal" or not ocr_sigs:
                continue
            best, best_dt = None, radius
            for i, s in enumerate(ocr_sigs):
                if i in used:
                    continue
                dt = abs(s.t - w.anchor_t)
                if dt <= best_dt:
                    best, best_dt = i, dt
            if best is not None:
                used.add(best)
                shift = ocr_sigs[best].t - w.anchor_t
                w.anchor_t += shift
                w.start += shift
                w.end += shift
                w.sources = sorted(set(w.sources) | {"scoreboard_ocr"})
                osig = ocr_sigs[best]
                w.score_before = osig.meta.get("prev") or w.score_before
                w.score_after = osig.meta.get("score") or w.score_after
    except Exception as exc:  # noqa: BLE001
        log.warning(f"[event_feed] OCR alignment skipped: {exc}")
    return windows


# --------------------------------------------------------------------------- #
# parsing helpers
# --------------------------------------------------------------------------- #
def _coerce_rows(rows) -> list[MatchEvent]:
    out: list[MatchEvent] = []
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        g = {str(k).strip().lower(): v for k, v in r.items()}
        minute = _to_int(g.get("minute", g.get("min", g.get("time"))))
        if minute is None:
            continue
        raw_type = str(g.get("type", g.get("event", g.get("kind", "")))).strip()
        kind = _map_kind(raw_type) or _map_kind(str(g.get("text", "")))
        if kind is None:
            continue
        out.append(MatchEvent(
            minute=minute,
            second=_to_float(g.get("second", g.get("sec", 0.0))) or 0.0,
            period=_to_int(g.get("period", g.get("half", 1))) or 1,
            kind=kind,
            team=_clean(g.get("team")),
            player=_clean(g.get("player", g.get("name"))),
            number=_to_int(g.get("number", g.get("no", g.get("shirt")))),
            importance=_to_float(g.get("importance", g.get("xt", g.get("xg")))) or
            _default_importance(kind),
            text=_clean(g.get("text", g.get("description", ""))) or ""))
    return out


_LINE_RE = re.compile(r"^\s*(\d{1,3})(?:\s*\+\s*\d+)?\s*[':.\-)]*\s*(.*)$")


def _parse_text(blob: str) -> list[MatchEvent]:
    out: list[MatchEvent] = []
    for line in blob.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        minute = int(m.group(1))
        rest = m.group(2)
        kind = _map_kind(rest)
        if kind is None:
            continue
        period = 2 if minute > 45 else 1
        team = None
        tm = re.search(r"\(([^)]+)\)", rest)
        if tm:
            team = tm.group(1).strip()
        player = None
        pm = re.search(r"[—\-:]\s*([A-ZÁÉÍÓÚÑÜ][\w'.\-]+(?:\s+[A-ZÁÉÍÓÚÑÜ][\w'.\-]+)?)",
                       rest)
        if pm:
            player = pm.group(1).strip()
        num = re.search(r"#\s*(\d{1,2})", rest)
        out.append(MatchEvent(
            minute=minute, second=0.0, period=period, kind=kind,
            team=team, player=player,
            number=int(num.group(1)) if num else None,
            importance=_default_importance(kind), text=line))
    return out


def _map_kind(text: str) -> str | None:
    if not text:
        return None
    t = str(text).lower()
    # longest keyword first so "red card" beats "card"
    for kw in sorted(_KIND_KEYWORDS, key=len, reverse=True):
        if kw in t:
            return _KIND_KEYWORDS[kw]
    if any(s in t for s in _SKIP_KINDS):
        return None
    return None


def _default_importance(kind: str) -> float:
    return {"goal": 1.0, "save": 0.8, "card": 0.7, "skill": 0.7,
            "chance": 0.6}.get(kind, 0.5)


def _parse_minute(v):
    """Leading match minute from an ESPN/feed clock string ("45+2'" -> 45)."""
    if v is None:
        return None
    m = re.match(r"\s*(\d{1,3})", str(v))
    return int(m.group(1)) if m else None


def _to_int(v):
    try:
        if v is None or v == "":
            return None
        return int(float(str(v).replace("'", "").strip()))
    except (TypeError, ValueError):
        return None


def _to_float(v):
    try:
        if v is None or v == "":
            return None
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def _clean(v):
    if v is None:
        return None
    s = str(v).strip()
    return s or None
