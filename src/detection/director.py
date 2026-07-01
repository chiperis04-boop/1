"""Blueprint Module 2 — the Director (LLM editing manifest).

Watches a single event window (frames sampled at ~1 fps) and emits a strict JSON
manifest of *creative & editing* decisions the rest of the pipeline obeys:

    {
      "event_detected": "goal",
      "main_hero_yolo_cls": "player",
      "main_hero_description": "Player 18 in blue jersey",
      "video_hook_text": "NICO PAZ CANNOT BE STOPPED!",
      "slomo_trigger_timestamp": 12.5,
      "slomo_duration": 4.0
    }

Backends (configurable, all optional):
  * gemini   — google-generativeai, multimodal (frames + prompt)
  * openai   — any OpenAI-compatible chat/vision endpoint (e.g. a local
               MiniCPM-V-2_6 served via Ollama / vLLM / LM Studio). Set
               OPENAI_BASE_URL + OPENAI_API_KEY (or director.base_url in cfg).
  * heuristic — no network: derives a sensible manifest from the EventWindow +
                ball/player geometry. This is the default fallback so the studio
                works fully offline.

The model is *constrained* to the schema (response_mime_type=application/json on
Gemini, json_object on OpenAI) and the result is validated/coerced, so a
malformed LLM reply degrades to the heuristic instead of crashing the render.
"""
from __future__ import annotations

import base64
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from ..utils.io import downscale_max, get_logger

log = get_logger()

# Strict schema we ask the model to fill and then validate against.
_SCHEMA_KEYS = {
    "event_detected": str,
    "main_hero_yolo_cls": str,
    "main_hero_description": str,
    "video_hook_text": str,
    "slomo_trigger_timestamp": float,
    "slomo_duration": float,
}

_SYSTEM_PROMPT = (
    "You are the creative director of a viral football highlights channel. "
    "You are given frames sampled at 1 fps from a short clip around a key "
    "moment. Analyse the action and return ONLY a JSON object with these keys:\n"
    '  "event_detected": one of goal|chance|save|skill|card,\n'
    '  "main_hero_yolo_cls": one of player|goalkeeper|ball,\n'
    '  "main_hero_description": short visual description to re-identify the hero '
    '(jersey colour + number if visible),\n'
    '  "video_hook_text": an ALL-CAPS punchy on-screen hook (max 7 words),\n'
    '  "slomo_trigger_timestamp": seconds from clip start where slow-mo should '
    "begin (the decisive beat: the shot/save/skill),\n"
    '  "slomo_duration": slow-mo length in seconds (2.0-5.0).\n'
    "Return strictly valid JSON, no commentary, no markdown fences."
)


@dataclass
class EditingManifest:
    event_detected: str = "goal"
    main_hero_yolo_cls: str = "player"
    main_hero_description: str = ""
    video_hook_text: str = ""
    slomo_trigger_timestamp: float = 0.0
    slomo_duration: float = 3.0
    source: str = "heuristic"          # which backend produced this

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def coerce(cls, data: dict, source: str) -> "EditingManifest":
        """Validate + coerce a raw dict to the schema; missing/garbage -> default."""
        m = cls(source=source)
        for key, typ in _SCHEMA_KEYS.items():
            if key not in data:
                continue
            try:
                setattr(m, key, typ(data[key]))
            except (TypeError, ValueError):
                log.warning(f"[director] bad value for '{key}': {data[key]!r}")
        m.event_detected = (m.event_detected or "goal").lower().strip()
        m.main_hero_yolo_cls = (m.main_hero_yolo_cls or "player").lower().strip()
        m.slomo_duration = float(min(max(m.slomo_duration, 1.5), 6.0))
        m.slomo_trigger_timestamp = max(0.0, float(m.slomo_trigger_timestamp))
        return m


# --------------------------------------------------------------------------- #
def generate_manifest(clip_path: str, window=None, cfg: dict | None = None,
                      track=None) -> EditingManifest:
    """Produce an editing manifest for one event clip.

    `window` is the EventWindow (for kind/score context), `track` is an optional
    TrackResult used to enrich the heuristic fallback (hero description, slow-mo
    timing from the ball's fastest moment).
    """
    cfg = cfg or {}
    d = cfg.get("director", {})
    backend = (d.get("backend") or "heuristic").lower()

    if backend in ("none", "off", "disabled"):
        return _heuristic(clip_path, window, track, cfg)

    try:
        frames = _sample_frames(clip_path, fps=float(d.get("sample_fps", 1.0)),
                                max_frames=int(d.get("max_frames", 16)))
        if not frames:
            raise RuntimeError("no frames sampled from clip")
        ctx = _context_hint(window)
        if backend == "gemini":
            raw = _call_gemini(frames, ctx, d)
        elif backend in ("openai", "minicpm", "ollama", "vllm"):
            raw = _call_openai(frames, ctx, d)
        else:
            raise NotImplementedError(f"director backend '{backend}'")
        m = EditingManifest.coerce(raw, source=backend)
        # never trust the model on event type when OCR already verified a goal
        if window is not None and getattr(window, "verified", False):
            m.event_detected = "goal"
        log.info(f"[director] manifest via {backend}: "
                 f"hero='{m.main_hero_description}' hook='{m.video_hook_text}'")
        return m
    except Exception as exc:  # noqa: BLE001
        log.warning(f"[director] backend '{backend}' failed ({exc}); "
                    f"using heuristic manifest")
        return _heuristic(clip_path, window, track, cfg)


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
def _call_gemini(frames: list[bytes], ctx: str, d: dict) -> dict:
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY/GOOGLE_API_KEY not set")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        d.get("model", "gemini-1.5-flash"),
        system_instruction=_SYSTEM_PROMPT,
        generation_config={"response_mime_type": "application/json",
                           "temperature": float(d.get("temperature", 0.4))},
    )
    parts = [ctx]
    for fb in frames:
        parts.append({"mime_type": "image/jpeg", "data": fb})
    resp = model.generate_content(parts)
    return _parse_json(resp.text)


def _call_openai(frames: list[bytes], ctx: str, d: dict) -> dict:
    """OpenAI-compatible chat/vision call. Works against api.openai.com or any
    local server that speaks the same API (Ollama, vLLM, LM Studio serving
    MiniCPM-V-2_6)."""
    from openai import OpenAI

    client = OpenAI(
        base_url=d.get("base_url") or os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY", "not-needed-for-local"),
    )
    content = [{"type": "text", "text": ctx}]
    for fb in frames:
        b64 = base64.b64encode(fb).decode("ascii")
        content.append({"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    resp = client.chat.completions.create(
        model=d.get("model", "minicpm-v"),
        messages=[{"role": "system", "content": _SYSTEM_PROMPT},
                  {"role": "user", "content": content}],
        response_format={"type": "json_object"},
        temperature=float(d.get("temperature", 0.4)),
    )
    return _parse_json(resp.choices[0].message.content)


# --------------------------------------------------------------------------- #
# heuristic fallback (no network)
# --------------------------------------------------------------------------- #
def _heuristic(clip_path, window, track, cfg) -> EditingManifest:
    kind = getattr(window, "kind", "goal") if window else "goal"
    m = EditingManifest(event_detected=kind, source="heuristic")

    # hero description + slow-mo timing from tracking geometry when available
    trigger, hero = None, ""
    if track is not None and getattr(track, "frames", None):
        trigger = _peak_ball_speed_time(track)
        hero = _describe_hero(track)
    if trigger is None:
        # decisive beat ~= just before the anchor inside the window
        if window is not None:
            trigger = max(0.0, (window.anchor_t - window.start) - 0.5)
        else:
            from ..edit import ff
            trigger = max(0.0, ff.duration(clip_path) * 0.6)

    m.slomo_trigger_timestamp = float(trigger)
    m.slomo_duration = float(cfg.get("edit", {}).get("effects", {})
                             .get("slowmo_window", 3.0))
    m.main_hero_description = hero or "the player on the ball"
    m.main_hero_yolo_cls = "player"
    m.video_hook_text = _default_hook(kind, window, cfg)
    return m


def _default_hook(kind: str, window, cfg) -> str:
    brand_hooks = (cfg.get("director", {}).get("hooks", {}) or {})
    if kind in brand_hooks and brand_hooks[kind]:
        return str(brand_hooks[kind][0]).upper()
    meta = (getattr(window, "meta", {}) or {}) if window is not None else {}
    energy = float(meta.get("energy", 0.0) or 0.0)
    player = meta.get("player")
    score = ""
    if window is not None and getattr(window, "score_after", None):
        score = f" {window.score_after}"
    # high-energy skill move -> punchy, context-aware hook (player from the log,
    # so it is data-supported — not an invented name). The VLM Director writes
    # the richer creative hook when configured; this is the offline fallback.
    if kind == "skill":
        if player and energy >= 0.5:
            return f"{str(player).upper()} IN HACKER MODE"
        return "HACKER MODE" if energy >= 0.5 else "OUTRAGEOUS SKILL!"
    defaults = {
        "goal": f"WHAT A GOAL!{score}",
        "chance": "SO CLOSE!",
        "save": "INCREDIBLE SAVE!",
        "skill": "OUTRAGEOUS SKILL!",
        "card": "STRAIGHT RED?!",
    }
    return defaults.get(kind, "WATCH THIS!")


def _peak_ball_speed_time(track) -> float | None:
    """Time (s) of the fastest ball movement — the decisive beat for slow-mo.

    Works with either a TrackResult (has `ball_path`) or a CropPlan / object
    exposing `frames` with per-frame `ball` centers.
    """
    fps = getattr(track, "fps", 30.0) or 30.0
    path = getattr(track, "ball_path", None)
    if not path:
        frames = getattr(track, "frames", None) or []
        path = [(ft.idx, ft.ball["center"][0], ft.ball["center"][1])
                for ft in frames if getattr(ft, "ball", None)]
    if not path or len(path) < 3:
        return None
    best_idx, best_v = path[0][0], 0.0
    for (i0, x0, y0), (i1, x1, y1) in zip(path, path[1:]):
        dt = max(1, i1 - i0)
        v = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / dt
        if v > best_v:
            best_v, best_idx = v, i0
    return best_idx / fps


def _describe_hero(track) -> str:
    kid = getattr(track, "key_track_id", None)
    if kid is None:
        kid = getattr(track, "hero_id", None)
    if kid is None:
        return ""
    return f"player (track #{kid}) nearest the ball at the decisive moment"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _sample_frames(clip_path: str, fps: float, max_frames: int) -> list[bytes]:
    """Return JPEG-encoded frames sampled at ~`fps` from the clip."""
    import cv2

    cap = cv2.VideoCapture(clip_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / max(0.1, fps))))
    frames: list[bytes] = []
    idx = 0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frame = downscale_max(frame, 768)   # cap VLM image tokens (ctx limit)
            ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok2:
                frames.append(buf.tobytes())
        idx += 1
    cap.release()
    return frames


def _context_hint(window) -> str:
    if window is None:
        return "Analyse this football clip and return the editing manifest JSON."
    bits = [f"Detector thinks this is a '{window.kind}'."]
    if getattr(window, "verified", False) and window.score_after:
        bits.append(f"Scoreboard confirms the score changed "
                    f"{window.score_before} -> {window.score_after}.")
    if getattr(window, "minute", None):
        bits.append(f"Match minute ~{window.minute}'.")
    bits.append("Clip starts a few seconds before the event. "
                "Return the editing manifest JSON.")
    return " ".join(bits)


def _parse_json(text: str) -> dict:
    if not text:
        raise ValueError("empty model response")
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return json.loads(text)
