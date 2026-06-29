"""Importable pipeline API shared by the CLI and the WebUI.

`run_pipeline(...)` runs the full chain and reports progress through an optional
callback, returning a structured `RunResult`. Every per-clip failure is isolated
and recorded (which stage failed + the error) instead of aborting the batch.

    from src.runner import run_pipeline
    result = run_pipeline("input/match.mp4", profile="tiktok",
                          on_progress=lambda p: print(p.stage, p.percent))
    for clip in result.clips:
        print(clip.status, clip.path)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .branding.overlays import apply_branding
from .detect.audio_energy import detect_audio
from .detect.commentary import detect_commentary
from .detect.fusion import fuse
from .detect.scene import detect_scenes
from .detect.scoreboard_ocr import detect_scoreboard
from .detect.types import Moment
from .edit import ff
from .edit.captions import caption_clip
from .edit.clipper import extract_clips
from .edit.compose import compose_clip
from .edit.effects import apply_slowmo, freeze_zoom_intro
from .edit.reframe import reframe_clip
from .ingest import ingest
from .utils.io import (ensure_dir, get_logger, load_branding, load_config,
                       load_json, save_json)

log = get_logger()


# --------------------------------------------------------------------------- #
# data types
# --------------------------------------------------------------------------- #
@dataclass
class Progress:
    stage: str               # ingest|detect|fuse|clip|render|done
    percent: float           # 0..100 overall
    message: str = ""
    clip_index: int | None = None
    clip_total: int | None = None


@dataclass
class ClipResult:
    index: int
    kind: str
    confidence: float
    t: float
    minute: int | None
    status: str = "pending"      # pending|ok|failed|skipped
    path: str | None = None
    caption_path: str | None = None
    error: str | None = None
    stage_failed: str | None = None


@dataclass
class RunResult:
    match: str
    profile: str
    out_dir: str
    status: str = "ok"           # ok|failed|empty
    mode: str = "per_clip"       # per_clip|compilation
    moments: int = 0
    goals: int = 0
    clips: list[ClipResult] = field(default_factory=list)
    reel_path: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


ProgressCb = Callable[[Progress], None]


# --------------------------------------------------------------------------- #
def run_pipeline(
    match: str,
    profile: str = "tiktok",
    config: str = "config/config.yaml",
    branding: str = "config/branding.yaml",
    out_root: str = "output",
    limit: int = 0,
    on_progress: ProgressCb | None = None,
    overrides: dict | None = None,
) -> RunResult:
    """Run the full pipeline. `overrides` is a shallow dict merged into cfg
    (e.g. {'vision': {'enabled': False}}) so the UI can toggle features."""
    def emit(stage, pct, msg="", **kw):
        if on_progress:
            try:
                on_progress(Progress(stage=stage, percent=round(pct, 1),
                                     message=msg, **kw))
            except Exception:  # a UI callback must never break the pipeline
                pass

    try:
        ff.ensure_tools()
    except ff.FFmpegError as e:
        return RunResult(match=match, profile=profile, out_dir="",
                         status="failed", error=str(e))

    cfg = load_config(config)
    brand = load_branding(branding)
    if overrides:
        _deep_merge(cfg, overrides)
    if profile not in cfg["render"]["profiles"]:
        return RunResult(match=match, profile=profile, out_dir="",
                         status="failed",
                         error=f"unknown profile '{profile}'")
    cfg["_active_profile"] = cfg["render"]["profiles"][profile]
    cfg["render"]["encoder"] = ff.pick_encoder(cfg["render"]["encoder"])

    # ensure the football models are present (auto-download, idempotent) so
    # telestration works on upload with no manual setup
    if cfg.get("vision", {}).get("enabled"):
        try:
            from .modelhub import ensure_models
            emit("models", 1, "checking / downloading models")
            ensure_models(cfg)
        except Exception as exc:  # noqa: BLE001  (never block on model fetch)
            log.warning(f"[models] ensure failed (will try generic): {exc}")

    name = Path(match).stem
    workdir = ensure_dir(Path(out_root) / name / "work")
    out_dir = ensure_dir(Path(out_root) / name)
    result = RunResult(match=match, profile=profile, out_dir=str(out_dir))

    # 1) ingest -------------------------------------------------------------
    emit("ingest", 2, "probing + extracting audio/proxy")
    info = ingest(match, workdir, cfg)

    # 2/3) detect + fuse ----------------------------------------------------
    emit("detect", 8, "scanning match for key moments")
    moments = _detect_and_fuse(info, cfg, workdir, emit)
    if limit:
        moments = sorted(moments, key=lambda m: m.confidence, reverse=True)[:limit]
        moments.sort(key=lambda m: m.t)
    result.moments = len(moments)
    result.goals = sum(m.kind == "goal" for m in moments)
    if not moments:
        result.status = "empty"
        emit("done", 100, "no moments detected — try lowering thresholds")
        return result

    # 4) cut clips ----------------------------------------------------------
    emit("clip", 30, f"cutting {len(moments)} clips")
    clips = extract_clips(info.src_path, moments, workdir, cfg)

    mode = cfg["render"].get("output_mode", "per_clip")
    result.mode = mode
    if mode == "compilation":
        _run_compilation(moments, clips, cfg, brand, out_dir, result, emit)
    else:
        _run_per_clip(moments, clips, cfg, brand, out_dir, result, emit)

    save_json(result.to_dict(), Path(out_dir) / "result.json")
    return result


def _run_per_clip(moments, clips, cfg, brand, out_dir, result, emit):
    n = len(clips)
    for i, (m, clip) in enumerate(zip(moments, clips)):
        base = 35 + (i / max(1, n)) * 60
        emit("render", base, f"{m.kind} (conf {m.confidence})",
             clip_index=i + 1, clip_total=n)
        cr = ClipResult(index=i, kind=m.kind, confidence=m.confidence,
                        t=m.t, minute=m.minute)
        _process_clip(i, m, clip, cfg, brand, out_dir, cr)
        if cr.status == "ok":
            _write_caption(cr, m, brand)
        result.clips.append(cr)
    ok = sum(c.status == "ok" for c in result.clips)
    result.status = "ok" if ok else "failed"
    emit("done", 100, f"rendered {ok}/{n} clips")


def _run_compilation(moments, clips, cfg, brand, out_dir, result, emit):
    """Render each moment as a segment (no per-segment music/intro/outro), then
    select enough to hit the target length and stitch into one reel."""
    from .edit.compilation import build_compilation, select_for_duration

    n = len(clips)
    items = []
    for i, (m, clip) in enumerate(zip(moments, clips)):
        base = 35 + (i / max(1, n)) * 50
        emit("render", base, f"segment {m.kind} (conf {m.confidence})",
             clip_index=i + 1, clip_total=n)
        cr = ClipResult(index=i, kind=m.kind, confidence=m.confidence,
                        t=m.t, minute=m.minute)
        _process_clip(i, m, clip, cfg, brand, out_dir, cr, segment_mode=True)
        result.clips.append(cr)
        if cr.status == "ok":
            items.append({"path": cr.path, "duration": ff.duration(cr.path),
                          "confidence": m.confidence, "order": i, "kind": m.kind})

    if not items:
        result.status = "failed"
        emit("done", 100, "no segments rendered")
        return

    comp = cfg["render"]["compilation"]
    chosen = select_for_duration(items, comp["target_seconds"],
                                 comp["max_seconds"], 0)
    emit("render", 92, f"assembling reel from {len(chosen)} segments")
    reel = str(Path(out_dir) / "reel_01.mp4")
    try:
        build_compilation([c["path"] for c in chosen], reel, cfg, brand)
        result.reel_path = reel
        result.status = "ok"
        _write_reel_caption(reel, chosen, brand)
        emit("done", 100, f"reel ready ({ff.duration(reel):.0f}s)")
    except Exception as exc:  # noqa: BLE001
        result.status = "failed"
        result.error = f"compilation failed: {exc}"
        emit("done", 100, result.error)


# --------------------------------------------------------------------------- #
def _detect_and_fuse(info, cfg, workdir, emit) -> list[Moment]:
    state = Path(workdir) / "moments.json"
    if state.exists():
        log.info("[detect] using cached moments.json")
        return [Moment(**m) for m in load_json(state)]

    signals = []
    emit("detect", 10, "audio energy")
    signals += detect_audio(info.audio_path, cfg)
    emit("detect", 15, "scene / replays")
    signals += detect_scenes(info.proxy_path, cfg)
    emit("detect", 20, "scoreboard OCR")
    signals += detect_scoreboard(info.proxy_path, cfg)
    emit("detect", 26, "commentary")
    signals += detect_commentary(info.audio_path, cfg)
    if cfg.get("detect", {}).get("action_spotting", {}).get("enabled"):
        emit("detect", 28, "action spotting")
        from .detect.action_spotting import detect_actions
        signals += detect_actions(info.src_path, cfg)

    moments = fuse(signals, cfg, info.duration)
    save_json([m.__dict__ for m in moments], state)
    return moments


def _process_clip(i, m: Moment, clip: str, cfg, brand, out_dir, cr: ClipResult,
                  segment_mode: bool = False):
    stage = "init"
    try:
        work = ensure_dir(Path(out_dir) / "work" / f"clip_{i:02d}")
        key_t = max(0.0, m.t - m.start)
        cur = clip
        stats = {}

        if cfg["vision"]["enabled"]:
            from .vision.detect_track import track_clip
            from .vision.stats import compute_stats
            from .vision.telestration import render_telestration

            stage = "track"
            track = track_clip(clip, cfg)

            # de-jitter tracker coordinates so circles/traces/camera glide
            sm = cfg["vision"].get("smoothing", {})
            if sm.get("enabled", True):
                from .vision.smoothing import smooth_track
                smooth_track(track, method=sm.get("method", "savgol"),
                             window=int(sm.get("window", 9)),
                             poly=int(sm.get("poly", 2)))

            # L3: team classification -> possession-aware protagonist
            if cfg["vision"].get("teams", {}).get("enabled"):
                from .vision.teams import TeamClassifier, pick_key_player
                team_of = TeamClassifier(cfg).classify(clip, track)
                kp = pick_key_player(track, team_of)
                if kp is not None:
                    track.key_track_id = kp

            # L2: pitch homography -> metric stats + grass-plane graphics
            calib = None
            if cfg["vision"].get("pitch", {}).get("enabled"):
                stage = "calibrate"
                from .vision.pitch import PitchEstimator
                calib = PitchEstimator(cfg).calibrate(clip)

            stats = compute_stats(track, calib)
            stage = "telestrate"
            cur = render_telestration(clip, track, str(work / "telestrated.mp4"),
                                      cfg, calib=calib)
            stage = "reframe"
            cur = reframe_clip(cur, track, str(work / "reframed.mp4"), cfg)
        else:
            from .edit.reframe import _letterbox
            stage = "reframe"
            cur = _letterbox(clip, str(work / "reframed.mp4"), cfg)

        if cfg["edit"]["effects"]["slowmo_on_key"]:
            stage = "slowmo"
            cur = apply_slowmo(cur, key_t, str(work / "slowmo.mp4"), cfg)
        if cfg["edit"]["effects"]["freeze_zoom"]:
            stage = "freeze"
            cur = freeze_zoom_intro(cur, key_t, str(work / "freeze.mp4"), cfg)

        stage = "captions"
        caps = caption_clip(cur, cfg)
        stage = "compose"
        cur = compose_clip(cur, str(work / "composed.mp4"), cfg, brand, caps,
                           add_music=not segment_mode)
        stage = "branding"
        if segment_mode:
            seg_dir = ensure_dir(Path(out_dir) / "work" / "segments")
            final = str(seg_dir / f"seg_{i:02d}_{m.kind}.mp4")
        else:
            final = str(Path(out_dir) / f"{i:02d}_{m.kind}.mp4")
        cur = apply_branding(cur, m, stats, final, cfg, brand,
                             with_intro_outro=not segment_mode)

        cr.status = "ok"
        cr.path = cur
    except Exception as exc:  # noqa: BLE001
        cr.status = "failed"
        cr.stage_failed = stage
        cr.error = str(exc)
        log.error(f"[clip {i}] failed at {stage}: {exc}")


def _write_caption(cr: ClipResult, m: Moment, brand: dict):
    ce = brand.get("caption_export", {})
    hashtags = " ".join(ce.get("hashtags", []))
    hook_pool = brand.get("hook", {}).get("templates", {}).get(m.kind, [""])
    hook = hook_pool[0] if hook_pool else ""
    text = ce.get("template", "{moment_type} {hook}\n{hashtags}").format(
        moment_type=m.kind.upper(), hook=hook, hashtags=hashtags)
    p = Path(cr.path).with_suffix(".txt")
    p.write_text(text, encoding="utf-8")
    cr.caption_path = str(p)


def _write_reel_caption(reel_path: str, chosen: list[dict], brand: dict):
    ce = brand.get("caption_export", {})
    hashtags = " ".join(ce.get("hashtags", []))
    goals = sum(c["kind"] == "goal" for c in chosen)
    headline = (f"{goals} goals" if goals else f"{len(chosen)} moments") + " in 60s"
    hook_pool = brand.get("hook", {}).get("templates", {}).get("goal", [""])
    hook = hook_pool[0] if hook_pool else ""
    text = ce.get("template", "{moment_type} {hook}\n{hashtags}").format(
        moment_type=headline.upper(), hook=hook, hashtags=hashtags)
    Path(reel_path).with_suffix(".txt").write_text(text, encoding="utf-8")


def _deep_merge(base: dict, over: dict):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
