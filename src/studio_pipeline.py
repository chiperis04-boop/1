"""Studio pipeline (v2) — the blueprint's 4-module chain end to end.

    full match
       │  ingest (probe + audio + proxy)             [src/ingest.py]
       ▼
    Scout      scout_events()                        [src/detection/scout.py]
       │  -> verified EventWindows
       ▼
    clip       extract_clips()                       [src/edit/clipper.py]
       │  one short clip per event
       ▼ (per clip)
    Cameraman  track() -> CropPlan (BoT-SORT + CMC)  [src/tracking/cameraman.py]
    Director   generate_manifest() (LLM/heuristic)   [src/detection/director.py]
    Homography compute_homography() (optional)       [src/graphics/homography.py]
    Composer   draw_graphics() on ORIGINAL space     [src/render/composer.py]
       │       -> Cameraman.render() crop to 9:16
       │       -> Composer.finish() slow-mo + text
       ▼
    finished 9:16 highlight (+ manifest.json + caption.txt)

Coordinate-space note: graphics (halo/trail) are computed in the original clip's
pixels, so they are drawn BEFORE the crop-to-9:16; slow-mo and typography are
full-frame and run after. Each per-clip stage is isolated: a failure is recorded
and the batch continues.

This sits ALONGSIDE the existing src/runner.py (v1) — it does not replace it.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .detect.types import Moment
from .agents.director_agent import plan_edit
from .detection.scout import EventWindow, scout_events
from .edit import ff
from .edit.clipper import extract_clips
from .graphics.homography import compute_homography
from .ingest import ingest
from .perception.bundle import build_bundle
from .render.composer import Composer
from .tracking.cameraman import Cameraman
from .utils.io import (ensure_dir, get_logger, load_branding, load_config,
                       save_json)
from .vision.analytics import analyze as analyze_clip

log = get_logger()


@dataclass
class StudioClip:
    index: int
    kind: str
    anchor_t: float
    confidence: float
    verified: bool
    status: str = "pending"            # pending|ok|failed
    path: str | None = None
    manifest: dict | None = None
    hero_id: int | None = None
    hero_number: int | None = None
    hero_source: str | None = None
    possession_pct: dict | None = None
    shots: int = 0
    replays: int = 0
    qa_score: float | None = None
    qa_issues: list | None = None
    revisions: int = 0
    stage_failed: str | None = None
    error: str | None = None


@dataclass
class StudioResult:
    match: str
    profile: str
    out_dir: str
    status: str = "ok"                 # ok|failed|empty
    windows: int = 0
    goals: int = 0
    clips: list[StudioClip] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


ProgressCb = Callable[[str, float, str], None]


def run_studio(
    match: str,
    profile: str = "tiktok",
    config: str = "config/config.yaml",
    branding: str = "config/branding.yaml",
    out_root: str = "output",
    limit: int = 0,
    on_progress: ProgressCb | None = None,
    overrides: dict | None = None,
) -> StudioResult:
    def emit(stage, pct, msg=""):
        if on_progress:
            try:
                on_progress(stage, round(pct, 1), msg)
            except Exception:  # a UI callback must never break the run
                pass

    try:
        ff.ensure_tools()
    except ff.FFmpegError as e:
        return StudioResult(match=match, profile=profile, out_dir="",
                            status="failed", error=str(e))

    cfg = load_config(config)
    brand = load_branding(branding)
    if overrides:
        _deep_merge(cfg, overrides)
    if profile not in cfg["render"]["profiles"]:
        return StudioResult(match=match, profile=profile, out_dir="",
                            status="failed", error=f"unknown profile '{profile}'")
    cfg["_active_profile"] = cfg["render"]["profiles"][profile]
    cfg["render"]["encoder"] = ff.pick_encoder(cfg["render"]["encoder"])

    # models (player/ball always; pitch when graphics homography is on)
    try:
        from .modelhub import ensure_models
        emit("models", 1, "ensuring models")
        ensure_models(cfg, include_pitch=_graphics_on(cfg),
                      include_seg=cfg.get("telestration", {}).get("occlusion", False))
    except Exception as exc:  # noqa: BLE001
        log.warning(f"[studio] model ensure failed (continuing): {exc}")

    name = Path(match).stem
    workdir = ensure_dir(Path(out_root) / name / "work")
    out_dir = ensure_dir(Path(out_root) / name)
    result = StudioResult(match=match, profile=profile, out_dir=str(out_dir))

    # 1) ingest --------------------------------------------------------------
    emit("ingest", 4, "probe + audio + proxy")
    info = ingest(match, workdir, cfg)

    # 2) scout ---------------------------------------------------------------
    emit("scout", 12, "discovering + verifying events")
    windows = scout_events(info.src_path, info.proxy_path, cfg, info.duration)
    if limit:
        windows = sorted(windows, key=lambda w: w.confidence, reverse=True)[:limit]
        windows.sort(key=lambda w: w.anchor_t)
    result.windows = len(windows)
    result.goals = sum(w.kind == "goal" for w in windows)
    if not windows:
        result.status = "empty"
        emit("done", 100, "no events found")
        return result

    # 3) clip ----------------------------------------------------------------
    emit("clip", 22, f"cutting {len(windows)} clips")
    moments = [_window_to_moment(w) for w in windows]
    clips = extract_clips(info.src_path, moments, workdir, cfg)

    # 4) per-clip studio chain ----------------------------------------------
    cam = Cameraman(cfg)                    # one instance -> YOLO loaded once
    composer = Composer(cfg, brand)
    n = len(clips)
    for i, (w, clip) in enumerate(zip(windows, clips)):
        base = 28 + (i / max(1, n)) * 68
        emit("render", base, f"{w.kind} @ {w.anchor_t:.0f}s ({i + 1}/{n})")
        sc = StudioClip(index=i, kind=w.kind, anchor_t=w.anchor_t,
                        confidence=w.confidence, verified=w.verified)
        _process(i, w, clip, cfg, brand, out_dir, cam, composer, sc)
        result.clips.append(sc)

    ok = sum(c.status == "ok" for c in result.clips)
    skipped = sum(c.status == "skipped" for c in result.clips)
    result.status = "ok" if (ok or skipped) else "failed"
    save_json(result.to_dict(), Path(out_dir) / "studio_result.json")
    emit("done", 100, f"finished {ok}/{n} highlights"
         + (f", {skipped} dropped by Director" if skipped else ""))
    return result


# --------------------------------------------------------------------------- #
def _process(i, w: EventWindow, clip: str, cfg, brand, out_dir, cam: Cameraman,
             composer: Composer, sc: StudioClip):
    stage = "init"
    try:
        work = ensure_dir(Path(out_dir) / "work" / f"clip_{i:02d}")

        stage = "shots"                     # segment broadcast cuts in this clip
        from .perception.shots import mark_duplicate_shots, segment_shots
        shots = mark_duplicate_shots(clip, segment_shots(clip, cfg), cfg)
        sc.shots = len(shots)
        sc.replays = sum(s.is_replay for s in shots)

        stage = "track"                     # BoT-SORT + CMC (once)
        frames, meta = cam.track_only(clip)
        # geometric hero + crop reset per shot (no glide across cuts)
        plan0 = cam.build_plan(frames, meta, shots=shots)

        stage = "director"                  # frame-aware EditPlan (VLM or heuristic)
        bundle = build_bundle(clip, cfg, shots=shots, frames=frames, window=w)
        plan = plan_edit(bundle, window=w, cfg=cfg, track=plan0)
        manifest = plan.to_manifest()       # bridge for graphics/typography
        sc.manifest = plan.to_dict()
        if not plan.keep_clip and cfg.get("director", {}).get("allow_curation", True):
            sc.status = "skipped"
            log.info(f"[studio] clip {i} dropped by Director (not a highlight)")
            return

        # P3: snap the plan's slow-mo onset / cut-in to the music beat
        if cfg.get("edit", {}).get("audio", {}).get("beat_sync", False):
            mtrack = _first_music(cfg)
            if mtrack:
                try:
                    from .edit.music import beat_align_plan
                    plan, _ = beat_align_plan(plan, mtrack, cfg)
                except Exception as exc:  # noqa: BLE001
                    log.warning(f"[studio] beat-sync skipped: {exc}")

        stage = "homography"                # optional grass-anchor + metric calib
        homography = compute_homography(clip, cfg) if _graphics_on(cfg) else None
        calib = getattr(homography, "calibration", None) if homography else None

        stage = "analytics"                 # teams + jersey numbers + possession
        analytics = analyze_clip(clip, plan0, calib=calib, cfg=cfg,
                                 manifest=manifest, geometric_hero=plan0.hero_id)
        sc.hero_id = analytics.hero_id
        sc.hero_number = analytics.hero_number
        sc.hero_source = analytics.hero_source
        sc.possession_pct = analytics.possession_share_pct()

        stage = "reid"                      # cross-shot hero Re-ID (follow across cuts)
        hero_ids = None
        if cfg.get("reid", {}).get("enabled", True) and shots and len(shots) > 1 \
                and analytics.hero_id is not None:
            try:
                from .vision.reid import (build_shot_track_embeddings,
                                          cross_shot_hero_map, per_frame_hero)
                embs = build_shot_track_embeddings(clip, frames, shots, cfg)
                hsh = _hero_shot(frames, shots, analytics.hero_id)
                hmap = cross_shot_hero_map(analytics.hero_id, embs, hsh,
                                           float(cfg.get("reid", {}).get("min_sim", 0.5)))
                hero_ids = per_frame_hero(len(frames), shots, hmap, analytics.hero_id)
            except Exception as exc:  # noqa: BLE001
                log.warning(f"[studio] reid skipped: {exc}")

        stage = "render+review"             # render plan -> QA (+critic) -> revise
        stats = _stats_from(analytics)
        tele_on = cfg.get("telestration", {}).get("enabled", True)
        # reaction/celebration windows (clip seconds) for stat-card gating:
        # broadcast replay inserts are natural reaction cuts.
        reaction = [(s.start, s.end) for s in shots if getattr(s, "is_replay", False)]
        counter = {"n": 0}

        def render_plan(p) -> str:
            counter["n"] += 1
            k = counter["n"]
            cplan = cam.build_plan(frames, meta, hero_id=analytics.hero_id,
                                   shots=shots, shot_edits=p.shots, hero_ids=hero_ids)
            world = screen = None
            if tele_on:
                world, screen = composer.make_annotators(cplan, p.to_manifest(),
                                                         analytics)
            rf = cam.render(clip, cplan, str(work / f"reframed_{k}.mp4"),
                            annotate_world=world, annotate_screen=screen,
                            intermediate=True)
            outp = str(work / f"render_{k}.mp4")
            composer.finish(rf, outp, manifest=p.to_manifest(), stats=stats,
                            beats=p.slowmo_beats, reaction=reaction)
            return outp

        final = str(Path(out_dir) / f"{i:02d}_{w.kind}.mp4")
        qcfg = cfg.get("qa", {})
        if qcfg.get("enabled", True):
            from .agents.critic import critique
            from .agents.llm_client import VisionLLMClient
            from .agents.review import review_and_revise
            from .qa.checks import qa_report
            prof = cfg["_active_profile"]
            expected = {"width": prof["width"], "height": prof["height"]}
            if qcfg.get("max_seconds"):
                expected["max_seconds"] = qcfg["max_seconds"]
            if qcfg.get("min_seconds"):
                expected["min_seconds"] = qcfg["min_seconds"]
            critic_fn = None
            if qcfg.get("use_critic", True) and VisionLLMClient(cfg).is_configured():
                critic_fn = lambda path, pl: critique(path, pl, cfg)  # noqa: E731
            res = review_and_revise(
                plan, render_fn=render_plan,
                qa_fn=lambda path: qa_report(path, cfg, expected),
                cfg=cfg, critic_fn=critic_fn,
                max_revisions=int(qcfg.get("max_revisions", 1)))
            os.replace(res.path, final)
            sc.qa_score = res.score
            sc.qa_issues = list(getattr(res.qa, "issues", []) or [])
            sc.revisions = max(0, res.attempts - 1)
            log.info(f"[studio] clip {i} QA score {res.score:.2f} "
                     f"({sc.revisions} revision(s)); issues={sc.qa_issues}")
        else:
            os.replace(render_plan(plan), final)

        stage = "branding"                  # reuse v1 intro/outro/lower-thirds
        try:
            from .branding.overlays import apply_branding
            moment = _window_to_moment(w)
            branded = str(Path(out_dir) / f"{i:02d}_{w.kind}_branded.mp4")
            final = apply_branding(final, moment, {}, branded, cfg, brand,
                                   with_intro_outro=True, composer_typography=True)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"[studio] branding skipped for clip {i}: {exc}")

        _write_caption(final, w, manifest, brand)
        sc.status = "ok"
        sc.path = final
        log.info(f"[studio] clip {i} ok -> {Path(final).name} "
                 f"(hero #{sc.hero_number} via {sc.hero_source})")
    except Exception as exc:  # noqa: BLE001
        sc.status = "failed"
        sc.stage_failed = stage
        sc.error = str(exc)
        log.error(f"[studio] clip {i} failed at {stage}: {exc}")


def _stats_from(analytics) -> dict | None:
    """Build the data-plate stats dict from analytics (possession share, etc.)."""
    stats: dict = {}
    share = analytics.possession_share_pct()
    if share:
        # ordered by team label for a stable "60% / 40%" plate
        stats["possession_pct"] = [share[t] for t in sorted(share)]
    return stats or None


def _window_to_moment(w: EventWindow) -> Moment:
    return Moment(t=w.anchor_t, start=w.start, end=w.end, confidence=w.confidence,
                  kind=w.kind, minute=w.minute, sources=list(w.sources),
                  meta={"verified": w.verified, "score_before": w.score_before,
                        "score_after": w.score_after})


def _write_caption(path: str, w: EventWindow, manifest, brand: dict):
    ce = brand.get("caption_export", {})
    hashtags = " ".join(ce.get("hashtags", []))
    hook = (getattr(manifest, "video_hook_text", "") or w.kind.upper())
    text = ce.get("template", "{moment_type} {hook}\n{hashtags}").format(
        moment_type=w.kind.upper(), hook=hook, hashtags=hashtags)
    Path(path).with_suffix(".txt").write_text(text, encoding="utf-8")


def _graphics_on(cfg: dict) -> bool:
    return bool(cfg.get("vision", {}).get("pitch", {}).get("enabled", False)
                or cfg.get("graphics", {}).get("enabled", False))


def _first_music(cfg: dict) -> str | None:
    """Peek the first music track (for beat-sync) without advancing anything."""
    from pathlib import Path as _P
    d = _P(cfg.get("edit", {}).get("audio", {}).get("music_dir", "assets/music"))
    if not d.exists():
        return None
    tracks = sorted(str(p) for p in d.glob("*")
                    if p.suffix.lower() in {".mp3", ".wav", ".m4a", ".aac"})
    return tracks[0] if tracks else None


def _hero_shot(frames, shots, hero_id):
    """Index of the shot where the hero track id appears most (for Re-ID ref)."""
    from .perception.shots import frame_segments
    segs = frame_segments(shots, len(frames))
    best_shot, best_count = None, -1
    for si, (a, b) in enumerate(segs):
        c = sum(1 for fi in range(a, min(b, len(frames)))
                if any(p["id"] == hero_id for p in getattr(frames[fi], "players", [])))
        if c > best_count:
            best_shot, best_count = si, c
    return best_shot


def _deep_merge(base: dict, over: dict):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
