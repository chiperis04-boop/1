"""Shot segmentation — split a clip into continuous broadcast camera takes.

Why this matters (P0): a single event clip from a TV feed usually contains
several camera cuts (wide -> close-up -> crowd -> wide, plus the broadcast's own
slow-mo replay). The v2 reframe smoothed ONE crop path across the whole clip, so
the virtual camera lurched at every cut and the tracker's IDs reset. By
segmenting shots first, the Cameraman can plan/smooth each shot independently
(crop resets at the cut) and the orchestrator can drop replays.

Backed by PySceneDetect (already a dependency). If PySceneDetect or its OpenCV
backend is unavailable, we degrade gracefully to a single shot spanning the clip
so the pipeline never hard-fails on this stage.

Replay note: `mark_duplicate_shots` flags only *near-identical repeated* shots
(a conservative, literal-duplicate heuristic). Robust semantic replay detection
(different angle, slow-mo, wipe transition) is intentionally deferred to the
frame-aware Director agent (P1) — see the implementation plan.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..utils.io import get_logger

log = get_logger()


@dataclass
class Shot:
    idx: int
    start: float                 # seconds
    end: float                   # seconds
    start_frame: int
    end_frame: int               # exclusive
    kind: str = ""               # filled by the Director later (wide/closeup/...)
    is_replay: bool = False
    signature: list[float] | None = field(default=None, repr=False)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def n_frames(self) -> int:
        return max(0, self.end_frame - self.start_frame)


def _probe_fps_frames(clip_path: str) -> tuple[float, int]:
    """Return (fps, n_frames) using OpenCV (falls back to ffprobe duration)."""
    try:
        import cv2
        cap = cv2.VideoCapture(clip_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if fps > 0 and n > 0:
            return float(fps), n
    except Exception:  # noqa: BLE001
        pass
    from ..edit import ff
    dur = ff.duration(clip_path)
    fps = 30.0
    return fps, int(round(dur * fps))


def segment_shots(clip_path: str, cfg: dict | None = None) -> list[Shot]:
    """Detect broadcast camera cuts and return contiguous shots covering the clip.

    Config (cfg['shots']):
      enabled        : bool  (default True)  -> if False, returns a single shot
      threshold      : float (default 27.0)  -> ContentDetector sensitivity
      min_seconds    : float (default 0.6)   -> merge shorter shots into neighbour
    """
    cfg = cfg or {}
    s = cfg.get("shots", {}) if isinstance(cfg, dict) else {}
    fps, n_frames = _probe_fps_frames(clip_path)

    if not s.get("enabled", True):
        return [_whole(clip_path, fps, n_frames)]

    threshold = float(s.get("threshold", 27.0))
    min_seconds = float(s.get("min_seconds", 0.6))

    try:
        from scenedetect import ContentDetector, detect
        scenes = detect(clip_path, ContentDetector(threshold=threshold))
    except Exception as exc:  # noqa: BLE001  (missing lib, bad codec, etc.)
        log.warning(f"[shots] scenedetect unavailable/failed ({exc}); "
                    "treating clip as a single shot")
        return [_whole(clip_path, fps, n_frames)]

    if not scenes:
        return [_whole(clip_path, fps, n_frames)]

    raw: list[Shot] = []
    for i, (a, b) in enumerate(scenes):
        start, end = float(a.get_seconds()), float(b.get_seconds())
        sf, ef = int(a.get_frames()), int(b.get_frames())
        raw.append(Shot(idx=i, start=start, end=end, start_frame=sf, end_frame=ef))

    merged = _merge_short(raw, min_seconds)
    # make the shots contiguous and clamp to the real frame count
    return _normalise(merged, n_frames, fps, clip_path)


def _whole(clip_path: str, fps: float, n_frames: int) -> Shot:
    dur = n_frames / fps if fps else 0.0
    return Shot(idx=0, start=0.0, end=dur, start_frame=0, end_frame=n_frames)


def _merge_short(shots: list[Shot], min_seconds: float) -> list[Shot]:
    if len(shots) <= 1:
        return shots
    out: list[Shot] = []
    for sh in shots:
        if out and sh.duration < min_seconds:
            # fold this tiny shot into the previous one
            prev = out[-1]
            prev.end = sh.end
            prev.end_frame = sh.end_frame
        else:
            out.append(sh)
    return out


def _normalise(shots: list[Shot], n_frames: int, fps: float,
               clip_path: str) -> list[Shot]:
    if not shots:
        return [_whole(clip_path, fps, n_frames)]
    shots = sorted(shots, key=lambda x: x.start_frame)
    # force contiguity: each shot starts where the previous ended
    shots[0].start_frame = 0
    shots[0].start = 0.0
    for i in range(1, len(shots)):
        shots[i].start_frame = shots[i - 1].end_frame
        shots[i].start = shots[i - 1].end
    if n_frames > 0:
        shots[-1].end_frame = n_frames
        shots[-1].end = n_frames / fps if fps else shots[-1].end
    for i, sh in enumerate(shots):
        sh.idx = i
    return shots


def frame_segments(shots: list[Shot], n_frames: int) -> list[tuple[int, int]]:
    """Return per-shot (start_frame, end_frame) ranges that exactly tile
    [0, n_frames), regardless of small off-by-one from the detector."""
    if not shots:
        return [(0, n_frames)]
    segs: list[tuple[int, int]] = []
    cursor = 0
    for i, sh in enumerate(shots):
        end = n_frames if i == len(shots) - 1 else min(n_frames, sh.end_frame)
        end = max(end, cursor + 1)
        segs.append((cursor, min(end, n_frames)))
        cursor = segs[-1][1]
        if cursor >= n_frames:
            break
    if segs and segs[-1][1] < n_frames:
        segs[-1] = (segs[-1][0], n_frames)
    return segs


# --------------------------------------------------------------------------- #
# conservative literal-duplicate replay flag (semantic replays => Director/P1)
# --------------------------------------------------------------------------- #
def mark_duplicate_shots(clip_path: str, shots: list[Shot],
                         cfg: dict | None = None) -> list[Shot]:
    """Flag shots that are near-identical repeats of an earlier shot.

    This catches *literal* duplicate inserts only (same framing repeated). It is
    deliberately conservative (high correlation threshold) to avoid dropping real
    play. True broadcast-replay detection (other angle / slow-mo) is the
    Director's job in P1.
    """
    cfg = cfg or {}
    s = cfg.get("shots", {}) if isinstance(cfg, dict) else {}
    if not s.get("detect_replays", True) or len(shots) < 2:
        return shots
    corr_thresh = float(s.get("replay_correlation", 0.97))

    try:
        import cv2
        import numpy as np
    except Exception:  # noqa: BLE001
        return shots

    cap = cv2.VideoCapture(clip_path)
    for sh in shots:
        sh.signature = _shot_signature(cap, sh, cv2, np)
    cap.release()

    for i in range(1, len(shots)):
        if shots[i].signature is None:
            continue
        for j in range(i):
            if shots[j].signature is None or shots[j].is_replay:
                continue
            c = _corr(shots[i].signature, shots[j].signature, np)
            if c >= corr_thresh:
                shots[i].is_replay = True
                log.info(f"[shots] shot {i} flagged as duplicate of {j} "
                         f"(corr={c:.3f})")
                break
    return shots


def _shot_signature(cap, sh: Shot, cv2, np):
    """Average HSV histogram over a few frames sampled across the shot."""
    n = sh.n_frames
    if n <= 0:
        return None
    idxs = [sh.start_frame + int(n * f) for f in (0.25, 0.5, 0.75)]
    hists = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(h, h)
        hists.append(h.flatten())
    if not hists:
        return None
    return list(np.mean(hists, axis=0))


def _corr(a, b, np) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return 0.0
    am, bm = a - a.mean(), b - b.mean()
    denom = (np.linalg.norm(am) * np.linalg.norm(bm))
    return float(am.dot(bm) / denom) if denom else 0.0



def shot_activity(shots, frames):
    """Per-shot play density: average players on screen + fraction of frames with
    a visible ball. Cutaways (crowd / celebration close-ups / replay graphics)
    have few players AND no ball; real action has players and/or the ball."""
    out = []
    n = len(frames)
    for s in shots:
        a, b = max(0, s.start_frame), min(n, s.end_frame)
        rng = frames[a:b]
        if not rng:
            out.append({"shot": s, "avg_players": 0.0, "ball_frac": 0.0})
            continue
        players = sum(len(getattr(f, "players", []) or []) for f in rng) / len(rng)
        ball = sum(1 for f in rng if getattr(f, "ball", None)) / len(rng)
        out.append({"shot": s, "avg_players": players, "ball_frac": ball})
    return out


def select_action_span(shots, frames, cfg=None, anchor_frame=None):
    """(i0, i1) frame range of the best CONTIGUOUS run of PLAYABLE shots (real
    action) around the anchor, dropping leading/trailing cutaways (crowd /
    celebration / replay graphics). Returns None to keep the whole clip when the
    feature is off, there is nothing to trim, or the signal is ambiguous."""
    cfg = cfg or {}
    d = cfg.get("director", {})
    if not d.get("trim_cutaways", True):
        return None
    if not shots or len(shots) < 2 or not frames:
        return None
    min_players = float(d.get("cutaway_min_players", 3.0))
    min_ball = float(d.get("cutaway_min_ball_frac", 0.05))
    min_span_s = float(d.get("action_min_seconds", 5.0))

    act = shot_activity(shots, frames)
    playable = [(a["avg_players"] >= min_players or a["ball_frac"] >= min_ball)
                for a in act]
    if all(playable) or not any(playable):
        return None                                # nothing to trim / ambiguous

    # contiguous runs of playable shots
    runs, i = [], 0
    while i < len(playable):
        if playable[i]:
            j = i
            while j + 1 < len(playable) and playable[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1

    chosen = None
    if anchor_frame is not None:
        for r in runs:
            if shots[r[0]].start_frame <= anchor_frame < shots[r[1]].end_frame:
                chosen = r
                break
    if chosen is None:
        chosen = max(runs, key=lambda r: shots[r[1]].end_frame - shots[r[0]].start_frame)

    i0, i1 = shots[chosen[0]].start_frame, shots[chosen[1]].end_frame
    total = shots[-1].end_frame
    if (i1 - i0) >= 0.92 * total:                  # basically the whole clip
        return None
    if (shots[chosen[1]].end - shots[chosen[0]].start) < min_span_s:
        return None                                # too short -> keep clip
    return (i0, i1)
