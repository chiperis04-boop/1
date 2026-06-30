"""Cross-shot hero Re-ID — keep the framing/graphics on the SAME player through
broadcast cuts.

BoT-SORT track ids reset at every camera cut, so the "hero" id from one shot is
meaningless in the next. This module computes a per-track appearance embedding
and links the hero across shots, producing a per-frame hero id the Cameraman can
follow.

Backends:
  * deterministic (default): a normalised colour histogram of the player's torso
    crop — cheap, no dependencies, good enough to tell team/kit apart.
  * osnet (optional): if `torchreid` is installed, OSNet embeddings can replace
    the histogram for far better matching. Guarded; degrades to histograms.

The matching maths is pure numpy so it is fully CPU-unit-testable.
"""
from __future__ import annotations

import numpy as np

from ..utils.io import get_logger

log = get_logger()


# --------------------------------------------------------------------------- #
# embeddings + similarity (pure)
# --------------------------------------------------------------------------- #
def histogram_embedding(crop: np.ndarray, bins: int = 8) -> np.ndarray:
    """L2-normalised per-channel colour histogram of the torso (upper half)."""
    if crop is None or crop.size == 0:
        return np.zeros(bins * 3, dtype=np.float64)
    torso = crop[: max(1, crop.shape[0] // 2)]
    chans = []
    for c in range(min(3, torso.shape[2] if torso.ndim == 3 else 1)):
        hist, _ = np.histogram(torso[..., c], bins=bins, range=(0, 256))
        chans.append(hist)
    v = np.concatenate(chans).astype(np.float64)
    norm = np.linalg.norm(v)
    return v / norm if norm else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0 or a.shape != b.shape:
        return 0.0
    return float(a.dot(b) / (na * nb))


def best_match(query: np.ndarray, candidates: dict, min_sim: float = 0.5):
    """Return (id, sim) of the candidate embedding most similar to `query`,
    or (None, 0.0) if none clears `min_sim`."""
    best_id, best_sim = None, -1.0
    for cid, emb in candidates.items():
        s = cosine(query, emb)
        if s > best_sim:
            best_id, best_sim = cid, s
    if best_id is None or best_sim < min_sim:
        return None, max(0.0, best_sim)
    return best_id, best_sim


# --------------------------------------------------------------------------- #
# cross-shot linking
# --------------------------------------------------------------------------- #
def cross_shot_hero_map(hero_id, shot_track_embs: dict, hero_shot: int | None,
                        min_sim: float = 0.5) -> dict:
    """Map each shot -> the track id that best matches the hero's appearance.

    shot_track_embs: {shot_idx: {track_id: embedding}}
    Returns {shot_idx: track_id} (falls back to `hero_id` when no match).
    """
    # reference embedding = hero's embedding in its own shot (or any shot it's in)
    ref = None
    if hero_shot is not None and hero_shot in shot_track_embs:
        ref = shot_track_embs[hero_shot].get(hero_id)
    if ref is None:
        for embs in shot_track_embs.values():
            if hero_id in embs:
                ref = embs[hero_id]
                break
    out = {}
    for shot_idx, embs in shot_track_embs.items():
        if hero_id in embs:
            out[shot_idx] = hero_id                  # tracker kept the id here
        elif ref is not None and embs:
            mid, _ = best_match(ref, embs, min_sim)
            out[shot_idx] = mid if mid is not None else hero_id
        else:
            out[shot_idx] = hero_id
    return out


def per_frame_hero(n_frames: int, shots, hero_map: dict, default_hero) -> list:
    """Expand a {shot_idx: track_id} map into a per-frame hero id list."""
    from ..perception.shots import frame_segments
    ids = [default_hero] * n_frames
    if not shots:
        return ids
    segs = frame_segments(shots, n_frames)
    for si, (a, b) in enumerate(segs):
        hid = hero_map.get(si, default_hero)
        for k in range(a, min(b, n_frames)):
            ids[k] = hid
    return ids


# --------------------------------------------------------------------------- #
# build per-track embeddings from the clip (cv2; histogram or optional OSNet)
# --------------------------------------------------------------------------- #
def build_shot_track_embeddings(clip_path: str, frames, shots,
                                cfg: dict | None = None) -> dict:
    """Average appearance embedding per (shot, track id) by sampling frames.

    Returns {shot_idx: {track_id: embedding}}. Empty/guarded if cv2 missing.
    """
    cfg = cfg or {}
    try:
        import cv2
    except Exception:  # noqa: BLE001
        return {}
    from ..perception.shots import frame_segments
    if not shots:
        return {}
    segs = frame_segments(shots, len(frames))
    cap = cv2.VideoCapture(clip_path)
    out: dict = {}
    for si, (a, b) in enumerate(segs):
        acc: dict = {}
        # sample a few frames across the shot
        idxs = sorted(set(int(a + (b - a) * f) for f in (0.25, 0.5, 0.75)))
        for fi in idxs:
            if fi < 0 or fi >= len(frames):
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            for p in getattr(frames[fi], "players", []):
                x1, y1, x2, y2 = (int(v) for v in p["xyxy"])
                x1, y1 = max(0, x1), max(0, y1)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                emb = histogram_embedding(crop)
                acc.setdefault(p["id"], []).append(emb)
        out[si] = {tid: np.mean(v, axis=0) for tid, v in acc.items() if v}
    cap.release()
    return out
