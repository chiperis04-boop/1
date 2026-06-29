"""Player + ball detection and tracking.

Runs an Ultralytics YOLO model on every frame of a *clip* (not the whole match),
then tracks detections with ByteTrack via the `supervision` library. Output is a
per-frame structure the telestration stage consumes.

We also pick the "key player" — the scorer/protagonist — heuristically as the
player closest to the ball during the decisive beat (the last third of the clip,
where the shot/goal happens). This drives the spotlight and motion arrow.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..utils.io import get_logger

log = get_logger()


@dataclass
class FrameDets:
    idx: int
    players: list[dict] = field(default_factory=list)   # {id, xyxy, center}
    ball: dict | None = None                            # {xyxy, center}


@dataclass
class TrackResult:
    width: int
    height: int
    fps: float
    frames: list[FrameDets]
    key_track_id: int | None = None
    ball_path: list[tuple[int, float, float]] = field(default_factory=list)  # (idx,x,y)


def track_clip(clip_path: str, cfg: dict) -> TrackResult:
    import cv2
    from ultralytics import YOLO
    import supervision as sv

    v = cfg["vision"]
    model = YOLO(v["player_model"])
    ball_model = YOLO(v["ball_model"]) if v.get("ball_model") else None

    tracker = sv.ByteTrack()
    # Resolve class ids by NAME from the model itself so any football model works
    # regardless of its class-index order; fall back to config indices.
    player_classes, ball_classes = _resolve_classes(model, v.get("classes", {}))
    log.info(f"[vision] player classes={sorted(player_classes)} "
             f"ball classes={sorted(ball_classes)} (from {model.names})")

    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames: list[FrameDets] = []
    ball_path: list[tuple[int, float, float]] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        res = model.predict(frame, imgsz=v["imgsz"], conf=v["conf_threshold"],
                            iou=v["iou_threshold"], device=v["device"], verbose=False)[0]
        dets = sv.Detections.from_ultralytics(res)

        # split players vs ball
        is_player = np.isin(dets.class_id, list(player_classes))
        player_dets = dets[is_player]
        player_dets = tracker.update_with_detections(player_dets)

        fd = FrameDets(idx=idx)
        for xyxy, tid in zip(player_dets.xyxy, player_dets.tracker_id):
            cx = float((xyxy[0] + xyxy[2]) / 2)
            cy = float((xyxy[1] + xyxy[3]) / 2)
            fd.players.append({"id": int(tid), "xyxy": [float(x) for x in xyxy],
                              "center": [cx, cy]})

        # ball: dedicated model if provided, else generic class from main model
        ball_xyxy = _best_ball(dets, ball_classes)
        if ball_model is not None:
            bres = ball_model.predict(frame, imgsz=v["imgsz"], conf=0.15,
                                      device=v["device"], verbose=False)[0]
            bdets = sv.Detections.from_ultralytics(bres)
            if len(bdets):
                ball_xyxy = bdets.xyxy[int(np.argmax(bdets.confidence))]
        if ball_xyxy is not None:
            bx = float((ball_xyxy[0] + ball_xyxy[2]) / 2)
            by = float((ball_xyxy[1] + ball_xyxy[3]) / 2)
            fd.ball = {"xyxy": [float(x) for x in ball_xyxy], "center": [bx, by]}
            ball_path.append((idx, bx, by))

        frames.append(fd)
        idx += 1

    cap.release()
    ball_path = _interpolate_ball(ball_path)

    result = TrackResult(width=width, height=height, fps=fps, frames=frames,
                         ball_path=ball_path)
    result.key_track_id = _pick_key_player(frames)
    log.info(f"[vision] tracked {idx} frames, key player id={result.key_track_id}")
    return result


# --------------------------------------------------------------------------- #
def _resolve_classes(model, cfg_classes: dict):
    """Map a YOLO model's class names to our (player, ball) id sets.

    'player' and 'goalkeeper' -> player; anything with 'ball' -> ball; referees
    are intentionally excluded. Falls back to config indices if names are absent.
    """
    names = getattr(model, "names", None) or {}
    player_ids, ball_ids = [], []
    for i, n in names.items():
        ln = str(n).lower()
        if "ball" in ln:
            ball_ids.append(int(i))
        elif "player" in ln or "keeper" in ln:
            player_ids.append(int(i))
    if not player_ids:
        player_ids = list(cfg_classes.get("player", [0]))
    if not ball_ids:
        ball_ids = list(cfg_classes.get("ball", [32]))
    return set(player_ids), set(ball_ids)


def _best_ball(dets, ball_classes):
    import numpy as np
    mask = np.isin(dets.class_id, list(ball_classes))
    bd = dets[mask]
    if len(bd) == 0:
        return None
    return bd.xyxy[int(np.argmax(bd.confidence))]


def _interpolate_ball(path):
    """Fill short gaps in ball detections by linear interpolation."""
    if len(path) < 2:
        return path
    full = []
    for (i0, x0, y0), (i1, x1, y1) in zip(path, path[1:]):
        full.append((i0, x0, y0))
        gap = i1 - i0
        if 1 < gap <= 8:
            for k in range(1, gap):
                a = k / gap
                full.append((i0 + k, x0 + a * (x1 - x0), y0 + a * (y1 - y0)))
    full.append(path[-1])
    return full


def _pick_key_player(frames: list[FrameDets]) -> int | None:
    """Protagonist = player most often nearest the ball during the final third
    of the clip (where the decisive action lives)."""
    if not frames:
        return None
    start = int(len(frames) * 0.6)
    votes: dict[int, int] = {}
    for fd in frames[start:]:
        if not fd.ball or not fd.players:
            continue
        bx, by = fd.ball["center"]
        nearest = min(fd.players,
                      key=lambda p: (p["center"][0] - bx) ** 2 + (p["center"][1] - by) ** 2)
        votes[nearest["id"]] = votes.get(nearest["id"], 0) + 1
    if not votes:
        return None
    return max(votes, key=votes.get)
