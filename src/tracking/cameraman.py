"""Blueprint Module 3 — the Cameraman (CMC tracking + smooth 9:16 viewport).

The shaky/erratic crop in raw broadcast footage comes from two things:
  1. the *broadcast* camera panning/zooming (global motion), and
  2. instantaneous jumps when the naive crop snaps to whatever is detected.

This module fixes both:

  * **Camera Motion Compensation (CMC):** runs YOLO + BoT-SORT with
    Generalized Motion Compensation enabled (Ultralytics `botsort.yaml`
    `gmc_method`: sparseOptFlow/orb/ecc). BoT-SORT estimates a global
    background-motion transform each frame and applies it inside the Kalman
    predict step, so track IDs stay stable while the broadcast camera moves.
  * **Target-centric, Kalman-smoothed viewport:** locks onto the hero track
    (from the Director's `main_hero_description`, else the player nearest the
    ball at the decisive beat), computes the centre of mass of hero + ball, and
    glides a 9:16 crop window to it via a constant-velocity Kalman filter
    (rolling-average fallback). No jerky instantaneous jumps.

Output is a `CropPlan` (per-frame crop boxes). `render()` writes the actual
vertical video, reusing the project's ffmpeg helpers for an audio-safe mux.

Backends:
  * "ultralytics" (default) — built-in BoT-SORT, GMC exposed via tracker yaml.
  * "roboflow"             — roboflow `trackers` BoTSORTTracker (no GMC knob;
                             we still apply our own viewport smoothing).
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..edit import ff
from ..utils.io import get_logger, resolve_device

log = get_logger()


@dataclass
class FrameTrack:
    idx: int
    players: list[dict] = field(default_factory=list)   # {id, cls, xyxy, center}
    ball: dict | None = None                            # {xyxy, center}


@dataclass
class CropPlan:
    src_w: int
    src_h: int
    fps: float
    crop_w: int
    crop_h: int
    out_w: int
    out_h: int
    hero_id: int | None
    # per-frame top-left of the crop window (len == n_frames)
    boxes: list[tuple[int, int]] = field(default_factory=list)
    # optional per-frame crop size (w,h) for per-shot zoom; falls back to crop_w/h
    sizes: list[tuple[int, int]] = field(default_factory=list)
    frames: list[FrameTrack] = field(default_factory=list)

    def box_at(self, idx: int) -> tuple[int, int, int, int]:
        i = min(max(idx, 0), len(self.boxes) - 1)
        x0, y0 = self.boxes[i]
        if self.sizes:
            cw, ch = self.sizes[min(i, len(self.sizes) - 1)]
        else:
            cw, ch = self.crop_w, self.crop_h
        return x0, y0, cw, ch


class Cameraman:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.v = cfg.get("vision", {})
        self.t = cfg.get("tracking", {})
        self.device = resolve_device(self.v.get("device", "cuda"))
        self._model = None
        self._ball_model = None

    # ----------------------------------------------------------------- track
    def _load(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.v["player_model"])
            if self.v.get("ball_model"):
                self._ball_model = YOLO(self.v["ball_model"])
        return self._model

    def track(self, clip_path: str) -> CropPlan:
        frames, meta = self.track_only(clip_path)
        return self._plan_from_frames(frames, meta)

    def track_only(self, clip_path: str):
        """Run detection + BoT-SORT (with CMC) and return (frames, meta) WITHOUT
        planning the crop, so analytics can resolve the hero first."""
        backend = (self.t.get("backend") or "ultralytics").lower()
        if backend == "roboflow":
            return self._track_roboflow(clip_path)
        return self._track_ultralytics(clip_path)

    def build_plan(self, frames, meta, hero_id: int | None = None,
                   shots=None, shot_edits=None, hero_ids=None) -> CropPlan:
        """Plan the smoothed 9:16 crop around an explicit hero track id.

        `shots` makes the crop reset at every broadcast cut. `shot_edits` sets a
        per-shot zoom/framing. `hero_ids` (per-frame hero track id, from
        cross-shot Re-ID) lets the camera follow the SAME player across cuts even
        when the tracker re-numbers him."""
        return self._plan_from_frames(frames, meta, hero_hint=hero_id,
                                      shots=shots, shot_edits=shot_edits,
                                      hero_ids=hero_ids)

    # --- Ultralytics BoT-SORT with GMC (default) -------------------------- #
    def _track_ultralytics(self, clip_path: str):
        import cv2
        model = self._load()
        names = getattr(model, "names", {}) or {}
        player_cls, ball_cls = _resolve_classes(names, self.v.get("classes", {}))
        tracker_yaml = self._botsort_yaml()

        cap = cv2.VideoCapture(clip_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        results = model.track(
            source=clip_path, tracker=tracker_yaml, persist=True,
            stream=True, conf=self.v.get("conf_threshold", 0.3),
            iou=self.v.get("iou_threshold", 0.5), imgsz=self.v.get("imgsz", 1280),
            device=self.device, verbose=False,
        )

        frames: list[FrameTrack] = []
        for idx, res in enumerate(results):
            ft = FrameTrack(idx=idx)
            boxes = getattr(res, "boxes", None)
            if boxes is not None and boxes.xyxy is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().numpy()
                cls = (boxes.cls.cpu().numpy() if boxes.cls is not None
                       else np.zeros(len(xyxy)))
                ids = (boxes.id.cpu().numpy() if boxes.id is not None
                       else -np.ones(len(xyxy)))
                ball_best, ball_score = None, -1.0
                conf = (boxes.conf.cpu().numpy() if boxes.conf is not None
                        else np.ones(len(xyxy)))
                for bb, c, tid, cf in zip(xyxy, cls, ids, conf):
                    center = [float((bb[0] + bb[2]) / 2), float((bb[1] + bb[3]) / 2)]
                    if int(c) in ball_cls:
                        if cf > ball_score:
                            ball_score = cf
                            ball_best = {"xyxy": [float(x) for x in bb], "center": center}
                    elif int(c) in player_cls:
                        ft.players.append({"id": int(tid), "cls": int(c),
                                           "xyxy": [float(x) for x in bb],
                                           "center": center})
                ft.ball = ball_best
            frames.append(ft)

        # optional dedicated ball pass (small-object) overrides ball detections
        if self._ball_model is not None:
            self._refine_ball(clip_path, frames)

        return frames, {"fps": fps, "w": w, "h": h}

    def _refine_ball(self, clip_path: str, frames: list[FrameTrack]):
        import cv2
        cap = cv2.VideoCapture(clip_path)
        idx = 0
        while idx < len(frames):
            ok, frame = cap.read()
            if not ok:
                break
            res = self._ball_model.predict(frame, imgsz=self.v.get("imgsz", 1280),
                                           conf=0.15, device=self.device,
                                           verbose=False)[0]
            bxy = getattr(res.boxes, "xyxy", None)
            if bxy is not None and len(bxy):
                conf = res.boxes.conf.cpu().numpy()
                bb = bxy.cpu().numpy()[int(np.argmax(conf))]
                frames[idx].ball = {"xyxy": [float(x) for x in bb],
                                    "center": [float((bb[0] + bb[2]) / 2),
                                               float((bb[1] + bb[3]) / 2)]}
            idx += 1
        cap.release()

    # --- roboflow trackers backend (no GMC; our smoothing still applies) -- #
    def _track_roboflow(self, clip_path: str):
        import cv2
        import supervision as sv
        from trackers import BoTSORTTracker

        model = self._load()
        names = getattr(model, "names", {}) or {}
        player_cls, ball_cls = _resolve_classes(names, self.v.get("classes", {}))
        tracker = BoTSORTTracker()

        cap = cv2.VideoCapture(clip_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames: list[FrameTrack] = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = model.predict(frame, imgsz=self.v.get("imgsz", 1280),
                                conf=self.v.get("conf_threshold", 0.3),
                                device=self.device, verbose=False)[0]
            dets = sv.Detections.from_ultralytics(res)
            ball = _best_ball(dets, ball_cls)
            pmask = np.isin(dets.class_id, list(player_cls))
            pdets = tracker.update(dets[pmask], frame)
            ft = FrameTrack(idx=idx)
            for bb, tid, c in zip(pdets.xyxy, pdets.tracker_id, pdets.class_id):
                ft.players.append({"id": int(tid), "cls": int(c),
                                   "xyxy": [float(x) for x in bb],
                                   "center": [float((bb[0] + bb[2]) / 2),
                                              float((bb[1] + bb[3]) / 2)]})
            if ball is not None:
                ft.ball = {"xyxy": [float(x) for x in ball],
                           "center": [float((ball[0] + ball[2]) / 2),
                                      float((ball[1] + ball[3]) / 2)]}
            frames.append(ft)
            idx += 1
        cap.release()
        return frames, {"fps": fps, "w": w, "h": h}

    # ------------------------------------------------------------- planning
    def _plan_from_frames(self, frames, meta, hero_hint: int | None = None,
                          shots=None, shot_edits=None, hero_ids=None):
        w, h, fps = meta["w"], meta["h"], meta["fps"]
        rf = self.cfg.get("edit", {}).get("reframe", {})
        aspect = rf.get("target_aspect", "9:16")
        aw, ah = (int(x) for x in aspect.split(":"))

        hero_id = hero_hint if hero_hint is not None else _pick_hero(frames)

        # Aggressive action-centric base zoom: size the crop so the hero fills
        # ~target_subject_height of the frame instead of using the full height
        # (which reads as a static letterbox slice). Per-shot Director zoom still
        # multiplies on top of this.
        base_zoom = 1.0
        if rf.get("mode", "action_track") != "letterbox":
            base_zoom = _auto_zoom(frames, hero_id, h, rf)
        crop_h = max(1, min(h, int(round(h / base_zoom))))
        crop_w = min(w, int(round(crop_h * aw / ah)))

        # follow a per-frame hero (cross-shot Re-ID) when supplied
        focus = _focus_points(frames, hero_ids if hero_ids is not None else hero_id,
                              w, h)
        n = len(frames)

        # per-shot smoothing segments (reset the camera at every cut)
        segments = None
        if shots:
            from ..perception.shots import frame_segments
            segments = frame_segments(shots, n)

        # per-frame crop size from the Director's per-shot zoom/framing
        cw_arr = np.full(n, float(crop_w))
        ch_arr = np.full(n, float(crop_h))
        if shot_edits and segments:
            for si, (a, b) in enumerate(segments):
                se = shot_edits[si] if si < len(shot_edits) else None
                if se is None:
                    continue
                z = max(0.6, min(2.5, float(getattr(se, "zoom", 1.0))))
                if getattr(se, "framing", "") == "letterbox_wide":
                    z = min(z, 1.0)            # never tighter than full for wide
                cw_arr[a:b] = min(w, crop_w / z)
                ch_arr[a:b] = min(h, crop_h / z)

        # Kalman-smoothed camera path (constant-velocity); EMA fallback
        cx, cy = _kalman_smooth(focus[:, 0], focus[:, 1],
                                fps=fps, cfg=self.t, segments=segments)
        cx = np.clip(cx, cw_arr / 2.0, w - cw_arr / 2.0)
        cy = np.clip(cy, ch_arr / 2.0, h - ch_arr / 2.0)

        prof = self.cfg.get("_active_profile", {"width": 1080, "height": 1920})
        boxes = [(int(round(x - cw / 2.0)), int(round(y - ch / 2.0)))
                 for x, y, cw, ch in zip(cx, cy, cw_arr, ch_arr)]
        sizes = [(int(round(cw)), int(round(ch))) for cw, ch in zip(cw_arr, ch_arr)]
        return CropPlan(src_w=w, src_h=h, fps=fps, crop_w=crop_w, crop_h=crop_h,
                        out_w=prof["width"], out_h=prof["height"], hero_id=hero_id,
                        boxes=boxes, sizes=sizes, frames=frames)

    def _botsort_yaml(self) -> str:
        """Write a BoT-SORT tracker config with GMC enabled and return its path.

        gmc_method is the camera-motion-compensation estimator: 'sparseOptFlow'
        (fast, default), 'orb', 'sift', or 'ecc'. Disabling it ('none') reverts
        to plain ByteTrack-style behaviour.
        """
        gmc = self.t.get("gmc_method", "sparseOptFlow")
        text = (
            "tracker_type: botsort\n"
            f"track_high_thresh: {self.t.get('track_high_thresh', 0.25)}\n"
            f"track_low_thresh: {self.t.get('track_low_thresh', 0.1)}\n"
            f"new_track_thresh: {self.t.get('new_track_thresh', 0.25)}\n"
            f"track_buffer: {self.t.get('track_buffer', 60)}\n"
            f"match_thresh: {self.t.get('match_thresh', 0.8)}\n"
            "fuse_score: true\n"
            f"gmc_method: {gmc}\n"
            "proximity_thresh: 0.5\n"
            "appearance_thresh: 0.25\n"
            f"with_reid: {str(bool(self.t.get('with_reid', False))).lower()}\n"
            # Ultralytics 8.3+/8.4 require a `model` key on the BoT-SORT config
            # (the ReID encoder; 'auto' lets it pick). Without it tracking raises
            # "'IterableSimpleNamespace' object has no attribute 'model'".
            f"model: {self.t.get('reid_model', 'auto')}\n"
        )
        d = Path(tempfile.gettempdir()) / "fhs_botsort.yaml"
        d.write_text(text, encoding="utf-8")
        return str(d)

    # --------------------------------------------------------------- render
    def render(self, clip_path: str, plan: CropPlan, out_path: str,
               annotate_world=None, annotate_screen=None,
               intermediate: bool = False) -> str:
        """Crop every frame to its planned window and resize to the target
        9:16 profile, then mux the original audio back (audio-safe).

        Optional annotators merge graphics into THIS pass (one fewer encode):
          * annotate_world(frame, idx) runs on the ORIGINAL frame before the crop
            (pitch-space halos/trail),
          * annotate_screen(frame, idx) runs on the cropped OUTPUT frame
            (screen-space HUD like the possession plate).
        Frames are piped straight into a single H.264 encode (no lossy mp4v
        intermediate); the upscaled crop uses Lanczos for crispness."""
        import cv2

        cap = cv2.VideoCapture(clip_path)
        encoder = ff.pick_encoder(self.cfg.get("render", {}).get("encoder", "libx264"))
        sink = ff.RawFrameSink(out_path, plan.out_w, plan.out_h, plan.fps,
                               encoder, audio_src=clip_path,
                               intermediate=intermediate)
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if annotate_world is not None:
                frame = annotate_world(frame, idx)       # pitch-space, pre-crop
            x0, y0, cw, ch = plan.box_at(idx)
            x0 = max(0, min(x0, plan.src_w - cw))
            y0 = max(0, min(y0, plan.src_h - ch))
            crop = frame[y0:y0 + ch, x0:x0 + cw]
            crop = cv2.resize(crop, (plan.out_w, plan.out_h),
                              interpolation=cv2.INTER_LANCZOS4)
            if annotate_screen is not None:
                crop = annotate_screen(crop, idx)        # HUD-space, post-crop
            sink.write(crop)
            idx += 1
        cap.release()
        sink.close()
        log.info(f"[cameraman] {plan.out_w}x{plan.out_h} CMC reframe -> "
                 f"{Path(out_path).name} (hero={plan.hero_id})")
        return out_path


# --------------------------------------------------------------------------- #
# functional convenience entrypoint
# --------------------------------------------------------------------------- #
def plan_vertical_crop(clip_path: str, cfg: dict,
                       hero_id: int | None = None) -> CropPlan:
    """Track a clip and return the smoothed 9:16 crop plan (no render)."""
    cam = Cameraman(cfg)
    backend = (cfg.get("tracking", {}).get("backend") or "ultralytics").lower()
    if backend == "roboflow":
        frames, meta = cam._track_roboflow(clip_path)
    else:
        frames, meta = cam._track_ultralytics(clip_path)
    return cam._plan_from_frames(frames, meta, hero_hint=hero_id)


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def _resolve_classes(names: dict, cfg_classes: dict):
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


def _best_ball(dets, ball_cls):
    mask = np.isin(dets.class_id, list(ball_cls))
    bd = dets[mask]
    if len(bd) == 0:
        return None
    return bd.xyxy[int(np.argmax(bd.confidence))]


def _auto_zoom(frames, hero_id, h, rf) -> float:
    """Base punch-in so the hero fills ~target_subject_height of the frame.

    zoom = frame_h / crop_h, where crop_h = hero_height / target. A small (far)
    hero -> higher zoom (tighter); clamped to [min_zoom, max_zoom] so a tiny
    misdetection can't over-zoom into a shaky crop.
    """
    target = float(rf.get("target_subject_height", 0.45))
    lo = float(rf.get("min_zoom", 1.0))
    hi = float(rf.get("max_zoom", 2.6))
    if target <= 0 or hero_id is None:
        return lo
    hs = [float(p["xyxy"][3] - p["xyxy"][1])
          for ft in frames for p in getattr(ft, "players", [])
          if p["id"] == hero_id]
    if not hs:
        return lo
    hero_h = float(np.median(hs))
    if hero_h <= 1.0:
        return lo
    zoom = (h * target) / hero_h
    return max(lo, min(hi, zoom))


def _pick_hero(frames: list[FrameTrack]) -> int | None:
    """Player most often nearest the ball during the final 40% of the clip."""
    if not frames:
        return None
    start = int(len(frames) * 0.6)
    votes: dict[int, int] = {}
    for ft in frames[start:]:
        if not ft.ball or not ft.players:
            continue
        bx, by = ft.ball["center"]
        nearest = min(ft.players, key=lambda p: (p["center"][0] - bx) ** 2
                      + (p["center"][1] - by) ** 2)
        votes[nearest["id"]] = votes.get(nearest["id"], 0) + 1
    return max(votes, key=votes.get) if votes else None


def _focus_points(frames, hero_id, w, h) -> np.ndarray:
    """Per-frame focus = weighted centre of mass of (hero, ball). Falls back to
    ball, then all-players centroid, then last-known / frame centre.

    `hero_id` may be a single track id or a per-frame list/array of ids
    (cross-shot Re-ID), so the camera can follow the same player across cuts."""
    per_frame = isinstance(hero_id, (list, tuple, np.ndarray))
    pts = []
    last = np.array([w / 2.0, h / 2.0])
    for i, ft in enumerate(frames):
        hid = (hero_id[i] if per_frame and i < len(hero_id) else
               (hero_id[-1] if per_frame and hero_id else hero_id))
        hero = next((p for p in ft.players if p["id"] == hid), None)
        ball = ft.ball
        if hero and ball:
            p = 0.5 * np.array(hero["center"]) + 0.5 * np.array(ball["center"])
        elif ball:
            p = np.array(ball["center"])
        elif hero:
            p = np.array(hero["center"])
        elif ft.players:
            p = np.mean([pl["center"] for pl in ft.players], axis=0)
        else:
            p = last
        last = p
        pts.append(p)
    return np.array(pts, dtype=np.float64) if pts else np.array([[w / 2, h / 2]])


def _kalman_smooth(xs: np.ndarray, ys: np.ndarray, fps: float, cfg: dict,
                   segments=None):
    """Constant-velocity Kalman smoothing of the focus path in x and y.

    If `segments` (list of (start,end) frame ranges, one per shot) is given,
    each shot is smoothed INDEPENDENTLY so the camera resets at every broadcast
    cut instead of gliding across it. Falls back to a causal EMA per segment if
    anything goes wrong. The process/measurement noise ratio controls how 'lazy'
    the camera is.
    """
    n = len(xs)
    if n == 0:
        return xs, ys
    segs = segments if segments else [(0, n)]
    cx = np.empty(n, dtype=np.float64)
    cy = np.empty(n, dtype=np.float64)
    last = 0
    for a, b in segs:
        a = max(0, min(a, n))
        b = max(a, min(b, n))
        if b <= a:
            continue
        sx, sy = _smooth_one(xs[a:b], ys[a:b], fps, cfg)
        cx[a:b] = sx
        cy[a:b] = sy
        last = b
    if last < n:                      # safety: cover any gap with the tail value
        sx, sy = _smooth_one(xs[last:n], ys[last:n], fps, cfg)
        cx[last:n] = sx
        cy[last:n] = sy
    return cx, cy


def _smooth_one(xs: np.ndarray, ys: np.ndarray, fps: float, cfg: dict):
    try:
        q = float(cfg.get("kalman_process_noise", 2.0))
        r = float(cfg.get("kalman_measurement_noise", 120.0))
        return _kalman_1d(xs, q, r, fps), _kalman_1d(ys, q, r, fps)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"[cameraman] kalman failed ({exc}); EMA fallback")
        a = float(cfg.get("smoothing", 0.85))
        return _ema(xs, a), _ema(ys, a)


def _kalman_1d(z: np.ndarray, q: float, r: float, fps: float) -> np.ndarray:
    """1-D constant-velocity Kalman filter + RTS (backward) smoothing.

    State = [position, velocity]. A forward-only Kalman filter lags behind fast
    motion, so the crop window trails the ball. Since the whole clip is known
    offline, we run the Rauch-Tung-Striebel backward pass to get the optimal
    *smoothed* (zero-lag) estimate — the camera then anticipates motion instead
    of chasing it. Returns the smoothed position per step.
    """
    n = len(z)
    if n == 0:
        return z
    dt = 1.0 / max(1.0, fps)
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    R = np.array([[r]])

    # ---- forward filter (store a-priori and a-posteriori states/covs) ----
    xf = np.zeros((n, 2, 1))      # filtered (a-posteriori) state
    Pf = np.zeros((n, 2, 2))      # filtered covariance
    xp = np.zeros((n, 2, 1))      # predicted (a-priori) state
    Pp = np.zeros((n, 2, 2))      # predicted covariance
    x = np.array([[z[0]], [0.0]])
    P = np.eye(2) * 1000.0
    for k in range(n):
        # predict
        x = F @ x
        P = F @ P @ F.T + Q
        xp[k], Pp[k] = x, P
        # update
        y = np.array([[z[k]]]) - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(2) - K @ H) @ P
        xf[k], Pf[k] = x, P

    # ---- RTS backward smoothing ----
    xs = xf.copy()
    Ps = Pf.copy()
    for k in range(n - 2, -1, -1):
        C = Pf[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xf[k] + C @ (xs[k + 1] - xp[k + 1])
        Ps[k] = Pf[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T
    return xs[:, 0, 0]


def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * out[i - 1] + (1 - alpha) * x[i]
    return out
