"""Player segmentation for under-player graphics compositing (Blueprint 4B).

The grounded lower-arc halo (`telestration.halo_grounded`) is a seg-mask-FREE
approximation of "graphics under the boots". For the real thing we need per-frame
player segmentation masks so tactical graphics (rings, trails, lines) can be
drawn on the grass and the player pixels re-pasted on top
(`graphics.homography.composite_under_players`).

`Occluder` wraps a YOLO*-seg model (Ultralytics; the generic COCO `*-seg.pt`
detects the `person` class = players and is auto-downloaded, or point
`telestration.occlusion_model` at a football-specific seg model). Everything is
best-effort and honest:

  * `available()` actually tries to LOAD the model. If Ultralytics/torch or the
    weights are missing it returns False, and the caller falls back to the
    grounded-arc halo — occlusion is never faked.
  * `player_mask(frame)` returns a boolean HxW union mask of all player masks,
    or None on any failure (so compositing degrades to graphics-on-top, never a
    crash).

This is a GPU stage in practice (seg on every frame); on CPU it still runs but
slowly, so it is opt-in via `telestration.occlusion`. Verify visual quality on
GPU.
"""
from __future__ import annotations

from ..utils.io import get_logger, resolve_device

log = get_logger()

# COCO seg class names we treat as "a player to keep on top of the graphics".
_PLAYER_NAME_HINTS = ("player", "person", "keeper", "goalkeeper", "referee")


class Occluder:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        tele = cfg.get("telestration", {})
        v = cfg.get("vision", {})
        self.enabled = bool(tele.get("occlusion", False))
        self.model_path = tele.get("occlusion_model", "yolo11x-seg.pt")
        self.conf = float(tele.get("occlusion_conf", 0.3))
        self.alpha = float(tele.get("occlusion_alpha", 0.85))
        self.imgsz = int(v.get("imgsz", 1280))
        self.device = resolve_device(v.get("device", "cuda"))
        self._model = None
        self._ok: bool | None = None          # tri-state: None=untried
        self._player_ids: set[int] | None = None

    # ------------------------------------------------------------------ load
    def available(self) -> bool:
        """True only if the seg model actually loads (honest — no faking)."""
        if not self.enabled:
            return False
        if self._ok is None:
            self._load()
        return bool(self._ok)

    def _load(self):
        if self._ok is not None:
            return self._model
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            names = getattr(self._model, "names", {}) or {}
            ids = {int(i) for i, n in names.items()
                   if any(h in str(n).lower() for h in _PLAYER_NAME_HINTS)}
            # COCO 'person' is class 0; default to it if names lookup found none
            self._player_ids = ids or {0}
            self._ok = True
            log.info(f"[occlusion] seg model loaded ({self.model_path}); "
                     f"player classes={sorted(self._player_ids)}")
        except Exception as exc:  # noqa: BLE001
            self._ok = False
            self._model = None
            log.warning(f"[occlusion] seg model unavailable ({exc}); "
                        "falling back to grounded-arc halo")
        return self._model

    # ------------------------------------------------------------------ mask
    def player_mask(self, frame_bgr):
        """Boolean HxW union of player segmentation masks for `frame_bgr`, or
        None on any failure / when disabled."""
        if not self.available():
            return None
        try:
            import cv2
            import numpy as np

            h, w = frame_bgr.shape[:2]
            res = self._model.predict(frame_bgr, imgsz=self.imgsz, conf=self.conf,
                                      device=self.device, verbose=False)[0]
            masks = getattr(res, "masks", None)
            data = getattr(masks, "data", None) if masks is not None else None
            if data is None or len(data) == 0:
                return np.zeros((h, w), dtype=bool)
            arr = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
            # keep only player-class instances when class ids are available
            cls = getattr(getattr(res, "boxes", None), "cls", None)
            if cls is not None and self._player_ids is not None:
                cls = cls.cpu().numpy() if hasattr(cls, "cpu") else np.asarray(cls)
                keep = [i for i in range(len(arr))
                        if int(cls[i]) in self._player_ids]
                if keep:
                    arr = arr[keep]
            union = arr.max(axis=0) > 0.5                 # (mh, mw) bool
            if union.shape != (h, w):                     # seg is at model res
                union = cv2.resize(union.astype(np.uint8), (w, h),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
            return union
        except Exception as exc:  # noqa: BLE001
            log.warning(f"[occlusion] seg inference failed ({exc}); no mask")
            return None
