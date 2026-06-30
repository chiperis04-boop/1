"""Blueprint Module 4 (part A) — the "Grass-Anchor" pitch homography.

To make tactical graphics (passing lines, player circles, zones) sit *flat on
the grass* with correct 3D perspective, we need the image->pitch projection
matrix H per frame. A pitch-keypoint model locates known landmarks; we solve H
from the visible correspondences and warp graphics coordinates with it.

Solvers (configurable):
  * "cv2"    (default) — cv2.findHomography(..., RANSAC). Robust to outliers /
                         mis-detected keypoints; matches src/vision/pitch.py.
  * "kornia" (points)  — kornia.geometry.homography.find_homography_dlt.
  * "kornia_lines"     — kornia.geometry.homography.find_homography_lines_dlt,
                         for *line-segment* correspondences (touchlines, box
                         lines). NOTE: the real Kornia function is
                         `find_homography_lines_dlt` — the name in some
                         write-ups ("find_homography_lines_reconstruction") does
                         not exist in Kornia.

Keypoint detection + per-frame solving reuses the tested `PitchEstimator`
(src/vision/pitch.py); this module adds the Kornia solvers, warp helpers, and
the **under-player compositing** that keeps graphics beneath the boots.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..utils.io import get_logger
from ..vision.pitch import PitchCalibration, PitchEstimator

log = get_logger()


@dataclass
class PitchProjector:
    """Wraps one frame's homography for cheap point projection both ways."""
    H: np.ndarray | None              # image -> pitch (metres)
    H_inv: np.ndarray | None = None   # pitch -> image (pixels)

    def __post_init__(self):
        if self.H is not None and self.H_inv is None:
            try:
                self.H_inv = np.linalg.inv(self.H)
            except np.linalg.LinAlgError:
                self.H_inv = None

    def image_to_pitch(self, x: float, y: float):
        return _apply_h(self.H, x, y)

    def pitch_to_image(self, x: float, y: float):
        return _apply_h(self.H_inv, x, y)

    @property
    def valid(self) -> bool:
        return self.H is not None


@dataclass
class GraphicsHomography:
    """Per-frame homographies for a clip + helpers to render grass-anchored
    graphics under players."""
    calibration: PitchCalibration = field(default_factory=PitchCalibration)
    solver: str = "cv2"

    def projector(self, idx: int) -> PitchProjector:
        fh = self.calibration.frames.get(idx)
        return PitchProjector(H=None if fh is None else fh.H)

    @property
    def coverage(self) -> float:
        return self.calibration.coverage


# --------------------------------------------------------------------------- #
def compute_homography(clip_path: str, cfg: dict) -> GraphicsHomography:
    """Detect pitch keypoints per frame and solve H for the whole clip.

    Delegates keypoint detection + cv2 RANSAC solving to the existing
    PitchEstimator, then (optionally) re-solves with Kornia when configured.
    """
    g = cfg.get("graphics", {})
    solver = (g.get("homography_solver") or "cv2").lower()

    est = PitchEstimator(cfg)
    if not est.enabled:
        log.info("[homography] pitch calibration disabled; graphics run in pixel space")
        return GraphicsHomography(solver=solver)

    calib = est.calibrate(clip_path)

    if solver in ("kornia", "kornia_lines"):
        _resolve_with_kornia(calib, solver)

    log.info(f"[homography] solver={solver}, coverage={calib.coverage:.0%}")
    return GraphicsHomography(calibration=calib, solver=solver)


# --------------------------------------------------------------------------- #
# Kornia solvers (operate on the keypoint correspondences PitchEstimator found)
# --------------------------------------------------------------------------- #
def _resolve_with_kornia(calib: PitchCalibration, solver: str):
    """Best-effort re-solve of each frame's H with Kornia. If Kornia or torch is
    unavailable, the cv2 RANSAC result from PitchEstimator is kept."""
    try:
        import torch
        import kornia.geometry.homography as kgh
    except Exception as exc:  # noqa: BLE001
        log.warning(f"[homography] kornia unavailable ({exc}); keeping cv2 result")
        return
    # PitchEstimator stored only the final H, not the raw correspondences, so a
    # full kornia re-solve would need the keypoints re-detected. We expose the
    # function wiring here and leave per-frame correspondences as a future hook;
    # this keeps the cv2 H (already robust) while documenting the exact API.
    _ = (torch, kgh)  # referenced so the import is meaningful
    log.info("[homography] kornia solver requested; using cv2 H "
             "(plug correspondences into find_homography_lines_dlt to enable)")


def find_homography_points_kornia(img_pts: np.ndarray, pitch_pts: np.ndarray):
    """Solve H (image->pitch) from point correspondences via Kornia DLT.

    img_pts/pitch_pts: (N, 2) arrays. Returns a 3x3 numpy array or None.
    """
    try:
        import torch
        from kornia.geometry.homography import find_homography_dlt
    except Exception as exc:  # noqa: BLE001
        log.warning(f"[homography] kornia points solve unavailable: {exc}")
        return None
    if len(img_pts) < 4:
        return None
    p1 = torch.from_numpy(np.asarray(img_pts, np.float32))[None]    # (1, N, 2)
    p2 = torch.from_numpy(np.asarray(pitch_pts, np.float32))[None]
    H = find_homography_dlt(p1, p2, solver="svd")
    return H[0].cpu().numpy()


def find_homography_lines_kornia(img_lines: np.ndarray, pitch_lines: np.ndarray):
    """Solve H from *line-segment* correspondences via Kornia.

    img_lines/pitch_lines: (N, 2, 2) arrays (each line = 2 endpoints). Uses the
    real Kornia function `find_homography_lines_dlt`. Returns 3x3 or None.
    """
    try:
        import torch
        from kornia.geometry.homography import find_homography_lines_dlt
    except Exception as exc:  # noqa: BLE001
        log.warning(f"[homography] kornia line solve unavailable: {exc}")
        return None
    if len(img_lines) < 4:
        return None
    ls1 = torch.from_numpy(np.asarray(img_lines, np.float32))[None]   # (1, N, 2, 2)
    ls2 = torch.from_numpy(np.asarray(pitch_lines, np.float32))[None]
    H = find_homography_lines_dlt(ls1, ls2)
    return H[0].cpu().numpy()


# --------------------------------------------------------------------------- #
# warp + compositing helpers
# --------------------------------------------------------------------------- #
def warp_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Project an (N, 2) array of points through a 3x3 homography."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    homog = np.hstack([pts, np.ones((len(pts), 1))])
    out = (H @ homog.T).T
    w = out[:, 2:3]
    w[np.abs(w) < 1e-9] = 1e-9
    return out[:, :2] / w


def composite_under_players(frame: np.ndarray, graphics_layer: np.ndarray,
                            player_masks: np.ndarray | None,
                            alpha: float = 0.85) -> np.ndarray:
    """Blend a tactical `graphics_layer` onto `frame` but *beneath* players.

    The "Crucial Layering Fix": graphics are drawn on the grass first, then the
    original player pixels are re-pasted on top using their segmentation masks,
    so circles/lines stay under the boots.

    * frame:          HxWx3 BGR
    * graphics_layer: HxWx4 BGRA (alpha channel = where graphics exist)
    * player_masks:   HxW bool/uint8 union of player segmentation masks (or None
                      to skip the re-paste — graphics then sit on top).
    """
    out = frame.copy()
    if graphics_layer is None:
        return out
    g_rgb = graphics_layer[:, :, :3]
    g_a = (graphics_layer[:, :, 3:4].astype(np.float32) / 255.0) * float(alpha)
    out = (g_rgb.astype(np.float32) * g_a +
           out.astype(np.float32) * (1.0 - g_a)).astype(np.uint8)
    if player_masks is not None:
        m = player_masks.astype(bool)
        out[m] = frame[m]            # re-paste players on top of the graphics
    return out


def _apply_h(H, x, y):
    if H is None:
        return None
    v = H @ np.array([x, y, 1.0], dtype=np.float64)
    if abs(v[2]) < 1e-9:
        return None
    return float(v[0] / v[2]), float(v[1] / v[2])
