"""Premium look-up — model restoration + a cinematic finishing grade.

Simply resizing a clip to 1080p does NOT make it look premium; a low-detail
broadcast crop stays soft. So this stage works in two complementary layers:

  1. MODEL restoration (opt-in, GPU) — Real-ESRGAN super-resolution + denoise
     reconstructs real high-frequency detail (and, optionally, GFPGAN face
     restoration for close-ups) instead of interpolating pixels. It is heavy
     (per-frame inference), so it is opt-in and honestly a "verify on GPU"
     path. If the package/weights are unavailable it degrades to the finishing
     grade below with a clear log — it NEVER fakes an "enhanced" result.

  2. FINISHING GRADE (always, ffmpeg) — a broadcast-style colour grade +
     contrast-adaptive sharpen + light temporal denoise that lifts a flat crop
     toward a punchy social-media look. Cheap, deterministic, no model needed;
     it also polishes the model pass output.

Config (config.yaml -> `enhance:`):
    enabled: false          # master switch (studio calls this only when true)
    backend: auto           # auto | realesrgan | grade
    model: realesr-general-x4v3
    scale: 2                # model upscale factor (frames are resized back)
    tile: 512               # tile size for limited VRAM (0 = whole frame)
    face_enhance: false     # GFPGAN face restoration (close-ups)
    half: true              # fp16 inference on GPU
    grade:                  # the always-available cinematic finish
      enabled: true
      contrast: 1.06
      brightness: 0.02
      saturation: 1.12
      gamma: 0.98
      sharpen: 0.8
      denoise: true
"""
from __future__ import annotations

from pathlib import Path

from ..edit import ff
from ..utils.io import get_logger, resolve_device

log = get_logger()


class _ModelUnavailable(Exception):
    """Raised when the Real-ESRGAN package/weights can't be loaded, so the
    caller degrades to the ffmpeg finishing grade (honest, not a fake pass)."""


class Enhancer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.e = cfg.get("enhance", {}) or {}

    # ------------------------------------------------------------------ public
    def enhance(self, in_path: str, out_path: str) -> str:
        """Return an enhanced clip. Tries the model path (when requested and
        available), always finishing with the cinematic grade. On any model
        failure it degrades to a grade-only pass so the render never breaks."""
        backend = (self.e.get("backend") or "auto").lower()
        if backend in ("auto", "realesrgan", "model"):
            try:
                return self._model_enhance(in_path, out_path)
            except _ModelUnavailable as exc:
                log.info(f"[enhance] model restoration unavailable ({exc}); "
                         "applying cinematic finishing grade only")
            except Exception as exc:  # noqa: BLE001
                log.warning(f"[enhance] model restoration failed ({exc}); "
                            "falling back to cinematic finishing grade")
        return self._grade(in_path, out_path)

    # ---------------------------------------------------------- model (GPU)
    def _model_enhance(self, in_path: str, out_path: str) -> str:
        """Real-ESRGAN per-frame super-resolution/denoise, frames resized back
        to the source resolution (detail reconstruction, not a bigger canvas),
        then the finishing grade. GPU-heavy: mark 'verify on GPU'."""
        import cv2

        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except Exception as exc:  # noqa: BLE001 -> package not installed
            raise _ModelUnavailable(f"realesrgan/basicsr import: {exc}")

        scale = int(self.e.get("scale", 2))
        model_name = str(self.e.get("model", "realesr-general-x4v3"))
        device = resolve_device(self.cfg.get("vision", {}).get("device", "cuda"))
        model, weights_url, netscale = _build_realesr_model(RRDBNet, model_name)
        try:
            upsampler = RealESRGANer(
                scale=netscale, model_path=weights_url, model=model,
                tile=int(self.e.get("tile", 512)), tile_pad=10, pre_pad=0,
                half=bool(self.e.get("half", True)) and device == "cuda",
                device=device)
        except Exception as exc:  # noqa: BLE001 -> weights download / CUDA issue
            raise _ModelUnavailable(f"RealESRGANer init: {exc}")

        face_enhancer = None
        if bool(self.e.get("face_enhance", False)):
            try:
                from gfpgan import GFPGANer
                face_enhancer = GFPGANer(
                    model_path=("https://github.com/TencentARC/GFPGAN/releases/"
                                "download/v1.3.0/GFPGANv1.3.pth"),
                    upscale=scale, arch="clean", channel_multiplier=2,
                    bg_upsampler=upsampler)
            except Exception as exc:  # noqa: BLE001
                log.warning(f"[enhance] face restoration unavailable ({exc}); "
                            "super-resolution only")

        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            raise _ModelUnavailable(f"cannot open {in_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or float(self.cfg["render"]["fps"])
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        encoder = ff.pick_encoder(self.cfg.get("render", {}).get("encoder",
                                                                 "libx264"))
        sr_tmp = str(Path(out_path).with_suffix("")) + "_sr.mp4"
        sink = ff.RawFrameSink(sr_tmp, w, h, fps, encoder, audio_src=in_path,
                               intermediate=True)
        n = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if face_enhancer is not None:
                    _, _, sr = face_enhancer.enhance(
                        frame, has_aligned=False, only_center_face=False,
                        paste_back=True)
                else:
                    sr, _ = upsampler.enhance(frame, outscale=scale)
                if sr.shape[1] != w or sr.shape[0] != h:
                    sr = cv2.resize(sr, (w, h), interpolation=cv2.INTER_AREA)
                sink.write(sr)
                n += 1
        finally:
            cap.release()
            sink.close()
        log.info(f"[enhance] Real-ESRGAN restored {n} frames "
                 f"(model={model_name}, verify on GPU) -> finishing grade")
        return self._grade(sr_tmp, out_path)

    # ---------------------------------------------------- finishing grade
    def _grade(self, in_path: str, out_path: str) -> str:
        """Broadcast-style colour grade + adaptive sharpen + light denoise. Pure
        ffmpeg, so it runs everywhere and never changes the resolution."""
        g = self.e.get("grade", {}) or {}
        if not g.get("enabled", True):
            return in_path

        contrast = float(g.get("contrast", 1.06))
        brightness = float(g.get("brightness", 0.02))
        saturation = float(g.get("saturation", 1.12))
        gamma = float(g.get("gamma", 0.98))
        sharpen = float(g.get("sharpen", 0.8))

        filters: list[str] = []
        if g.get("denoise", True):
            filters.append("hqdn3d=1.5:1.5:6:6")   # light spatial+temporal denoise
        filters.append(
            f"eq=contrast={contrast}:brightness={brightness}:"
            f"saturation={saturation}:gamma={gamma}")
        if sharpen > 0:
            filters.append(f"unsharp=5:5:{sharpen}:5:5:0.0")
        filters.append("format=yuv420p")
        vf = ",".join(filters)

        encoder = ff.pick_encoder(self.cfg.get("render", {}).get("encoder",
                                                                 "libx264"))
        cmd = ["ffmpeg", "-y", "-i", in_path, "-vf", vf, *ff.venc_args(encoder)]
        if ff.has_audio(in_path):
            cmd += ["-c:a", "copy"]
        cmd += [out_path]
        ff.run(cmd, desc="enhance grade")
        log.info(f"[enhance] cinematic grade -> {Path(out_path).name}")
        return out_path


# --------------------------------------------------------------------------- #
def _build_realesr_model(RRDBNet, name: str):
    """Return (arch, weights_url, netscale) for a supported Real-ESRGAN model.

    'realesr-general-x4v3' is the best general-purpose choice for real-world
    (noisy/compressed) footage; 'RealESRGAN_x4plus' is the classic photographic
    model. The SRVGG general model is loaded lazily to avoid importing it unless
    selected."""
    n = (name or "").lower()
    base = ("https://github.com/xinntao/Real-ESRGAN/releases/download/")
    if "general" in n:
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64,
                                num_conv=32, upscale=4, act_type="prelu")
        url = base + "v0.2.5.0/realesr-general-x4v3.pth"
        return model, url, 4
    if "anime" in n:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6,
                        num_grow_ch=32, scale=4)
        url = base + "v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
        return model, url, 4
    # default: RealESRGAN_x4plus (photographic)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                    num_grow_ch=32, scale=4)
    url = base + "v0.1.0/RealESRGAN_x4plus.pth"
    return model, url, 4
