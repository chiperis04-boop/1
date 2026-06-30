"""Blueprint Module 4 (part B) — the Composer (FX, graphics, typography, slow-mo).

Takes a (already 9:16-reframed) clip plus the tracking, the Director's manifest
and the pitch homography, and produces the finished, branded highlight:

  1. **Grass-anchored graphics layer** — player halo (supervision CircleAnnotator
     under the hero), glowing ball trail (supervision TraceAnnotator), and
     optional tactical shapes warped by the homography so they lie on the pitch.
     Graphics are drawn first, then player pixels are re-pasted on top
     (composite_under_players) so nothing floats over the boots.
  2. **Audio-safe slow-motion** — only the decisive window
     (manifest.slomo_trigger_timestamp .. +slomo_duration) is stretched, using
     the ffmpeg filtergraph from the blueprint (setpts for video, atempo chain
     for pitch-preserving audio) and concatenated back with the rest.
  3. **Premium typography** — the hook text + data plates (Shot/Sprint/Beaten)
     rendered as high-fidelity overlays. Default engine is Pillow (TTF font +
     drop shadow + semi-transparent plate, no ImageMagick needed); a MoviePy
     TextClip engine is available via render.typography_engine = "moviepy".

Everything routes A/V through the project's `ff` helpers so audio stays valid.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..edit import ff
from ..utils.io import get_logger

log = get_logger()


@dataclass
class Composer:
    cfg: dict
    brand: dict

    # ------------------------------------------------------------------ API
    def compose(self, clip_path: str, out_path: str, track=None, manifest=None,
                homography=None, stats: dict | None = None, analytics=None) -> str:
        """Convenience: graphics + fx on a single clip.

        NOTE: graphics are drawn in `track`'s pixel space, so this assumes
        `clip_path` is in that same space. In the full studio pipeline the
        orchestrator instead calls draw_graphics() on the ORIGINAL clip, then
        reframes, then finish() — see src/studio_pipeline.py.
        """
        work = Path(out_path).with_suffix("")
        graphics_mp4 = f"{work}_gfx.mp4"
        cur = clip_path
        if self.cfg.get("telestration", {}).get("enabled", True) and track is not None:
            try:
                cur = self.draw_graphics(cur, graphics_mp4, track, manifest,
                                         homography, analytics)
            except Exception as exc:  # noqa: BLE001
                log.warning(f"[composer] graphics pass skipped: {exc}")
        return self.finish(cur, out_path, manifest, stats)

    def finish(self, clip_path: str, out_path: str, manifest=None,
               stats: dict | None = None, beats=None) -> str:
        """Space-independent passes: audio-safe slow-mo + premium typography.
        Safe to run on an already-reframed (9:16) clip.

        `beats` (list of agents.SlowmoBeat) enables MULTI-beat slow-mo from the
        Director; if omitted, falls back to the single manifest beat."""
        work = Path(out_path).with_suffix("")
        slowmo_mp4 = f"{work}_slomo.mp4"
        cur = clip_path
        slowmo_on = self.cfg.get("edit", {}).get("effects", {}).get("slowmo_on_key", True)
        if slowmo_on and beats:
            try:
                cur = self._slowmo_multi(cur, slowmo_mp4, beats)
            except Exception as exc:  # noqa: BLE001
                log.warning(f"[composer] multi-beat slow-mo skipped: {exc}")
        elif slowmo_on and manifest is not None:
            try:
                cur = self._slowmo_segment(
                    cur, slowmo_mp4,
                    trigger=float(getattr(manifest, "slomo_trigger_timestamp", 0.0)),
                    duration=float(getattr(manifest, "slomo_duration", 3.0)))
            except Exception as exc:  # noqa: BLE001
                log.warning(f"[composer] slow-mo skipped: {exc}")
        try:
            cur = self._typography(cur, out_path, manifest, stats)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"[composer] typography fell back to passthrough: {exc}")
            if cur != out_path:
                ff.standardize(cur, out_path, *self._profile_wh(),
                               self.cfg["render"]["fps"],
                               ff.pick_encoder(self.cfg["render"]["encoder"]))
        return out_path

    # -------------------------------------------------------- 1) graphics
    def make_annotators(self, track, manifest, analytics=None):
        """Build two per-frame annotators used by the single-pass renderer:

          * world(frame, idx)  — pitch/world-space graphics (player halos, ball
            trail) drawn in the ORIGINAL clip pixels, BEFORE the 9:16 crop.
          * screen(frame, idx) — HUD/screen-space graphics (the POSSESSION
            plate) drawn AFTER the crop, in OUTPUT pixels, so it sits correctly
            in the 9:16 frame instead of being cropped away.

        Merging annotation into the crop pass removes a whole lossy re-encode
        generation, and splitting world/screen fixes HUD positioning.
        """
        tele = self.cfg.get("telestration", {})
        hero_id = (getattr(analytics, "hero_id", None)
                   if analytics is not None else None) or self._hero_id(track, manifest)
        frames = getattr(track, "frames", [])
        ball_trail: list[tuple[float, float]] = []
        hero_state: dict = {}

        def world(frame, idx):
            ft = frames[idx] if idx < len(frames) else None
            return self._annotate_frame(frame, ft, hero_id, ball_trail, tele,
                                        analytics, hero_state)

        def screen(frame, idx):
            ft = frames[idx] if idx < len(frames) else None
            self._draw_possession_plate(frame, ft, analytics)
            return frame

        return world, screen

    def draw_graphics(self, clip_path, out_path, track, manifest, homography,
                      analytics=None):
        """Standalone graphics pass (used by compose() when there is no separate
        crop step). Draws world + screen graphics in the same pixel space."""
        import cv2

        fps = getattr(track, "fps", self.cfg["render"]["fps"]) or 30.0
        world, screen = self.make_annotators(track, manifest, analytics)
        cap = cv2.VideoCapture(clip_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        sink = ff.RawFrameSink(out_path, w, h, fps,
                               ff.pick_encoder(self.cfg["render"]["encoder"]),
                               audio_src=clip_path, intermediate=True)
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = world(frame, idx)
            frame = screen(frame, idx)
            sink.write(frame)
            idx += 1
        cap.release()
        sink.close()
        log.info(f"[composer] graphics layer -> {Path(out_path).name}")
        return out_path

    def _annotate_frame(self, frame, ft, hero_id, ball_trail, tele, analytics=None,
                        hero_state=None):
        import cv2
        if ft is None:
            return frame
        default_color = tuple(tele.get("spotlight_color", [0, 220, 255]))
        team_halos = tele.get("team_halos", True) and analytics is not None

        # team-coloured halo under EVERY player (club colours), if known
        if team_halos:
            for p in getattr(ft, "players", []):
                if p["id"] == hero_id:
                    continue
                col = analytics.color_for_track(p["id"], default_color)
                self._foot_ellipse(frame, p["xyxy"], col,
                                   max(1, tele.get("line_thickness", 4) - 2))

        # hero halo (thicker; team colour if available) + jersey number label.
        # Use a temporally-smoothed/persistent box so the halo doesn't flicker
        # when the hero detection drops for a few frames.
        if tele.get("spotlight_scorer", True) and hero_id is not None:
            hero = (_smooth_hero(ft, hero_id, hero_state, tele)
                    if hero_state is not None
                    else next((p for p in getattr(ft, "players", [])
                               if p["id"] == hero_id), None))
            if hero:
                col = (analytics.color_for_track(hero_id, default_color)
                       if analytics is not None else default_color)
                self._foot_ellipse(frame, hero["xyxy"], col,
                                   tele.get("line_thickness", 4) + 1)
                num = (analytics.jerseys.number_of.get(hero_id)
                       if analytics is not None else None)
                if num is not None:
                    x1, y1, x2, y2 = (int(v) for v in hero["xyxy"])
                    cv2.putText(frame, f"#{num}", (x1, max(12, y1 - 8)),
                                cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(frame, f"#{num}", (x1, max(12, y1 - 8)),
                                cv2.FONT_HERSHEY_DUPLEX, 0.9, col, 2, cv2.LINE_AA)

        # glowing ball trail
        if tele.get("ball_trail", True) and getattr(ft, "ball", None):
            bx, by = ft.ball["center"]
            ball_trail.append((bx, by))
            if len(ball_trail) > tele.get("trail_length", 30):
                ball_trail.pop(0)
            self._draw_trail(frame, ball_trail, tuple(tele.get("trail_color", [255, 255, 255])))
        return frame

    @staticmethod
    def _foot_ellipse(frame, xyxy, color, thickness):
        import cv2
        x1, y1, x2, y2 = (int(v) for v in xyxy)
        cx, cyb = int((x1 + x2) / 2), int(y2)
        axes = (max(8, int((x2 - x1) * 0.6)), max(4, int((x2 - x1) * 0.22)))
        cv2.ellipse(frame, (cx, cyb), axes, 0, 0, 360,
                    tuple(int(c) for c in color), thickness, cv2.LINE_AA)

    def _draw_possession_plate(self, frame, ft, analytics):
        """Live 'POSSESSION' plate when a confirmed possession run is active."""
        import cv2
        if ft is None or analytics is None:
            return
        run = analytics.possession.run_at(getattr(ft, "idx", -1))
        if run is None:
            return
        num = analytics.jerseys.number_of.get(run.track_id)
        label = "POSSESSION" + (f"  #{num}" if num is not None else "")
        col = analytics.color_for_track(run.track_id, (0, 220, 255))
        h, w = frame.shape[:2]
        x, y = int(w * 0.04), int(h * 0.06)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 10, y - th - 12), (x + tw + 14, y + 10),
                      (0, 0, 0), -1)
        cv2.rectangle(overlay, (x - 10, y - th - 12), (x - 4, y + 10),
                      tuple(int(c) for c in col), -1)        # team-colour accent
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, label, (x + 6, y), cv2.FONT_HERSHEY_DUPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)

    @staticmethod
    def _draw_trail(frame, pts, color):
        import cv2
        n = len(pts)
        for i in range(1, n):
            a = i / n
            thick = max(1, int(6 * a))
            p0 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
            p1 = (int(pts[i][0]), int(pts[i][1]))
            cv2.line(frame, p0, p1, color, thick, cv2.LINE_AA)

    # ---------------------------------------------------- 2) slow-motion
    def _slowmo_segment(self, clip_path, out_path, trigger, duration):
        """Stretch only [trigger, trigger+duration] and concat with the rest.

        Implements the blueprint's pitch-preserving recipe: video `setpts`,
        audio `atempo` (chained so factors <0.5 are legal). Falls back to
        returning the input unchanged if the window is degenerate.
        """
        total = ff.duration(clip_path)
        factor = float(self.cfg["edit"]["effects"].get("slowmo_factor", 0.4))
        factor = min(max(factor, 0.1), 1.0)
        t0 = max(0.0, min(trigger, total))
        t1 = max(t0, min(trigger + duration, total))
        if total <= 0 or (t1 - t0) < 0.3:
            return clip_path

        v_setpts = 1.0 / factor          # e.g. factor 0.4 -> setpts 2.5*PTS
        atempo = _atempo_chain(factor)
        has_audio = ff.has_audio(clip_path)
        out_fps = int(self.cfg["render"]["fps"])
        eff = self.cfg.get("edit", {}).get("effects", {})
        interp = bool(eff.get("slowmo_interpolate", True))
        quality = eff.get("slowmo_interpolation_quality", "mci")
        encoder = ff.pick_encoder(self.cfg["render"]["encoder"])

        def _filtergraph(use_interp: bool) -> str:
            slow = f"setpts={v_setpts:.4f}*(PTS-STARTPTS)"
            if use_interp:
                slow += "," + ff.minterpolate_expr(out_fps, quality)
            v = [
                f"[0:v]trim=0:{t0:.3f},setpts=PTS-STARTPTS[v0]",
                f"[0:v]trim={t0:.3f}:{t1:.3f},{slow}[v1]",
                f"[0:v]trim={t1:.3f},setpts=PTS-STARTPTS[v2]",
            ]
            parts = "[v0][v1][v2]"
            a_filters = []
            if has_audio:
                a_filters = [
                    f"[0:a]atrim=0:{t0:.3f},asetpts=PTS-STARTPTS[a0]",
                    f"[0:a]atrim={t0:.3f}:{t1:.3f},asetpts=PTS-STARTPTS,{atempo}[a1]",
                    f"[0:a]atrim={t1:.3f},asetpts=PTS-STARTPTS[a2]",
                ]
                parts = "[v0][a0][v1][a1][v2][a2]"
            concat = (f"{parts}concat=n=3:v=1:a={1 if has_audio else 0}"
                      f"[vout]" + ("[aout]" if has_audio else ""))
            return ";".join(v + a_filters + [concat])

        def _encode(use_interp: bool) -> None:
            cmd = ["ffmpeg", "-y", "-i", clip_path,
                   "-filter_complex", _filtergraph(use_interp), "-map", "[vout]"]
            if has_audio:
                cmd += ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]
            # intermediate quality: typography runs after this, branding after that
            cmd += [*ff.venc_args(encoder, intermediate=True), out_path]
            ff.run(cmd, desc="slowmo segment")

        try:
            _encode(interp)
        except ff.FFmpegError as exc:
            if interp:
                log.warning(f"[composer] minterpolate slow-mo failed ({exc}); "
                            "retrying with plain frame-stretch")
                _encode(False)
            else:
                raise
        log.info(f"[composer] slow-mo {factor}x on [{t0:.1f}-{t1:.1f}]s "
                 f"{'(interpolated) ' if interp else ''}-> {Path(out_path).name}")
        return out_path

    # ---------------------------------------------------- 2b) multi-beat slow-mo
    def _slowmo_multi(self, clip_path, out_path, beats):
        """Apply several slow-mo windows (the Director's decisive beats) in one
        pass: the timeline is split into alternating normal/slow segments and
        concatenated. Pitch-preserving audio (atempo chain); motion-interpolated
        video (with graceful fallback to plain stretch)."""
        total = ff.duration(clip_path)
        if total <= 0:
            return clip_path
        segs = _segments_from_beats(beats, total)
        if not any(f is not None for _, _, f in segs):
            return clip_path
        has_audio = ff.has_audio(clip_path)
        out_fps = int(self.cfg["render"]["fps"])
        eff = self.cfg.get("edit", {}).get("effects", {})
        interp = bool(eff.get("slowmo_interpolate", True))
        quality = eff.get("slowmo_interpolation_quality", "mci")
        encoder = ff.pick_encoder(self.cfg["render"]["encoder"])
        n = len(segs)

        def _filtergraph(use_interp: bool) -> str:
            v, a = [], []
            for i, (s, e, f) in enumerate(segs):
                if f is None:
                    v.append(f"[0:v]trim={s:.3f}:{e:.3f},setpts=PTS-STARTPTS[v{i}]")
                else:
                    slow = f"setpts={1.0 / f:.4f}*(PTS-STARTPTS)"
                    if use_interp:
                        slow += "," + ff.minterpolate_expr(out_fps, quality)
                    v.append(f"[0:v]trim={s:.3f}:{e:.3f},{slow}[v{i}]")
                if has_audio:
                    af = (f"[0:a]atrim={s:.3f}:{e:.3f},asetpts=PTS-STARTPTS"
                          + (f",{_atempo_chain(f)}" if f is not None else "")
                          + f"[a{i}]")
                    a.append(af)
            if has_audio:
                join = "".join(f"[v{i}][a{i}]" for i in range(n))
                concat = f"{join}concat=n={n}:v=1:a=1[vout][aout]"
            else:
                join = "".join(f"[v{i}]" for i in range(n))
                concat = f"{join}concat=n={n}:v=1:a=0[vout]"
            return ";".join(v + a + [concat])

        def _encode(use_interp: bool) -> None:
            cmd = ["ffmpeg", "-y", "-i", clip_path,
                   "-filter_complex", _filtergraph(use_interp), "-map", "[vout]"]
            if has_audio:
                cmd += ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]
            cmd += [*ff.venc_args(encoder, intermediate=True), out_path]
            ff.run(cmd, desc="slowmo multi")

        try:
            _encode(interp)
        except ff.FFmpegError as exc:
            if interp:
                log.warning(f"[composer] multi-beat minterpolate failed ({exc}); "
                            "retrying plain stretch")
                _encode(False)
            else:
                raise
        n_slow = sum(1 for _, _, f in segs if f is not None)
        log.info(f"[composer] {n_slow} slow-mo beat(s) -> {Path(out_path).name}")
        return out_path

    # ---------------------------------------------------- 3) typography
    def _typography(self, clip_path, out_path, manifest, stats):
        engine = (self.cfg.get("render", {}).get("typography_engine") or "pillow").lower()
        hook = (getattr(manifest, "video_hook_text", "") or "").strip()
        plates = self._stat_plates(stats)
        if engine == "moviepy":
            return self._typography_moviepy(clip_path, out_path, hook, plates)
        return self._typography_pillow(clip_path, out_path, hook, plates)

    def _typography_pillow(self, clip_path, out_path, hook, plates):
        """Premium overlay via Pillow: TTF font, drop shadow, alpha plates.
        Reliable (no ImageMagick) and the default engine."""
        import cv2
        from PIL import Image, ImageDraw, ImageFont

        font_path = self._font_path()
        cap = cv2.VideoCapture(clip_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or self.cfg["render"]["fps"]
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        hook_font = _load_font(font_path, int(h * 0.045))
        plate_font = _load_font(font_path, int(h * 0.026))

        sink = ff.RawFrameSink(out_path, w, h, fps,
                               ff.pick_encoder(self.cfg["render"]["encoder"]),
                               audio_src=clip_path)
        hook_frames = int(fps * 2.0)      # show hook ~2s

        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
            draw = ImageDraw.Draw(img, "RGBA")
            if hook and idx < hook_frames:
                _draw_hook(draw, img.size, hook, hook_font)
            _draw_plates(draw, img.size, plates, plate_font)
            frame = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
            sink.write(frame)
            idx += 1
            _ = total
        cap.release()
        sink.close()
        log.info(f"[composer] pillow typography -> {Path(out_path).name}")
        return out_path

    def _typography_moviepy(self, clip_path, out_path, hook, plates):
        """Blueprint-requested MoviePy TextClip engine. Handles both MoviePy 1.x
        and 2.x signatures; raises on failure so compose() can fall back.

        NOTE: do NOT import `moviepy.editor` here — it exists only in MoviePy 1.x
        and was removed in 2.x. The version-agnostic `_moviepy_*` shims below
        import the concrete classes (trying 2.x first, then 1.x), so this engine
        works on whichever MoviePy is installed."""
        base = _moviepy_videoclip(clip_path)
        overlays = [base]
        W, H = base.size
        font = self._font_path()
        if hook:
            overlays.append(_moviepy_text(hook, fontsize=int(H * 0.05), font=font,
                                          pos=("center", int(H * 0.10)),
                                          duration=min(2.0, base.duration)))
        y = int(H * 0.58)
        for label in plates:
            overlays.append(_moviepy_text(label, fontsize=int(H * 0.03), font=font,
                                          pos=(int(W * 0.06), y),
                                          duration=base.duration))
            y += int(H * 0.05)
        comp = _moviepy_composite(overlays)
        comp.write_videofile(out_path, codec="libx264", audio_codec="aac",
                             fps=self.cfg["render"]["fps"], logger=None)
        log.info(f"[composer] moviepy typography -> {Path(out_path).name}")
        return out_path

    # ------------------------------------------------------------- helpers
    def _hero_id(self, track, manifest):
        return getattr(track, "key_track_id", None) or getattr(track, "hero_id", None)

    def _stat_plates(self, stats: dict | None) -> list[str]:
        if not stats:
            return []
        out = []
        if stats.get("possession_pct"):
            pp = stats["possession_pct"]
            out.append("POSSESSION  " + " / ".join(f"{v}%" for v in pp))
        if "shot_distance_m" in stats:
            out.append(f"SHOT  {stats['shot_distance_m']:.0f} M")
        if "sprint_distance_m" in stats:
            out.append(f"SPRINT  {stats['sprint_distance_m']:.0f} M")
        if "players_beaten" in stats:
            out.append(f"BEATEN  {stats['players_beaten']}")
        if "top_speed_kmh" in stats:
            out.append(f"TOP SPEED  {stats['top_speed_kmh']:.0f} KM/H")
        return out

    def _font_path(self):
        cap = self.cfg.get("edit", {}).get("captions", {})
        font = self.brand.get("font") or cap.get("font", "assets/fonts/Inter-Bold.ttf")
        return font if font and Path(font).exists() else None

    def _profile_wh(self):
        p = self.cfg.get("_active_profile", {"width": 1080, "height": 1920})
        return p["width"], p["height"]


# --------------------------------------------------------------------------- #
# functional entrypoint
# --------------------------------------------------------------------------- #
def compose_highlight(clip_path: str, out_path: str, cfg: dict, brand: dict,
                      track=None, manifest=None, homography=None,
                      stats: dict | None = None) -> str:
    return Composer(cfg, brand).compose(clip_path, out_path, track=track,
                                        manifest=manifest, homography=homography,
                                        stats=stats)


# --------------------------------------------------------------------------- #
# pillow text rendering
# --------------------------------------------------------------------------- #
def _smooth_hero(ft, hero_id, state: dict, tele: dict):
    """Return a temporally-smoothed/persistent hero box {'xyxy': [...]}.

    Detection flickers frame-to-frame; without this the hero halo blinks. We
    EMA-smooth the box when the hero is seen and HOLD the last box for up to
    `halo_hold_frames` when the detection drops, so the halo glides and persists
    instead of popping.
    """
    import numpy as np
    hold = int(tele.get("halo_hold_frames", 6))
    alpha = float(tele.get("halo_smooth", 0.5))
    cur = None
    if ft is not None:
        cur = next((p for p in getattr(ft, "players", []) if p["id"] == hero_id), None)
    if cur is not None:
        box = np.asarray(cur["xyxy"], dtype=np.float64)
        if state.get("box") is not None:
            box = alpha * np.asarray(state["box"], dtype=np.float64) + (1 - alpha) * box
        state["box"] = box.tolist()
        state["miss"] = 0
        return {"xyxy": state["box"]}
    # hero missing this frame: persist the last known box for a short while
    if state.get("box") is not None and state.get("miss", 0) < hold:
        state["miss"] = state.get("miss", 0) + 1
        return {"xyxy": state["box"]}
    return None


def _load_font(path, size):
    from PIL import ImageFont
    try:
        if path:
            return ImageFont.truetype(path, size)
    except Exception:  # noqa: BLE001
        pass
    return ImageFont.load_default()


def _text_size(draw, text, font):
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return r - l, b - t
    except Exception:  # noqa: BLE001
        return draw.textlength(text, font=font), font.size


def _draw_hook(draw, size, text, font):
    W, H = size
    tw, th = _text_size(draw, text, font)
    x = (W - tw) // 2
    y = int(H * 0.10)
    pad = int(th * 0.5)
    # semi-transparent plate behind the hook
    draw.rounded_rectangle([x - pad, y - pad, x + tw + pad, y + th + pad],
                           radius=pad, fill=(0, 0, 0, 140))
    # drop shadow then text
    draw.text((x + 3, y + 3), text, font=font, fill=(0, 0, 0, 200))
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))


def _draw_plates(draw, size, plates, font):
    if not plates:
        return
    W, H = size
    x = int(W * 0.06)
    # stat plates sit in the social safe-zone (above the bottom ~20% UI band)
    y = int(H * 0.58)
    for label in plates:
        tw, th = _text_size(draw, label, font)
        pad = int(th * 0.45)
        draw.rounded_rectangle([x - pad, y - pad, x + tw + pad, y + th + pad],
                               radius=int(pad * 0.8), fill=(0, 0, 0, 130))
        draw.text((x + 2, y + 2), label, font=font, fill=(0, 0, 0, 200))
        draw.text((x, y), label, font=font, fill=(0, 220, 255, 255))
        y += th + int(th * 1.1)


# --------------------------------------------------------------------------- #
# moviepy compatibility shims (1.x vs 2.x)
# --------------------------------------------------------------------------- #
def _moviepy_videoclip(path):
    try:
        from moviepy import VideoFileClip            # moviepy 2.x
    except Exception:  # noqa: BLE001
        from moviepy.editor import VideoFileClip     # moviepy 1.x
    return VideoFileClip(path)


def _moviepy_text(text, fontsize, font, pos, duration):
    try:
        from moviepy import TextClip                 # 2.x
        kw = {"text": text, "font_size": fontsize, "color": "white"}
        if font:
            kw["font"] = font
        clip = TextClip(**kw)
    except Exception:  # noqa: BLE001
        from moviepy.editor import TextClip           # 1.x
        kw = {"txt": text, "fontsize": fontsize, "color": "white"}
        if font:
            kw["font"] = font
        clip = TextClip(**kw)
    for setter in ("with_position", "set_position"):
        if hasattr(clip, setter):
            clip = getattr(clip, setter)(pos); break
    for setter in ("with_duration", "set_duration"):
        if hasattr(clip, setter):
            clip = getattr(clip, setter)(duration); break
    return clip


def _moviepy_composite(clips):
    try:
        from moviepy import CompositeVideoClip        # 2.x
    except Exception:  # noqa: BLE001
        from moviepy.editor import CompositeVideoClip  # 1.x
    return CompositeVideoClip(clips)


# --------------------------------------------------------------------------- #
def _segments_from_beats(beats, total: float):
    """Turn slow-mo beats into a contiguous segment list tiling [0,total].

    Returns [(start, end, factor_or_None)] where None = play at normal speed.
    Beats are clamped to [0,total], sorted, and overlaps are clipped so the
    segments never overlap.
    """
    norm = []
    for b in beats or []:
        s = max(0.0, float(getattr(b, "start", 0.0)))
        e = min(total, float(getattr(b, "end", 0.0)))
        f = float(getattr(b, "factor", 0.4))
        if e > s:
            norm.append((s, e, min(max(f, 0.1), 1.0)))
    norm.sort(key=lambda x: x[0])
    # clip overlaps
    clipped = []
    cursor = 0.0
    for s, e, f in norm:
        s = max(s, cursor)
        if e <= s:
            continue
        clipped.append((s, e, f))
        cursor = e
    # build full timeline with normal-speed gaps
    segs = []
    cursor = 0.0
    for s, e, f in clipped:
        if s > cursor + 1e-3:
            segs.append((cursor, s, None))
        segs.append((s, e, f))
        cursor = e
    if cursor < total - 1e-3:
        segs.append((cursor, total, None))
    return segs or [(0.0, total, None)]


def _atempo_chain(factor: float) -> str:
    """Build an atempo filter chain for an arbitrary slow factor.

    ffmpeg's atempo only accepts 0.5..2.0 per instance, so factors below 0.5 are
    realised by chaining (e.g. 0.4 -> atempo=0.5,atempo=0.8)."""
    f = max(0.1, min(factor, 2.0))
    chain = []
    remaining = f
    while remaining < 0.5 - 1e-6:
        chain.append(0.5)
        remaining /= 0.5
    chain.append(round(remaining, 4))
    return ",".join(f"atempo={c}" for c in chain)
