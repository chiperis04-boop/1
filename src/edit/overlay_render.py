"""Professional HUD rendering with Pillow + alpha-composited via FFmpeg.

This module is the single typography/graphics engine for every on-screen text
element (hooks, lower-thirds, stat cards, captions, watermark, intro/outro
cards). It deliberately replaces the old ``cv2.putText`` / ffmpeg ``drawtext``
approach, which produced an amateur look and — on builds without libfreetype —
did not work at all.

Why Pillow instead of drawtext:
  * real font shaping with the bundled **Inter** typeface (a modern sans well
    suited to sports graphics), with a crisp stroke + drop shadow so text stays
    legible over both bright grass and white kits;
  * rounded, semi-transparent dark "cards" behind metrics so they read as a
    designed HUD rather than floating text;
  * full control over **safe zones** so our graphics never collide with the
    broadcaster's own scoreboard / logos.

Each element is rasterised once to a full-frame RGBA PNG (transparent
background, the widget drawn at its final position). Compositing is then a
straightforward FFmpeg ``overlay`` per element, gated by ``enable='between(t,
start,end)'`` and given smooth alpha ``fade`` in/out — so timing and fades are
frame-accurate and driven entirely by event timecodes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from ..utils.io import get_logger
from . import ff

log = get_logger()

# A bundled, guaranteed-present font. Branding may override the family but we
# always have something real to fall back to (never the blocky PIL bitmap).
_BUNDLED_FONT = "assets/fonts/Inter-Bold.ttf"
_DEJAVU_FALLBACKS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
)


# --------------------------------------------------------------------------- #
# element model
# --------------------------------------------------------------------------- #
@dataclass
class Overlay:
    """One time-bounded full-frame RGBA PNG to composite over the clip."""
    png: str
    start: float = 0.0
    end: float = 1e9
    fade_in: float = 0.25
    fade_out: float = 0.25


@dataclass
class SafeZones:
    """Fractional keep-out bands where broadcaster graphics usually live.

    Custom HUD elements are positioned *inside* the remaining safe band so they
    never clip the match score or channel logos baked into the source feed.
    """
    top: float = 0.10        # top 10% (broadcast scorebug / clock)
    bottom: float = 0.10     # bottom 10% (sponsor / ticker)

    def band(self, height: int) -> tuple[int, int]:
        return int(height * self.top), int(height * (1.0 - self.bottom))


def zones_from_cfg(cfg: dict) -> SafeZones:
    sz = ((cfg.get("render", {}) or {}).get("safe_zones", {}) or {})
    return SafeZones(top=float(sz.get("top", 0.10)),
                     bottom=float(sz.get("bottom", 0.10)))


def clean_text(text: str | None) -> str:
    """Drop glyphs the bundled sans can't render (emoji / pictographs) so HUD
    text never shows tofu boxes, and normalise whitespace."""
    if not text:
        return ""
    out = []
    for ch in text:
        cp = ord(ch)
        emoji = (0x1F000 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF
                 or 0x2B00 <= cp <= 0x2BFF or 0xFE00 <= cp <= 0xFE0F
                 or cp in (0x200D, 0x20E3))
        if not emoji:
            out.append(ch)
    return " ".join("".join(out).split())


# --------------------------------------------------------------------------- #
# font handling
# --------------------------------------------------------------------------- #
def resolve_font_path(preferred: str | None = None) -> str | None:
    candidates = [preferred, _BUNDLED_FONT, *_DEJAVU_FALLBACKS]
    for c in candidates:
        if c and Path(c).exists():
            return c
    return None


@lru_cache(maxsize=64)
def _font(path: str | None, size: int):
    from PIL import ImageFont
    if path and Path(path).exists():
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


# --------------------------------------------------------------------------- #
# low-level canvas
# --------------------------------------------------------------------------- #
class Canvas:
    """A transparent, frame-sized RGBA canvas with HUD drawing helpers."""

    def __init__(self, width: int, height: int, font_path: str | None = None):
        from PIL import Image, ImageDraw
        self.w, self.h = width, height
        self.font_path = resolve_font_path(font_path)
        self.img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        self.draw = ImageDraw.Draw(self.img)

    # -- measuring -------------------------------------------------------- #
    def text_wh(self, text: str, size: int, stroke: int = 0) -> tuple[int, int]:
        font = _font(self.font_path, size)
        l, t, r, b = self.draw.textbbox((0, 0), text, font=font,
                                        stroke_width=stroke)
        return r - l, b - t

    def wrap(self, text: str, size: int, max_w: int) -> list[str]:
        """Greedy word-wrap so long hooks/captions fit inside the safe band."""
        words = text.split()
        if not words:
            return []
        lines, cur = [], words[0]
        for w in words[1:]:
            if self.text_wh(cur + " " + w, size)[0] <= max_w:
                cur += " " + w
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
        return lines

    # -- primitives ------------------------------------------------------- #
    def rounded_card(self, box: tuple[int, int, int, int], radius: int,
                     fill=(12, 14, 18, 170), accent=None, accent_w: int = 0):
        x0, y0, x1, y1 = box
        self.draw.rounded_rectangle(box, radius=radius, fill=fill)
        if accent and accent_w > 0:
            # a slim accent bar down the left edge = signature channel colour
            self.draw.rounded_rectangle(
                (x0, y0, x0 + accent_w, y1),
                radius=accent_w // 2 if accent_w > 2 else 0, fill=accent)

    def text(self, xy: tuple[int, int], text: str, size: int,
             color=(255, 255, 255, 255), anchor: str = "la",
             stroke: int = 0, stroke_fill=(0, 0, 0, 235),
             shadow: tuple[int, int] | None = (0, 3)):
        font = _font(self.font_path, size)
        if shadow is not None:
            self.draw.text((xy[0] + shadow[0], xy[1] + shadow[1]), text,
                           font=font, fill=(0, 0, 0, 150), anchor=anchor,
                           stroke_width=stroke, stroke_fill=(0, 0, 0, 150))
        self.draw.text(xy, text, font=font, fill=color, anchor=anchor,
                       stroke_width=stroke, stroke_fill=stroke_fill)

    def save(self, path: str) -> str:
        self.img.save(path)
        return path


# --------------------------------------------------------------------------- #
# colour helpers (config stores BGR for OpenCV; PIL wants RGB)
# --------------------------------------------------------------------------- #
def bgr_to_rgba(bgr, alpha: int = 255) -> tuple[int, int, int, int]:
    if not bgr or len(bgr) < 3:
        return (0, 220, 255, alpha)
    b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
    return (r, g, b, alpha)


# --------------------------------------------------------------------------- #
# high-level widgets — each returns an Overlay (or None)
# --------------------------------------------------------------------------- #
def hook_overlay(text: str, w: int, h: int, font: str, out_png: str,
                 zones: SafeZones, accent=(0, 220, 255),
                 start: float = 0.0, end: float = 1.6) -> Overlay | None:
    text = clean_text(text)
    if not text:
        return None
    c = Canvas(w, h, font)
    top, _ = zones.band(h)
    size = max(34, int(w * 0.058))
    max_w = int(w * 0.80)
    lines = c.wrap(text, size, max_w)
    line_h = c.text_wh("Ag", size)[1] + int(size * 0.32)
    block_h = line_h * len(lines)
    pad = int(size * 0.7)
    y0 = top + int(h * 0.04)
    # card behind the hook
    block_w = max(c.text_wh(ln, size, stroke=2)[0] for ln in lines)
    cx = w // 2
    c.rounded_card((cx - block_w // 2 - pad, y0 - pad,
                    cx + block_w // 2 + pad, y0 + block_h + pad),
                   radius=int(pad * 0.9), fill=(10, 12, 16, 150),
                   accent=bgr_to_rgba(accent), accent_w=max(6, w // 150))
    y = y0
    for ln in lines:
        c.text((cx, y), ln, size, anchor="ma", stroke=2)
        y += line_h
    c.save(out_png)
    return Overlay(png=out_png, start=start, end=end, fade_in=0.2, fade_out=0.3)


def lower_third_overlay(label: str, w: int, h: int, font: str, out_png: str,
                        zones: SafeZones, accent=(0, 220, 255),
                        start: float = 0.0, end: float = 1e9) -> Overlay | None:
    label = clean_text(label)
    if not label:
        return None
    c = Canvas(w, h, font)
    _, bottom = zones.band(h)
    size = max(30, int(w * 0.044))
    tw, th = c.text_wh(label, size)
    pad_x, pad_y = int(size * 0.6), int(size * 0.42)
    x0 = int(w * 0.055)
    y1 = bottom - int(h * 0.02)
    y0 = y1 - (th + pad_y * 2)
    c.rounded_card((x0, y0, x0 + tw + pad_x * 2 + 14, y1),
                   radius=int((y1 - y0) * 0.28), fill=(12, 14, 18, 185),
                   accent=bgr_to_rgba(accent), accent_w=max(7, w // 130))
    c.text((x0 + pad_x + 14, y0 + pad_y), label, size, anchor="la", stroke=1)
    c.save(out_png)
    return Overlay(png=out_png, start=start, end=end, fade_in=0.25, fade_out=0.3)


def stats_card_overlay(lines: list[str], w: int, h: int, font: str,
                       out_png: str, zones: SafeZones, accent=(0, 220, 255),
                       title: str = "BREAKDOWN",
                       start: float = 0.0, end: float = 1e9) -> Overlay | None:
    lines = [clean_text(ln) for ln in lines if clean_text(ln)]
    if not lines:
        return None
    c = Canvas(w, h, font)
    top, _ = zones.band(h)
    tsize = max(22, int(w * 0.030))
    lsize = max(26, int(w * 0.038))
    pad = int(lsize * 0.55)
    gap = int(lsize * 0.5)
    title_h = c.text_wh(title, tsize)[1]
    line_h = c.text_wh("Ag", lsize)[1] + gap
    body_w = max([c.text_wh(title, tsize)[0]] +
                 [c.text_wh(ln, lsize)[0] for ln in lines])
    card_w = body_w + pad * 2 + 14
    card_h = pad * 2 + title_h + int(gap * 1.2) + line_h * len(lines)
    x1 = w - int(w * 0.045)
    x0 = x1 - card_w
    y0 = top + int(h * 0.11)
    c.rounded_card((x0, y0, x1, y0 + card_h), radius=int(pad * 0.9),
                   fill=(12, 14, 18, 180), accent=bgr_to_rgba(accent),
                   accent_w=max(7, w // 130))
    tx = x0 + pad + 14
    c.text((tx, y0 + pad), title, tsize, color=bgr_to_rgba(accent),
           anchor="la", stroke=0, shadow=(0, 2))
    y = y0 + pad + title_h + int(gap * 1.2)
    for ln in lines:
        c.text((tx, y), ln, lsize, anchor="la", stroke=1)
        y += line_h
    c.save(out_png)
    return Overlay(png=out_png, start=start, end=end, fade_in=0.3, fade_out=0.3)


def watermark_overlay(text: str, w: int, h: int, font: str, out_png: str,
                      zones: SafeZones, opacity: float = 0.6) -> Overlay | None:
    text = clean_text(text)
    if not text:
        return None
    c = Canvas(w, h, font)
    top, _ = zones.band(h)
    size = max(20, int(w * 0.030))
    alpha = int(max(0.0, min(1.0, opacity)) * 255)
    tw, _ = c.text_wh(text, size)
    c.text((w - int(w * 0.04) - tw, top + int(h * 0.005)), text, size,
           color=(255, 255, 255, alpha), anchor="la", stroke=0, shadow=(0, 2))
    c.save(out_png)
    return Overlay(png=out_png, start=0.0, end=1e9, fade_in=0.1, fade_out=0.1)


def caption_overlay(text: str, w: int, h: int, font: str, out_png: str,
                    zones: SafeZones, start: float, end: float) -> Overlay | None:
    text = clean_text(text)
    if not text:
        return None
    c = Canvas(w, h, font)
    _, bottom = zones.band(h)
    size = max(34, int(w * 0.058))
    max_w = int(w * 0.9)
    lines = c.wrap(text, size, max_w)
    line_h = c.text_wh("Ag", size, stroke=2)[1] + int(size * 0.22)
    block_h = line_h * len(lines)
    # captions sit in the lower third, above the bottom safe band
    y0 = bottom - int(h * 0.06) - block_h
    cx = w // 2
    # soft pill behind the words for contrast on busy backgrounds
    block_w = max(c.text_wh(ln, size, stroke=2)[0] for ln in lines)
    pad = int(size * 0.4)
    c.rounded_card((cx - block_w // 2 - pad, y0 - pad // 2,
                    cx + block_w // 2 + pad, y0 + block_h + pad // 2),
                   radius=int(size * 0.5), fill=(0, 0, 0, 110))
    y = y0
    for ln in lines:
        c.text((cx, y), ln, size, anchor="ma", stroke=3)
        y += line_h
    c.save(out_png)
    return Overlay(png=out_png, start=start, end=end, fade_in=0.12,
                   fade_out=0.12)


def text_card_png(text: str, w: int, h: int, font: str, out_png: str,
                  accent=(0, 220, 255)) -> str:
    """An opaque full-frame intro/outro card (centred title + accent rule)."""
    from PIL import Image
    c = Canvas(w, h, font)
    # opaque dark background
    c.img = Image.new("RGBA", (w, h), (10, 12, 16, 255))
    from PIL import ImageDraw
    c.draw = ImageDraw.Draw(c.img)
    size = max(48, int(w * 0.085))
    lines = c.wrap(clean_text(text), size, int(w * 0.86)) or [""]
    line_h = c.text_wh("Ag", size)[1] + int(size * 0.3)
    total = line_h * len(lines)
    y = h // 2 - total // 2
    for ln in lines:
        c.text((w // 2, y), ln, size, anchor="ma", stroke=2)
        y += line_h
    # accent rule under the title
    rule_w = int(w * 0.3)
    c.draw.rounded_rectangle(
        (w // 2 - rule_w // 2, y + int(size * 0.1),
         w // 2 + rule_w // 2, y + int(size * 0.1) + max(6, w // 160)),
        radius=4, fill=bgr_to_rgba(accent))
    c.save(out_png)
    return out_png


# --------------------------------------------------------------------------- #
# compositing
# --------------------------------------------------------------------------- #
def composite(video_in: str, overlays: list[Overlay], out_path: str,
              encoder: str, audio_from: str | None = None) -> str:
    """Alpha-composite a list of time-bounded PNG overlays onto ``video_in``.

    Each overlay PNG is looped, given an alpha fade in/out aligned to the main
    timeline, and overlaid only between its start/end. Audio is copied from
    ``video_in`` (or ``audio_from`` if given) untouched.
    """
    overlays = [o for o in (overlays or []) if o and Path(o.png).exists()]
    if not overlays:
        # nothing to draw — passthrough copy so callers get a stable file
        ff.run(["ffmpeg", "-y", "-i", video_in, "-c", "copy", out_path],
               desc="overlay passthrough")
        return out_path

    inputs: list[str] = ["-i", video_in]
    for o in overlays:
        inputs += ["-loop", "1", "-i", o.png]

    filt: list[str] = []
    last = "0:v"
    for i, o in enumerate(overlays, start=1):
        end = min(o.end, 1e8)
        fo_start = max(o.start, end - o.fade_out)
        chain = (f"[{i}:v]format=rgba,"
                 f"fade=t=in:st={o.start:.3f}:d={max(0.01, o.fade_in):.3f}:alpha=1,"
                 f"fade=t=out:st={fo_start:.3f}:d={max(0.01, o.fade_out):.3f}:alpha=1"
                 f"[ov{i}]")
        filt.append(chain)
        nxt = f"v{i}"
        filt.append(
            f"[{last}][ov{i}]overlay=0:0:format=auto:"
            f"enable='between(t,{o.start:.3f},{end:.3f})'[{nxt}]")
        last = nxt

    asrc = audio_from or video_in
    cmd = ["ffmpeg", "-y", *inputs]
    map_audio: list[str] = []
    if audio_from and audio_from != video_in:
        cmd += ["-i", audio_from]
        if ff.has_audio(audio_from):
            map_audio = ["-map", f"{len(overlays) + 1}:a"]
    elif ff.has_audio(video_in):
        map_audio = ["-map", "0:a"]

    cmd += ["-filter_complex", ";".join(filt), "-map", f"[{last}]", *map_audio,
            *ff.venc_args(encoder)]
    if map_audio:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    cmd += ["-shortest", out_path]
    ff.run(cmd, desc="composite overlays")
    return out_path


def cleanup(paths: list[str]) -> None:
    for p in paths:
        try:
            Path(p).unlink()
        except OSError:
            pass
