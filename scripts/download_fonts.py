#!/usr/bin/env python3
"""Download the reference-look display fonts (Montserrat / Teko) into
assets/fonts so the Composer/branding can prefer them over Inter/DejaVu.

Graceful by design: every font is attempted from a list of mirrors; the first
that yields a valid TTF wins. Missing fonts are logged, never fatal — the
renderer degrades to Inter, then DejaVu, then a built-in font.

Usage:
    python scripts/download_fonts.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"

TARGETS: dict[str, list[str]] = {
    "Montserrat-Bold.ttf": [
        "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Bold.ttf",
        "https://github.com/google/fonts/raw/main/ofl/montserrat/static/Montserrat-Bold.ttf",
    ],
    "Montserrat-SemiBold.ttf": [
        "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-SemiBold.ttf",
    ],
    "Teko-Bold.ttf": [
        "https://github.com/google/fonts/raw/main/ofl/teko/static/Teko-Bold.ttf",
        "https://github.com/google/fonts/raw/main/ofl/teko/Teko%5Bwght%5D.ttf",
    ],
    "Teko-Medium.ttf": [
        "https://github.com/google/fonts/raw/main/ofl/teko/static/Teko-Medium.ttf",
    ],
}

_TTF_MAGIC = (b"\x00\x01\x00\x00", b"true", b"OTTO", b"ttcf")


def _looks_like_ttf(data: bytes) -> bool:
    return len(data) > 4096 and data[:4] in _TTF_MAGIC


def fetch(url: str, timeout: int = 30) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "fhs-fonts/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {url} -> {exc}")
        return None


def main() -> int:
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    got = 0
    for name, urls in TARGETS.items():
        dest = FONT_DIR / name
        if dest.exists() and dest.stat().st_size > 4096:
            print(f"= {name} already present")
            got += 1
            continue
        for url in urls:
            data = fetch(url)
            if data and _looks_like_ttf(data):
                dest.write_bytes(data)
                print(f"+ {name} <- {url} ({len(data)//1024} KB)")
                got += 1
                break
        else:
            print(f"- {name}: no mirror worked (renderer will fall back)")
    print(f"done: {got}/{len(TARGETS)} fonts available in {FONT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
