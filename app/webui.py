"""Gradio WebUI for Football Highlight Studio.

A browser front-end over `src.runner`:
  * Create tab  — upload a full match, choose a platform profile, toggle
                  features, then watch live progress and preview the rendered
                  clips with their captions and download links.
  * Library tab — browse, preview, re-download and delete past renders.

Launch:  python -m src.pipeline webui      (or  python studio.py)
"""
from __future__ import annotations

import queue
import shutil
import threading
from pathlib import Path

import gradio as gr

from src.runner import run_pipeline
from src.utils.io import load_config


def _patch_gradio_client_schema_bug() -> None:
    """Defensive guard around gradio_client JSON-schema -> python-type parsing.

    The *root* fix for the historical GET / -> 500 crashes is the coherent
    version lock in requirements.txt (gradio 4.44.1 with era-matched
    fastapi/starlette/pydantic); with those pins get_api_info() succeeds on its
    own. This wrapper is kept only as a harmless belt-and-suspenders net: if a
    future schema edge case feeds a bool / non-dict node into the parser it
    degrades to "Any"/"bool" instead of raising. It is a no-op when the
    internals already behave, so it does NOT mask the version contract."""
    try:
        import gradio_client.utils as u
    except Exception:
        return

    _get_type = getattr(u, "get_type", None)
    if _get_type is not None:
        def get_type(schema):
            if not isinstance(schema, dict):
                return "Any"
            return _get_type(schema)
        u.get_type = get_type

    _js = getattr(u, "_json_schema_to_python_type", None)
    if _js is not None:
        def _json_schema_to_python_type(schema, defs=None):
            if isinstance(schema, bool):
                return "bool"
            try:
                return _js(schema, defs)
            except Exception:
                return "Any"
        u._json_schema_to_python_type = _json_schema_to_python_type


_patch_gradio_client_schema_bug()

INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _profiles() -> list[str]:
    try:
        return list(load_config()["render"]["profiles"].keys())
    except Exception:
        return ["tiktok", "reels", "shorts", "youtube"]


def _clips_for(match_name: str) -> list[str]:
    d = OUTPUT_DIR / match_name
    if not d.exists():
        return []
    return sorted(str(p) for p in d.glob("*.mp4"))


def _caption_for(clip_path: str) -> str:
    txt = Path(clip_path).with_suffix(".txt")
    return txt.read_text(encoding="utf-8") if txt.exists() else ""


def _all_outputs(match_name: str) -> list[str]:
    d = OUTPUT_DIR / match_name
    if not d.exists():
        return []
    return sorted(str(p) for p in d.glob("*.mp4")) + \
           sorted(str(p) for p in d.glob("*.txt"))


def _matches() -> list[str]:
    if not OUTPUT_DIR.exists():
        return []
    return sorted(p.name for p in OUTPUT_DIR.iterdir() if p.is_dir())


# --------------------------------------------------------------------------- #
# create / render
# --------------------------------------------------------------------------- #
def render_job(video_path, profile, output_mode, target_seconds, use_vision,
               slowmo, freeze, telestration, limit, zscore, min_conf, music_vol):
    """Generator: streams a log, then yields final preview state."""
    empty = (gr.update(), gr.update(), gr.update(), gr.update())
    if not video_path:
        yield "⚠️ Upload a match video first.", *empty
        return

    src = Path(video_path)
    dst = INPUT_DIR / src.name
    if src.resolve() != dst.resolve():
        shutil.copy(src, dst)
    match_name = dst.stem

    tgt = float(target_seconds)
    overrides = {
        "vision": {"enabled": bool(use_vision)},
        "telestration": {"enabled": bool(telestration)},
        "edit": {
            "effects": {"slowmo_on_key": bool(slowmo), "freeze_zoom": bool(freeze)},
            "audio": {"music_volume": float(music_vol)},
        },
        "detect": {"audio": {"zscore_threshold": float(zscore)}},
        "fusion": {"min_confidence": float(min_conf)},
        "render": {
            "output_mode": output_mode,
            "compilation": {
                "target_seconds": tgt,
                "min_seconds": max(10.0, tgt - 12),
                "max_seconds": tgt + 12,
            },
        },
    }

    q: queue.Queue = queue.Queue()
    holder: dict = {}

    def cb(p):
        q.put(("p", p))

    def worker():
        try:
            holder["result"] = run_pipeline(
                str(dst), profile=profile, limit=int(limit),
                on_progress=cb, overrides=overrides)
        except Exception as exc:  # noqa: BLE001
            holder["error"] = str(exc)
        finally:
            q.put(("done", None))

    threading.Thread(target=worker, daemon=True).start()

    log: list[str] = [f"▶ Starting: {match_name}  (profile={profile})"]
    yield "\n".join(log), *empty

    while True:
        kind, payload = q.get()
        if kind == "p":
            bar = "█" * int(payload.percent / 5)
            extra = ""
            if payload.clip_index:
                extra = f"  [clip {payload.clip_index}/{payload.clip_total}]"
            log.append(f"{payload.percent:5.1f}% |{bar:<20}| "
                       f"{payload.stage}: {payload.message}{extra}")
            yield "\n".join(log[-30:]), *empty
        else:
            break

    if "error" in holder:
        log.append(f"❌ ERROR: {holder['error']}")
        yield "\n".join(log[-30:]), *empty
        return

    result = holder["result"]
    ok = [c for c in result.clips if c.status == "ok"]
    failed = [c for c in result.clips if c.status == "failed"]
    log.append(f"✅ Done — {len(ok)} clip(s) rendered, {len(failed)} failed, "
               f"{result.goals} goal(s) of {result.moments} moment(s).")
    for c in failed:
        log.append(f"   • clip {c.index} failed at {c.stage_failed}: {c.error}")

    clips = _clips_for(match_name)
    first = clips[0] if clips else None
    yield (
        "\n".join(log[-30:]),
        gr.update(choices=clips, value=first),     # clip dropdown
        first,                                       # video preview
        _caption_for(first) if first else "",        # caption box
        _all_outputs(match_name),                    # downloadable files
    )


def on_pick_clip(clip_path):
    if not clip_path:
        return None, ""
    return clip_path, _caption_for(clip_path)


# --------------------------------------------------------------------------- #
# library
# --------------------------------------------------------------------------- #
def load_match(match_name):
    clips = _clips_for(match_name) if match_name else []
    first = clips[0] if clips else None
    return (gr.update(choices=clips, value=first), first,
            _caption_for(first) if first else "", _all_outputs(match_name or ""))


def delete_match(match_name):
    if match_name:
        shutil.rmtree(OUTPUT_DIR / match_name, ignore_errors=True)
    remaining = _matches()
    return (gr.update(choices=remaining, value=None),
            gr.update(choices=[], value=None), None, "", [])


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def build() -> gr.Blocks:
    with gr.Blocks(title="Football Highlight Studio", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# ⚽ Football Highlight Studio\n"
                    "Upload a full match → auto-detected, telestrated, branded "
                    "vertical highlights.")

        with gr.Tab("Create"):
            with gr.Row():
                with gr.Column(scale=1):
                    video = gr.Video(label="Full match video", sources=["upload"])
                    profile = gr.Dropdown(_profiles(), value="tiktok",
                                          label="Output profile")
                    output_mode = gr.Radio(
                        ["per_clip", "compilation"], value="per_clip",
                        label="Output",
                        info="per_clip = one clip per moment (~15-23s); "
                             "compilation = one reel of several moments")
                    target_seconds = gr.Slider(
                        15, 90, value=45, step=5,
                        label="Reel target length (s) — compilation mode")
                    with gr.Accordion("Features", open=True):
                        use_vision = gr.Checkbox(True, label="GPU telestration + action-aware reframe")
                        telestration = gr.Checkbox(True, label="Draw arrows / spotlight / ball trail")
                        slowmo = gr.Checkbox(True, label="Slow-mo on the key beat")
                        freeze = gr.Checkbox(True, label="Freeze-zoom call-out intro")
                    with gr.Accordion("Detection tuning", open=False):
                        limit = gr.Slider(0, 30, value=0, step=1,
                                          label="Max clips (0 = all)")
                        zscore = gr.Slider(1.0, 4.0, value=2.2, step=0.1,
                                           label="Audio sensitivity (lower = more moments)")
                        min_conf = gr.Slider(0.1, 1.0, value=0.45, step=0.05,
                                             label="Min confidence")
                        music_vol = gr.Slider(0.0, 1.0, value=0.35, step=0.05,
                                              label="Music volume")
                    run_btn = gr.Button("Render highlights", variant="primary")
                with gr.Column(scale=1):
                    logbox = gr.Textbox(label="Progress", lines=16,
                                        max_lines=16, interactive=False)
                    clip_dd = gr.Dropdown([], label="Rendered clips")
                    preview = gr.Video(label="Preview")
                    caption = gr.Textbox(label="Caption / hashtags", lines=4)
                    files = gr.Files(label="Download")

            run_btn.click(
                render_job,
                inputs=[video, profile, output_mode, target_seconds, use_vision,
                        slowmo, freeze, telestration,
                        limit, zscore, min_conf, music_vol],
                outputs=[logbox, clip_dd, preview, caption, files],
            )
            clip_dd.change(on_pick_clip, inputs=clip_dd, outputs=[preview, caption])

        with gr.Tab("Library"):
            with gr.Row():
                match_dd = gr.Dropdown(_matches(), label="Match")
                refresh = gr.Button("↻ Refresh")
                delete = gr.Button("🗑 Delete", variant="stop")
            with gr.Row():
                with gr.Column():
                    lib_clip = gr.Dropdown([], label="Clips")
                    lib_files = gr.Files(label="Files")
                with gr.Column():
                    lib_preview = gr.Video(label="Preview")
                    lib_caption = gr.Textbox(label="Caption", lines=4)

            refresh.click(lambda: gr.update(choices=_matches()), outputs=match_dd)
            match_dd.change(load_match, inputs=match_dd,
                            outputs=[lib_clip, lib_preview, lib_caption, lib_files])
            lib_clip.change(on_pick_clip, inputs=lib_clip,
                            outputs=[lib_preview, lib_caption])
            delete.click(delete_match, inputs=match_dd,
                         outputs=[match_dd, lib_clip, lib_preview, lib_caption, lib_files])

        gr.Markdown("---\nTip: start with **GPU telestration off** for a fast dry "
                    "run, review the detected moments, then re-render with it on.")
    return demo


def launch(host: str = "0.0.0.0", port: int = 7860, share: bool = False):
    demo = build()
    demo.queue()           # enable streaming generators + concurrency
    demo.launch(server_name=host, server_port=port, share=share)


if __name__ == "__main__":
    launch()
