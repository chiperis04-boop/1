"""Deploy Football Highlight Studio on Modal (https://modal.com).

Modal is serverless: dependencies are baked into a container Image, a GPU is
attached per-function, models/outputs live on persistent Volumes, and the Gradio
WebUI is served as an ASGI web endpoint. (So on Modal you do NOT use studio.py's
runtime bootstrap — that's for a plain VM.)

Quick start:
    pip install modal
    modal token new                       # one-time auth
    modal run modal_app.py::setup_models   # one-time: populate the models Volume
    modal serve modal_app.py               # dev: hot-reload + temporary URL
    modal deploy modal_app.py              # production: stable URL

Tested against the Modal 1.0 SDK (May 2025): `@app.function` + `@modal.concurrent`
+ `@modal.asgi_app`, `Image.add_local_dir`, `Volume.from_name`. If your SDK
differs, check https://modal.com/docs.
"""
from __future__ import annotations

import modal

APP_NAME = "football-highlight-studio"
REMOTE = "/root/fhs"                       # project root inside the container
GPU = "L40S"                               # L40S (target) | T4 | L4 | A10G | A100 | H100

app = modal.App(APP_NAME)

# --------------------------------------------------------------------------- #
# Container image: ffmpeg + OpenCV system libs + Python deps + project code.
# Note: PyPI `torch` ships CUDA builds by default, so no special index is needed
# on a GPU container.
# --------------------------------------------------------------------------- #
image = (
    modal.Image.debian_slim(python_version="3.11")
    # libgl*/glib for OpenCV; fonts-dejavu-core is a guaranteed drawtext font
    # fallback in case the bundled Inter TTF is ever absent.
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0", "fonts-dejavu-core")
    .pip_install("torch", "torchvision")
    .pip_install_from_requirements("requirements.txt")
    # Route every lazily-loaded model cache onto the persistent models Volume
    # (mounted at {REMOTE}/models) instead of ephemeral container storage, so a
    # cold container never re-downloads faster-whisper / EasyOCR / ultralytics.
    .env({
        "HF_HOME": f"{REMOTE}/models/cache/hf",
        "HF_HUB_CACHE": f"{REMOTE}/models/cache/hf/hub",
        "FHS_OCR_DIR": f"{REMOTE}/models/easyocr",
        "YOLO_CONFIG_DIR": f"{REMOTE}/models/cache/ultralytics",
    })
    .add_local_dir(
        ".", remote_path=REMOTE,
        # don't ship local junk / things that live on Volumes instead
        ignore=["**/.venv/**", "**/__pycache__/**", "**/*.pyc",
                "bin/**", "input/**", "output/**", "models/**", ".git/**"],
    )
)

# Persistent storage: models downloaded once, outputs kept across runs.
models_vol = modal.Volume.from_name("fhs-models", create_if_missing=True)
output_vol = modal.Volume.from_name("fhs-output", create_if_missing=True)
VOLUMES = {f"{REMOTE}/models": models_vol, f"{REMOTE}/output": output_vol}

# Optional Hugging Face token (for model downloads). Create it with:
#   modal secret create huggingface-secret HF_TOKEN=hf_xxx
# required=False -> deploy/runs still work without it (the model repos are public).
try:
    HF_SECRET = [modal.Secret.from_name("huggingface-secret", required=False)]
except TypeError:                          # older Modal SDK without `required`
    HF_SECRET = []


# --------------------------------------------------------------------------- #
# One-time: populate the models Volume (run: modal run modal_app.py::setup_models)
# Downloads ALL models for 100% offline operation: YOLO player/ball/pitch +
# faster-whisper + EasyOCR + COCO fallback, then prints a size manifest so you
# can confirm the weights actually landed on the Volume.
# --------------------------------------------------------------------------- #
@app.function(image=image, volumes={f"{REMOTE}/models": models_vol},
              secrets=HF_SECRET, timeout=60 * 60)
def setup_models():
    import os
    import subprocess
    os.chdir(REMOTE)
    # stream child output straight to the Modal logs (check=False -> visible
    # failures don't abort the manifest)
    proc = subprocess.run(["python", "scripts/fetch_models.py"], check=False)
    models_vol.commit()                    # persist downloaded weights
    print(f"models volume populated (fetch_models exit={proc.returncode})")
    return proc.returncode


# --------------------------------------------------------------------------- #
# The WebUI, served on a GPU container as an ASGI app.
#   - timeout: max seconds a single render request may run. A full 1080p match
#     can take ~45-75 min on an L40S, and the render streams progress back over
#     one long-lived request, so this must comfortably exceed the worst case
#     (2h) or the UI render is killed mid-way.
#   - scaledown_window: keep the GPU warm this long after the last request
#   - @modal.concurrent: serve several UI sessions from one container (heavy
#     render concurrency is separately gated by Gradio's .queue()).
# --------------------------------------------------------------------------- #
@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=HF_SECRET,
              timeout=2 * 60 * 60, scaledown_window=300)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    import os
    import sys
    sys.path.insert(0, REMOTE)
    os.chdir(REMOTE)                       # so 'input'/'output' resolve to the Volume

    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app
    from app.webui import build

    demo = build().queue()                 # .queue() enables the streaming progress
    fastapi_app = FastAPI()
    return mount_gradio_app(fastapi_app, demo, path="/")


# --------------------------------------------------------------------------- #
# Optional: run a whole match headless on a GPU without the UI, e.g.
#   modal run modal_app.py::process --match-url https://.../match.mp4 --mode compilation
# (Browser upload is fine for small files; for big matches prefer a Volume/URL.)
# --------------------------------------------------------------------------- #
@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=HF_SECRET,
              timeout=2 * 60 * 60)
def process(match_url: str, profile: str = "tiktok", mode: str = "per_clip"):
    import os
    import sys
    import urllib.request
    sys.path.insert(0, REMOTE)
    os.chdir(REMOTE)

    os.makedirs("input", exist_ok=True)
    local = "input/match.mp4"
    urllib.request.urlretrieve(match_url, local)

    from src.runner import run_pipeline
    result = run_pipeline(local, profile=profile,
                          overrides={"render": {"output_mode": mode}})
    output_vol.commit()                    # persist results
    print("status:", result.status, "reel:", result.reel_path,
          "clips:", sum(c.status == "ok" for c in result.clips))
    return result.to_dict()
