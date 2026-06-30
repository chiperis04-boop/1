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

# Persistent storage: models downloaded once, inputs uploaded once, outputs kept.
models_vol = modal.Volume.from_name("fhs-models", create_if_missing=True)
input_vol = modal.Volume.from_name("fhs-input", create_if_missing=True)
output_vol = modal.Volume.from_name("fhs-output", create_if_missing=True)
VOLUMES = {
    f"{REMOTE}/models": models_vol,
    f"{REMOTE}/input": input_vol,     # `modal volume put fhs-input <file>` -> pick in the UI
    f"{REMOTE}/output": output_vol,
}

# Optional Hugging Face token (for model downloads). Create it with:
#   modal secret create huggingface-secret HF_TOKEN=hf_xxx
# required=False -> deploy/runs still work without it (the model repos are public).
try:
    HF_SECRET = [modal.Secret.from_name("huggingface-secret", required=False)]
except TypeError:                          # older Modal SDK without `required`
    HF_SECRET = []


# --------------------------------------------------------------------------- #
# Vision-LLM (the Director / Critic "brain"), served as an OpenAI-compatible
# endpoint by vLLM on its own GPU container. Open-source, runs 24/7 on an L40S.
#   modal run   modal_app.py::setup_vlm   # one-time: cache the weights on Volume
#   modal deploy modal_app.py             # serves /vlm alongside the WebUI
# Then point the studio at it (see _vlm_overrides + DEPLOY_MODAL.md):
#   modal secret create fhs-vlm FHS_VLM_URL=https://<you>--...-vlm.modal.run
# Default model fits one L40S (48 GB) comfortably; swap via the FHS_VLM_MODEL env.
# --------------------------------------------------------------------------- #
VLM_MODEL = "Qwen/Qwen2-VL-7B-Instruct"    # open-source multimodal Director brain
VLM_SERVED_NAME = "qwen2-vl"

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm>=0.6.3", "qwen-vl-utils", "huggingface_hub>=0.24,<1.0")
    .env({"HF_HOME": f"{REMOTE}/models/cache/hf",
          "HF_HUB_CACHE": f"{REMOTE}/models/cache/hf/hub",
          "VLLM_DO_NOT_TRACK": "1"})
)


@app.function(image=vllm_image, volumes={f"{REMOTE}/models": models_vol},
              secrets=HF_SECRET, timeout=60 * 60)
def setup_vlm():
    """One-time: download the vision-LLM weights onto the models Volume so the
    always-on server has zero cold-download."""
    from huggingface_hub import snapshot_download
    path = snapshot_download(VLM_MODEL)
    models_vol.commit()
    print(f"vLLM model cached: {VLM_MODEL} -> {path}")
    return path


@app.function(image=vllm_image, gpu=GPU, volumes={f"{REMOTE}/models": models_vol},
              secrets=HF_SECRET, timeout=24 * 60 * 60, scaledown_window=20 * 60)
@modal.concurrent(max_inputs=16)           # vLLM batches requests internally
@modal.web_server(8000, startup_timeout=15 * 60)
def vlm():
    """OpenAI-compatible vision endpoint (vLLM serving Qwen2-VL).

    Exposes /v1/chat/completions with image_url content — exactly what
    src/agents/llm_client.VisionLLMClient (backend='openai') speaks."""
    import shlex
    import subprocess
    cmd = (
        "python -m vllm.entrypoints.openai.api_server "
        f"--model {VLM_MODEL} --served-model-name {VLM_SERVED_NAME} "
        "--host 0.0.0.0 --port 8000 "
        "--max-model-len 8192 --limit-mm-per-prompt image=16 "
        "--gpu-memory-utilization 0.92 --trust-remote-code"
    )
    subprocess.Popen(shlex.split(cmd))


def _vlm_overrides() -> dict:
    """Director/Critic config overrides pointing at the deployed vLLM endpoint,
    enabled only when FHS_VLM_URL is set (else {} -> offline heuristic)."""
    import os
    url = os.environ.get("FHS_VLM_URL")
    if not url:
        return {}
    base = url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return {"director": {"backend": "openai", "base_url": base,
                         "model": os.environ.get("FHS_VLM_MODEL", VLM_SERVED_NAME)}}


# Secret carrying FHS_VLM_URL (the deployed vLLM endpoint) for worker functions.
# Create after deploying once you know the /vlm URL:
#   modal secret create fhs-vlm FHS_VLM_URL=https://<you>--...-vlm.modal.run
try:
    VLM_SECRET = HF_SECRET + [modal.Secret.from_name("fhs-vlm", required=False)]
except TypeError:
    VLM_SECRET = HF_SECRET


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
@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=VLM_SECRET,
              timeout=2 * 60 * 60, scaledown_window=300)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    import os
    import sys
    sys.path.insert(0, REMOTE)
    os.chdir(REMOTE)                       # so 'input'/'output' resolve to the Volume

    # If a vLLM endpoint is wired (FHS_VLM_URL), expose it as OPENAI_BASE_URL so
    # a config with director.backend=openai / qa.use_critic uses the Director +
    # Critic. Without it the studio stays on the offline heuristic.
    vurl = os.environ.get("FHS_VLM_URL")
    if vurl:
        base = vurl.rstrip("/")
        os.environ["OPENAI_BASE_URL"] = base if base.endswith("/v1") else base + "/v1"

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


# --------------------------------------------------------------------------- #
# v2 studio chain headless (Scout -> Director -> Cameraman+CMC -> Composer):
#   modal run modal_app.py::studio --match-url https://.../match.mp4
# Set director.backend=gemini via the huggingface/secret env if you want the LLM
# manifest; otherwise it uses the offline heuristic director.
# --------------------------------------------------------------------------- #
@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=VLM_SECRET,
              timeout=2 * 60 * 60)
def studio(match_url: str, profile: str = "tiktok"):
    import os
    import sys
    import urllib.request
    sys.path.insert(0, REMOTE)
    os.chdir(REMOTE)

    os.makedirs("input", exist_ok=True)
    local = "input/match.mp4"
    urllib.request.urlretrieve(match_url, local)

    from src.studio_pipeline import run_studio
    overrides = _vlm_overrides() or None   # use the Director VLM if wired, else heuristic
    result = run_studio(local, profile=profile, overrides=overrides)
    output_vol.commit()
    ok = sum(c.status == "ok" for c in result.clips)
    print(f"studio status={result.status} events={result.windows} "
          f"goals={result.goals} finished={ok}/{len(result.clips)} "
          f"director={'vllm' if overrides else 'heuristic'}")
    return result.to_dict()
