# Deploying on Modal (modal.com)

Modal is serverless GPU. The deploy model is different from a plain VM:

| Plain VM (`studio.py`) | Modal (`modal_app.py`) |
|---|---|
| bootstrap venv + deps at runtime | deps baked into a container **Image** |
| local disk for models/output | persistent **Volumes** |
| you manage the process | Modal autoscales; pay per second of GPU |
| `python3 studio.py` | `modal deploy modal_app.py` |

So on Modal use **`modal_app.py`**, not `studio.py`.

## 1. Install + auth
```bash
pip install modal
modal token new          # opens a browser to authenticate
```

## 2. Populate the models Volume (one-time)
```bash
# optional but recommended (you have an HF key): faster, no rate limits
modal secret create huggingface-secret HF_TOKEN=hf_your_token
modal run modal_app.py::setup_models
```
This downloads **every** model the pipeline needs for fully offline operation
and commits them to the `fhs-models` Volume:
- the football detectors (player/goalkeeper/referee/ball, a dedicated ball
  detector, and pitch keypoints) from public Hugging Face repos,
- the **faster-whisper** model (commentary + captions),
- the **EasyOCR** detection/recognition weights (scoreboard),
- a generic **yolov8x.pt** as a last-resort player-model fallback.

It then prints a size manifest so you can confirm the weights actually landed
(expect roughly ~0.8–1 GB; allow a few minutes). All caches are routed onto the
Volume, so a cold GPU container never re-downloads them. The default caption
font (Inter-Bold, OFL-1.1) is bundled in the image, so `drawtext` works without
any extra setup. No Roboflow step needed. Models are also fetched lazily on
first upload if the Volume is empty, so telestration still works out of the box.
The `huggingface-secret` is optional (repos are public) but uses your HF token
when present.

## 3. Run it

Dev (hot reload, temporary `-dev` URL):
```bash
modal serve modal_app.py
```
Production (stable URL, stays deployed):
```bash
modal deploy modal_app.py
```
Modal prints a URL like `https://<you>--football-highlight-studio-web.modal.run`.
Open it — that's the same WebUI: **upload a match, pick options, Render**.

## 4. Configuration knobs (top of `modal_app.py`)
- `GPU` — set to **`"A100-80GB"`** (the v2 target; mature Ampere stack). Other
  options: `"L40S"`, `"A10G"`, `"H100"`.
- `VLM_MODEL` / `VLM_FALLBACK_MODELS` — the Director-VLM and its fallback chain
  (`Qwen2.5-VL-72B-AWQ` -> `32B-AWQ` -> `7B` -> offline heuristic).
- `timeout` on `web` — max seconds a single render may run (set to **2h** so a
  full 1080p match streaming progress over one request isn't killed mid-render).
- `scaledown_window` — how long to keep the GPU warm after the last request
  (trade responsiveness vs cost).
- `@modal.concurrent(max_inputs=...)` — concurrent UI sessions per container
  (heavy render concurrency is separately gated by Gradio's `.queue()`).
- `max_containers=1` on `web` and `vlm` — one warm GPU container each (the 72B
  VLM cold start is large; avoid duplicate loads).

## 5. (v2) Director-VLM (Qwen2.5-VL-72B-AWQ) on A100-80GB + the Critic

The frame-aware Director and the review-loop Critic need a multimodal model.
`modal_app.py` serves an **open-source** vision-LLM via **vLLM** as an
OpenAI-compatible endpoint. The v2 target is **`Qwen/Qwen2.5-VL-72B-Instruct-AWQ`
on a single A100-80GB** (~40–45GB AWQ weights + KV-cache + the vision encoder at
`--gpu-memory-utilization 0.90`). By default the studio runs the **offline
heuristic** Director; enabling the VLM is purely additive.

**Why A100/Ampere over Blackwell (RTX PRO 6000):** the inference stack is
mature — prebuilt `flash-attn`/`flashinfer` wheels exist and vLLM runs on a
stock CUDA 12.x image, with no `nvcc`/`sm_120`/CUDA-12.8 build dance. Because the
`vllm_image` is now a CUDA **devel** base (ships `nvcc`), the old
`VLLM_USE_FLASHINFER_SAMPLER=0` workaround is **no longer needed** (it was only
required on the `nvcc`-less `debian_slim` image). Re-add it as an env var only if
a future vLLM build regresses on Ampere.

```bash
# 5.1 one-time: cache the ~40GB AWQ weights on the fhs-models Volume
modal run modal_app.py::setup_vlm

# 5.2 deploy (this also serves the /vlm endpoint next to the WebUI)
modal deploy modal_app.py
# note the printed vlm URL, e.g. https://<you>--football-highlight-studio-vlm.modal.run

# 5.3 wire the studio/UI to the endpoint, then redeploy
modal secret create fhs-vlm FHS_VLM_URL=https://<you>--football-highlight-studio-vlm.modal.run
modal deploy modal_app.py
```

### The working vLLM command (in `vlm()`)
```bash
vllm serve Qwen/Qwen2.5-VL-72B-Instruct-AWQ \
  --served-model-name qwen2.5-vl \
  --quantization awq \
  --max-model-len 16384 \
  --limit-mm-per-prompt '{"image": 8}' \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000
```
The earlier deploy hit a vLLM **argument error**: `--limit-mm-per-prompt` must be
a **JSON string** (`'{"image": 8}'`) passed as a single argv element (not the old
`image=8` form), and `--quantization awq` must match an AWQ checkpoint. Both are
fixed above.

### Reproducible pins (baked into `vllm_image`)
| package | version |
|---|---|
| base image | `nvidia/cuda:12.4.1-devel-ubuntu22.04` (has nvcc) |
| vllm | 0.7.3 |
| transformers | 4.49.0 |
| accelerate | 1.3.0 |
| autoawq | 0.2.8 |
| qwen-vl-utils | 0.0.10 |

Qwen2.5-VL needs `transformers>=4.49` and a vLLM build with Qwen2.5-VL + AWQ
support; the set above was chosen to be internally coherent.

### Verify (check BOTH a 200 AND a real multi-image request)
```bash
BASE="https://<you>--football-highlight-studio-vlm.modal.run/v1"
curl -s "$BASE/models" -o /dev/null -w "models HTTP %{http_code}\n"   # expect 200
curl -s "$BASE/chat/completions" -H 'Content-Type: application/json' -d '{
  "model":"qwen2.5-vl",
  "messages":[{"role":"user","content":[
    {"type":"text","text":"Is this a goal? Reply JSON."},
    {"type":"image_url","image_url":{"url":"https://picsum.photos/id/1011/640/360"}},
    {"type":"image_url","image_url":{"url":"https://picsum.photos/id/1012/640/360"}}
  ]}]}' | head -c 400
```

### Fallback chain
If the 72B can't be served (quota/OOM), serve a smaller model with **no code
change** by overriding the env and redeploying:
```bash
modal secret create fhs-vlm FHS_VLM_URL=... FHS_VLM_MODEL=Qwen/Qwen2.5-VL-32B-Instruct-AWQ
```
`VLM_FALLBACK_MODELS` in `modal_app.py` documents the order
(`72B-AWQ -> 32B-AWQ -> 7B`). If the endpoint is unreachable the studio
**automatically** drops to the offline heuristic Director — a render never
blocks on the VLM.

How it's consumed:
- **Headless** `studio`/`studio_local` auto-pick it up via `FHS_VLM_URL`
  (`_vlm_overrides()` sets `director.backend=openai` + `base_url`):
  ```bash
  modal run modal_app.py::studio --match-url "https://.../match.mp4"   # logs: director=vllm
  ```
- **WebUI**: `web` exports `OPENAI_BASE_URL` from `FHS_VLM_URL`. To switch the UI
  path on, set in `config/config.yaml`:
  ```yaml
  director:
    backend: openai            # use the served vision-LLM (was: heuristic)
    model: qwen2.5-vl
  qa:
    use_critic: true           # also run the review-loop Critic on the output
  ```

### Not executed in this repo's sandbox
The `vlm`/`setup_vlm` functions and the wiring are written against the
documented Modal + vLLM APIs but were **not run here** (no GPU/Modal account in
the build sandbox). Validate on the A100-80GB with `setup_vlm` + the verify curls
above; the `/v1/models`=200 AND a real multi-image request must both succeed.

## Important Modal-specific caveats

1. **Big match uploads.** Browser → web-endpoint upload works for small/medium
   files, but multi-GB matches are slow and may hit request limits. Two better
   options for a full match:

   **(a) Upload to the server, then pick it in the UI** (recommended):
   ```bash
   modal volume put fhs-input "C:\Users\you\Downloads\Jordan vs Argentina.mp4"
   ```
   Then open the WebUI → **Create** tab → press **↻** next to *"…or pick a match
   already on the server"* → select the file from the dropdown → **Render**.
   The `input/` directory is a persistent `fhs-input` Volume, so the upload
   survives cold starts and is visible to the running app.

   **(b) Headless, no UI** (point at a URL):
   ```bash
   modal run modal_app.py::process --match-url "https://.../match.mp4" --mode compilation
   ```
2. **Output persistence.** Outputs are written to the `fhs-output` Volume. The
   headless `process` function calls `output_vol.commit()`. For the long-running
   WebUI, files are visible within the live container; if you need them durable
   across cold starts, periodically commit the volume or download via the UI.
3. **Cold starts.** First request after scale-to-zero pays image+model load.
   Keeping models on a Volume (step 2) and a non-zero `scaledown_window` reduces
   this.
4. **`vision.device`** stays `cuda` (config default) — correct on a Modal GPU.
   On a CPU-only Modal function set it to `cpu` and disable telestration.
5. **Cost.** You pay per GPU-second. A 90-min match with vision on can take a
   while; estimate before batch runs. The `--no-vision` dry run is cheap.

## Not executed in this repo's sandbox
This file + `modal_app.py` are written against the documented Modal 1.0 API but
were **not run here** (no Modal account/GPU in the build sandbox). Validate with
`modal serve` first; adjust decorator names if your installed Modal SDK differs
(see https://modal.com/docs/guide/modal-1-0-migration).
