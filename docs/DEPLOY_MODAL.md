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
- `GPU` — set to **`"L40S"`** (the configured target). Other options:
  `"T4"` (cheapest) ... `"L4"`, `"A10G"`, `"A100"`, `"H100"`.
- `timeout` on `web` — max seconds a single render may run (set to **2h** so a
  full 1080p match streaming progress over one request isn't killed mid-render).
- `scaledown_window` — how long to keep the GPU warm after the last request
  (trade responsiveness vs cost).
- `@modal.concurrent(max_inputs=...)` — concurrent UI sessions per container
  (heavy render concurrency is separately gated by Gradio's `.queue()`).

## 5. (v3) Enable the AI Director + Critic — serve a vision-LLM with vLLM

The frame-aware Director and the review-loop Critic need a multimodal model.
`modal_app.py` includes an **open-source** vision-LLM served by **vLLM** as an
OpenAI-compatible endpoint (default `Qwen/Qwen2-VL-7B-Instruct`, fits one L40S).
By default the studio runs the **offline heuristic** Director; enabling the VLM
is purely additive.

```bash
# 5.1 one-time: cache the model weights on the fhs-models Volume
modal run modal_app.py::setup_vlm

# 5.2 deploy (this now also serves the /vlm endpoint next to the WebUI)
modal deploy modal_app.py
# note the printed vlm URL, e.g. https://<you>--football-highlight-studio-vlm.modal.run

# 5.3 wire the studio/UI to the endpoint, then redeploy
modal secret create fhs-vlm FHS_VLM_URL=https://<you>--football-highlight-studio-vlm.modal.run
modal deploy modal_app.py
```

How it's consumed:
- **Headless** `studio` auto-picks it up via `FHS_VLM_URL` (`_vlm_overrides()`
  sets `director.backend=openai` + `base_url`), so a run uses the real Director:
  ```bash
  modal run modal_app.py::studio --match-url "https://.../match.mp4"
  # logs show: director=vllm
  ```
- **WebUI**: the `web` function exports `OPENAI_BASE_URL` from `FHS_VLM_URL`. To
  actually switch the UI path on, set in `config/config.yaml`:
  ```yaml
  director:
    backend: openai            # use the served vision-LLM (was: heuristic)
    model: qwen2-vl
  qa:
    use_critic: true           # also run the review-loop Critic on the output
  ```
  (`base_url` is taken from `OPENAI_BASE_URL`/the endpoint; no secret in git.)

Knobs:
- `VLM_MODEL` (top of `modal_app.py`) — swap the brain. A bigger model
  (`Qwen/Qwen2-VL-72B-Instruct-AWQ`) wants most of the L40S to itself; keep the
  7B for co-residence headroom. `FHS_VLM_MODEL` overrides the served name.
- The Director runs **once per clip** (dozens of calls/match), the Critic once
  per render attempt — affordable on a 24/7 L40S; cost is irrelevant per the
  brief but latency adds a little per clip.
- Everything degrades gracefully: if the endpoint is down or unset, the studio
  falls back to the heuristic Director and QA-only review — it never fails a
  render because the brain is unavailable.

### Not executed in this repo's sandbox
The `vlm`/`setup_vlm` functions and the wiring are written against the
documented Modal + vLLM APIs but were **not run here** (no GPU/Modal account in
the build sandbox). Validate with `modal serve modal_app.py` and a `setup_vlm`
run first; pin `vllm`/model revisions once confirmed on the L40S.

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
