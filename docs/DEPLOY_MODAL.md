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

## Important Modal-specific caveats

1. **Big match uploads.** Browser → web-endpoint upload is fine for small/medium
   files, but multi-GB matches are slow and may hit request limits. For large
   matches prefer the headless path:
   ```bash
   modal run modal_app.py::process --match-url "https://.../match.mp4" --mode compilation
   ```
   or upload the file into the `fhs-output`/a data Volume first and point the
   runner at it.
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
