#!/usr/bin/env python3
"""Football Highlight Studio — single-file self-bootstrapping launcher.

Upload THIS file to your GPU server and run:

    python3 studio.py

On first run it will, using only the Python standard library:
  1. fetch the project source (if `src/` isn't already next to this file),
  2. create a local virtual environment (`.venv`),
  3. install all Python dependencies (PyTorch + the rest),
  4. make sure `ffmpeg`/`ffprobe` are available (downloads a static build if not),
  5. download the detection models,
  6. launch the WebUI at http://<server>:7860

Subsequent runs skip steps that are already done (a marker file is used), so
startup is fast.

Environment overrides (optional):
  FHS_REPO         git/zip source URL (default below) — set to YOUR fork
  FHS_TORCH_INDEX  pip index-url for a CUDA build of torch, e.g.
                   https://download.pytorch.org/whl/cu121
  FHS_PORT         WebUI port (default 7860)
  FHS_HOST         WebUI host (default 0.0.0.0)
  FHS_SHARE        "1" to create a public Gradio share link
  FHS_SKIP_MODELS  "1" to skip model download at startup

CLI passthrough (instead of the WebUI):
  python3 studio.py run input/match.mp4 --profile tiktok
  python3 studio.py detect input/match.mp4
  python3 studio.py list-profiles
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
BIN = ROOT / "bin"
MARKER = VENV / ".bootstrapped"
REPO = os.environ.get(
    "FHS_REPO",
    "https://github.com/your-org/football-highlight-studio",
)
PASSTHROUGH = {"run", "detect", "list-profiles"}
FFMPEG_STATIC = ("https://johnvansickle.com/ffmpeg/releases/"
                 "ffmpeg-release-amd64-static.tar.xz")


# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    print(f"[studio] {msg}", flush=True)


def venv_python() -> Path:
    return VENV / ("Scripts" if os.name == "nt" else "bin") / \
        ("python.exe" if os.name == "nt" else "python")


def in_target_venv() -> bool:
    try:
        return Path(sys.executable).resolve() == venv_python().resolve()
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# 1. source
# --------------------------------------------------------------------------- #
def ensure_sources() -> None:
    if (ROOT / "src").exists() and (ROOT / "requirements.txt").exists():
        return
    log("project source not found next to studio.py — fetching it…")
    tmp = Path(tempfile.mkdtemp())
    fetched = _git_clone(tmp) or _zip_download(tmp)
    if not fetched:
        log("ERROR: could not fetch the project source.")
        log(f"       Set FHS_REPO to a reachable URL (current: {REPO}),")
        log("       or copy the project files next to studio.py manually.")
        sys.exit(1)
    # copy everything (except an existing studio.py / .venv) into ROOT
    for item in fetched.iterdir():
        if item.name in {".git", ".venv", "studio.py"}:
            continue
        dst = ROOT / item.name
        if dst.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)
    shutil.rmtree(tmp, ignore_errors=True)
    log("source ready.")


def _git_clone(tmp: Path):
    if shutil.which("git") is None:
        return None
    target = tmp / "repo"
    try:
        subprocess.run(["git", "clone", "--depth", "1", REPO, str(target)],
                       check=True)
        return target
    except subprocess.CalledProcessError:
        return None


def _zip_download(tmp: Path):
    zip_url = REPO.rstrip("/") + "/archive/refs/heads/main.zip"
    dest = tmp / "src.zip"
    try:
        log(f"downloading {zip_url}")
        urllib.request.urlretrieve(zip_url, dest)
        import zipfile
        with zipfile.ZipFile(dest) as zf:
            zf.extractall(tmp)
        # the archive extracts to <repo>-main/
        for d in tmp.iterdir():
            if d.is_dir() and (d / "src").exists():
                return d
    except Exception as exc:  # noqa: BLE001
        log(f"zip download failed: {exc}")
    return None


# --------------------------------------------------------------------------- #
# 2. venv + 3. deps
# --------------------------------------------------------------------------- #
def ensure_venv() -> None:
    if venv_python().exists():
        return
    log("creating virtual environment (.venv)…")
    import venv as _venv
    _venv.EnvBuilder(with_pip=True).create(VENV)


def ensure_deps() -> None:
    if MARKER.exists():
        return
    py = str(venv_python())
    log("upgrading pip…")
    subprocess.run([py, "-m", "pip", "install", "-q", "--upgrade",
                    "pip", "wheel"], check=True)

    torch_index = os.environ.get("FHS_TORCH_INDEX")
    log("installing torch / torchvision "
        f"({'CUDA: ' + torch_index if torch_index else 'default index'})…")
    cmd = [py, "-m", "pip", "install", "-q", "torch", "torchvision"]
    if torch_index:
        cmd += ["--index-url", torch_index]
    subprocess.run(cmd, check=True)

    log("installing project requirements (this can take a few minutes)…")
    subprocess.run([py, "-m", "pip", "install", "-q", "-r",
                    str(ROOT / "requirements.txt")], check=True)

    MARKER.write_text("ok", encoding="utf-8")
    log("dependencies installed.")


# --------------------------------------------------------------------------- #
# 4. ffmpeg
# --------------------------------------------------------------------------- #
def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    local = BIN / "ffmpeg"
    if local.exists():
        os.environ["PATH"] = f"{BIN}{os.pathsep}{os.environ.get('PATH', '')}"
        return
    log("ffmpeg not found — downloading a static build…")
    BIN.mkdir(exist_ok=True)
    try:
        import lzma
        import tarfile
        tmp = Path(tempfile.mkdtemp())
        xz = tmp / "ff.tar.xz"
        urllib.request.urlretrieve(FFMPEG_STATIC, xz)
        tar = tmp / "ff.tar"
        with lzma.open(xz) as fsrc, open(tar, "wb") as fdst:
            shutil.copyfileobj(fsrc, fdst)
        with tarfile.open(tar) as tf:
            tf.extractall(tmp)
        for d in tmp.glob("ffmpeg-*-static"):
            shutil.copy2(d / "ffmpeg", BIN / "ffmpeg")
            shutil.copy2(d / "ffprobe", BIN / "ffprobe")
        for b in ("ffmpeg", "ffprobe"):
            (BIN / b).chmod(0o755)
        os.environ["PATH"] = f"{BIN}{os.pathsep}{os.environ.get('PATH', '')}"
        shutil.rmtree(tmp, ignore_errors=True)
        log("ffmpeg ready (local static build).")
    except Exception as exc:  # noqa: BLE001
        log(f"WARNING: automatic ffmpeg install failed: {exc}")
        log("         Install ffmpeg manually (apt install ffmpeg) and re-run.")


# --------------------------------------------------------------------------- #
# 5. models
# --------------------------------------------------------------------------- #
def ensure_models() -> None:
    if os.environ.get("FHS_SKIP_MODELS") == "1":
        return
    script = ROOT / "scripts" / "download_models.sh"
    flag = ROOT / "models" / ".downloaded"
    if flag.exists() or not script.exists():
        return
    log("downloading detection models…")
    env = dict(os.environ)
    # run with the venv python on PATH so the script's `python` calls resolve
    env["PATH"] = f"{venv_python().parent}{os.pathsep}{env.get('PATH','')}"
    try:
        subprocess.run(["bash", str(script)], check=True, env=env)
        flag.parent.mkdir(exist_ok=True)
        flag.write_text("ok", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        log(f"WARNING: model download step failed: {exc}")
        log("         You can re-run it later: bash scripts/download_models.sh")


# --------------------------------------------------------------------------- #
# launch
# --------------------------------------------------------------------------- #
def launch_app() -> None:
    ensure_ffmpeg()
    ensure_models()
    args = sys.argv[1:]
    sys.path.insert(0, str(ROOT))

    if args and args[0] in PASSTHROUGH:
        from src.pipeline import app
        app(args)                      # typer app, full CLI
        return

    host = os.environ.get("FHS_HOST", "0.0.0.0")
    port = int(os.environ.get("FHS_PORT", "7860"))
    share = os.environ.get("FHS_SHARE") == "1"
    log(f"starting WebUI on http://{host}:{port}")
    from app.webui import launch
    launch(host=host, port=port, share=share)


def main() -> None:
    ensure_sources()
    if not in_target_venv():
        ensure_venv()
        ensure_deps()
        # hand off to the venv interpreter, keeping any CLI args
        os.execv(str(venv_python()), [str(venv_python()), str(Path(__file__).resolve()), *sys.argv[1:]])
    launch_app()


if __name__ == "__main__":
    main()
