"""Vision-LLM client — one thin, provider-agnostic way for the agents to ask a
multimodal model for a JSON answer.

Backends (all open-source-friendly, configured via cfg['director']):
  * "openai" / "vllm" / "ollama" / "minicpm" — any OpenAI-compatible chat/vision
    endpoint. Point `director.base_url` at a local vLLM serving Qwen2-VL /
    MiniCPM-V. This is the recommended local path (cost-free, offline-capable).
  * "gemini" — google-generativeai multimodal.

Design notes:
  * `chat_json()` returns a parsed dict, retrying on transient failures and
    tolerating fenced / chatty replies. The low-level call lives in `_complete`
    so tests can monkeypatch it without any network.
  * `is_configured()` lets callers cheaply decide whether to even try the model
    before falling back to the offline heuristic.
"""
from __future__ import annotations

import base64
import json
import os
import time

from ..utils.io import get_logger

log = get_logger()

_OPENAI_BACKENDS = {"openai", "vllm", "ollama", "minicpm", "lmstudio", "local"}


class VisionLLMClient:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        d = (cfg or {}).get("director", {})
        llm = (cfg or {}).get("llm", {})
        # The `llm:` config section (NVIDIA NIM / any OpenAI-compatible endpoint)
        # takes precedence; director.* stays a fallback for older configs.
        self.base_url = (llm.get("base_url") or d.get("base_url")
                         or os.environ.get("OPENAI_BASE_URL"))
        # api key: env FIRST (never hard-depend on a key committed to the repo)
        self.api_key = (os.environ.get("NVIDIA_API_KEY")
                        or (llm.get("api_key") or "").strip()
                        or os.environ.get("OPENAI_API_KEY"))
        self.model = (llm.get("model_name") or d.get("model", "") or "")
        # a configured llm endpoint forces the OpenAI-compatible transport
        self.backend = ("openai" if llm.get("base_url")
                        else (d.get("backend") or "heuristic").lower())
        self.temperature = float(d.get("temperature", 0.4))
        self.max_retries = int(d.get("max_retries", 2))
        self.timeout = float(d.get("timeout", 60.0))
        self.system_prompt = d.get("system_prompt", "")
        # hosted NIM VLMs accept only ONE image/request -> tile the sampled
        # keyframes into a single contact-sheet so the model still sees the whole
        # sequence (motion / ball path / footwork). Disable for a true multi-image
        # endpoint (local vLLM) via llm.tile_frames=false.
        self.tile_frames = bool(llm.get("tile_frames", True))
        self.max_sampled_frames = int(llm.get("max_sampled_frames", 24))

    # ------------------------------------------------------------------ public
    def is_configured(self) -> bool:
        """True if a real model backend is reachable-by-config (not heuristic)."""
        if self.backend == "gemini":
            return bool(os.environ.get("GEMINI_API_KEY")
                        or os.environ.get("GOOGLE_API_KEY"))
        if self.backend in _OPENAI_BACKENDS:
            # a local vLLM/Ollama needs only a base_url; hosted OpenAI needs a key
            return bool(self.base_url or os.environ.get("OPENAI_API_KEY"))
        return False

    def chat_json(self, system: str, text: str,
                  images: list[bytes] | None = None) -> dict:
        """Ask the model and return a parsed JSON dict. Raises on hard failure."""
        images = images or []
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = self._complete(system, text, images)
                return parse_json(raw)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                log.warning(f"[llm] attempt {attempt + 1} failed: {exc}")
                time.sleep(min(2.0 * attempt, 4.0))
        raise RuntimeError(f"vision-LLM call failed: {last_exc}")

    # --------------------------------------------------------------- transport
    def _complete(self, system: str, text: str, images: list[bytes]) -> str:
        if self.backend == "gemini":
            return self._complete_gemini(system, text, images)
        if self.backend in _OPENAI_BACKENDS:
            return self._complete_openai(system, text, images)
        raise NotImplementedError(f"llm backend '{self.backend}'")

    def _complete_openai(self, system: str, text: str, images: list[bytes]) -> str:
        from openai import OpenAI
        if self.tile_frames and len(images) > 1:
            from ..utils.io import montage_jpeg
            images = montage_jpeg(images, max_frames=self.max_sampled_frames)
        client = OpenAI(base_url=self.base_url,
                        api_key=self.api_key or "not-needed-local",
                        timeout=self.timeout)
        content: list[dict] = [{"type": "text", "text": text}]
        for fb in images:
            b64 = base64.b64encode(fb).decode("ascii")
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        resp = client.chat.completions.create(
            model=self.model or "qwen2-vl",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": content}],
            response_format={"type": "json_object"},
            temperature=self.temperature,
        )
        return resp.choices[0].message.content

    def _complete_gemini(self, system: str, text: str, images: list[bytes]) -> str:
        import google.generativeai as genai
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY/GOOGLE_API_KEY not set")
        genai.configure(api_key=key)
        model = genai.GenerativeModel(
            self.model or "gemini-1.5-flash",
            system_instruction=system,
            generation_config={"response_mime_type": "application/json",
                               "temperature": self.temperature})
        parts: list = [text]
        for fb in images:
            parts.append({"mime_type": "image/jpeg", "data": fb})
        return model.generate_content(parts).text


# --------------------------------------------------------------------------- #
def parse_json(text: str) -> dict:
    """Tolerant JSON extraction: strips markdown fences / prose around the obj."""
    if not text:
        raise ValueError("empty model response")
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("model did not return a JSON object")
    return obj
