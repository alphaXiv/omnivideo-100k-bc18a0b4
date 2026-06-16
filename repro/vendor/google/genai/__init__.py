"""A drop-in shim for the subset of `google.genai` used by the OmniVideo-100K
data engine, routing every `generate_content` call to OpenRouter's
OpenAI-compatible chat-completions endpoint instead of Google's Gemini API.

Why: the released pipeline talks to Gemini via google-genai (text, inline video
mp4 bytes, and inline audio bytes). The reproduction has an OpenRouter key, not a
Gemini key, and OpenRouter serves the same Gemini models with `video` + `audio`
input modalities. This shim translates the genai call shape into OpenRouter
content parts (`video_url`, `input_audio`, `image_url`, `text`) so the repo's own
scripts and prompts run unmodified.

Activated by putting `repro/vendor` first on PYTHONPATH so `from google import
genai` / `from google.genai import types` resolve here."""

import os
import json
import time
import base64
import urllib.request
import urllib.error

from . import types  # noqa: F401  (re-exported so `from google.genai import types` works)

_BASE = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
_URL = _BASE + "/chat/completions"
_MAX_RETRIES = int(os.environ.get("OR_MAX_RETRIES", "6"))


class _Usage:
    def __init__(self, prompt=0, completion=0, thinking=0):
        self.prompt_token_count = prompt
        self.candidates_token_count = completion
        self.thoughts_token_count = thinking


class _Response:
    def __init__(self, text, usage):
        self.text = text
        self.usage_metadata = usage


def _audio_format(mime_type):
    fmt = (mime_type or "").split("/")[-1].lower()
    # OpenRouter accepts wav, mp3, aiff, aac, ogg, flac, m4a, pcm16, pcm24.
    return {
        "mpeg": "mp3", "mp3": "mp3", "wav": "wav", "x-wav": "wav",
        "aac": "aac", "m4a": "m4a", "mp4": "m4a", "x-m4a": "m4a",
        "opus": "ogg", "ogg": "ogg", "flac": "flac",
    }.get(fmt, fmt)


def _part_to_openrouter(part):
    text = getattr(part, "text", None)
    if text is not None:
        return {"type": "text", "text": text}
    blob = getattr(part, "inline_data", None)
    if blob is not None and getattr(blob, "data", None) is not None:
        mime = blob.mime_type or "application/octet-stream"
        b64 = base64.b64encode(blob.data).decode("ascii")
        if mime.startswith("video/"):
            return {"type": "video_url", "video_url": {"url": "data:%s;base64,%s" % (mime, b64)}}
        if mime.startswith("audio/"):
            return {"type": "input_audio", "input_audio": {"data": b64, "format": _audio_format(mime)}}
        if mime.startswith("image/"):
            return {"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (mime, b64)}}
        return {"type": "file", "file": {"filename": "file", "file_data": "data:%s;base64,%s" % (mime, b64)}}
    return None


def _build_content(contents):
    if isinstance(contents, str):
        return [{"type": "text", "text": contents}]
    parts = getattr(contents, "parts", None)
    if parts is None and isinstance(contents, (list, tuple)):
        parts = contents
    out = []
    for p in parts or []:
        if isinstance(p, str):
            out.append({"type": "text", "text": p})
            continue
        cp = _part_to_openrouter(p)
        if cp is not None:
            out.append(cp)
    return out


class _Models:
    def __init__(self, client):
        self._client = client

    def generate_content(self, model=None, contents=None, config=None, **kwargs):
        content = _build_content(contents)
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": "Bearer %s" % self._client.api_key,
            "Content-Type": "application/json",
            "HTTP-Referer": "https://openresearch.alphaxiv.org",
            "X-Title": "OmniVideo-100K reproduction",
        }
        last_err = None
        for attempt in range(_MAX_RETRIES):
            try:
                req = urllib.request.Request(_URL, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self._client.timeout) as resp:
                    obj = json.loads(resp.read().decode("utf-8"))
                if obj.get("error"):
                    raise RuntimeError("OpenRouter error: %s" % json.dumps(obj["error"])[:400])
                choice = obj["choices"][0]["message"]
                text = choice.get("content") or ""
                if isinstance(text, list):
                    text = "".join(seg.get("text", "") for seg in text if isinstance(seg, dict))
                if not text.strip():
                    raise RuntimeError("empty completion: %s" % json.dumps(obj)[:400])
                u = obj.get("usage", {}) or {}
                details = u.get("completion_tokens_details", {}) or {}
                usage = _Usage(
                    u.get("prompt_tokens", 0),
                    u.get("completion_tokens", 0),
                    details.get("reasoning_tokens", 0),
                )
                return _Response(text, usage)
            except urllib.error.HTTPError as e:
                try:
                    detail = e.read().decode("utf-8")[:400]
                except Exception:
                    detail = ""
                last_err = RuntimeError("HTTP %s: %s" % (e.code, detail))
                if e.code in (408, 409, 425, 429, 500, 502, 503, 504, 529):
                    time.sleep(min(2 ** attempt, 30))
                    continue
                raise last_err
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(min(2 ** attempt, 30))
                continue
        raise last_err if last_err else RuntimeError("generate_content failed")


class Client:
    def __init__(self, api_key=None, http_options=None, vertexai=None, **kwargs):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("API_KEY")
        if not self.api_key:
            raise RuntimeError("No API key: set OPENROUTER_API_KEY / API_KEY")
        self.timeout = float(os.environ.get("OR_HTTP_TIMEOUT", "600"))
        self.models = _Models(self)
