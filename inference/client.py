"""inference.client – minimal OpenAI-compatible HTTP helper

This client is deliberately thin: it converts Python dicts → HTTP requests and
returns the raw JSON response so higher-level code (AgentLoop) can deal with
streaming, tool-tag parsing, etc.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import requests

_LOCAL_HANDLERS: dict[str, callable] = {}          # url → chat_fn

def register_local(url: str, chat_fn: callable) -> None:
    _LOCAL_HANDLERS[url] = chat_fn
class InferenceClient:
    """Simple wrapper around one vLLM server (OpenAI compatible)."""

    def __init__(self, base_url: str, model: str, *, timeout: int = 60):
        if base_url.endswith("/"):
            base_url = base_url[:-1]
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self._local_fn = _LOCAL_HANDLERS.get(base_url)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def chat(self, messages: List[Dict[str, str]], *, stream: bool = False, **kw) -> Dict[str, Any]:
        if self._local_fn:                          # ← new fast‑path
            return self._local_fn(messages, **kw)
        """POST /v1/chat/completions.

        Parameters
        ----------
        messages : list of {role, content}
        stream   : if True, returns the requests Response iterator; caller must handle SSE.
        kw       : any extra OpenAI parameters (temperature, max_tokens, etc.)
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            **kw,
        }
        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
                stream=stream,
            )
            resp.raise_for_status()
            return resp.json() if not stream else resp
        except requests.RequestException as e:
            raise RuntimeError(f"[InferenceClient] HTTP error: {e}") from e

    def list_models(self) -> List[str]:
        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            return [m["id"] for m in data.get("data", [])]
        except requests.RequestException:
            return [] 