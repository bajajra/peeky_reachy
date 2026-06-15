"""LLM backends for the bedtime-story / lullaby generators.

Three implementations, all conforming to ``LLMBackend.complete(system, user)``:

- ``AnthropicBackend`` — Claude API via the ``anthropic`` SDK. Uses prompt
  caching on the system prompt (per the ``claude-api`` skill guidance) so
  re-generating with the same persona is cheap.
- ``OllamaBackend`` — local LLM via Ollama's HTTP API (default
  ``http://192.168.1.253:11434`` since spark is the active GPU box).
- ``TemplateBackend`` — deterministic, dependency-free, **no network**. The
  one that always works; selected automatically as a fallback whenever the
  preferred backend is missing its dep, key, or host.

Selection priority (``make_backend``):
    explicit ``backend=`` arg  >  ``PEEKY_STORY_BACKEND`` env  >  auto.
Auto: anthropic if ``ANTHROPIC_API_KEY`` is set and the SDK imports;
otherwise ollama if its host responds; otherwise template.

Each backend is expected to *fail soft* — ``complete`` returns ``None`` on
any error, and callers compose with the template fallback.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

log = logging.getLogger("peeky.generate.backend")


class LLMBackend(ABC):
    name: str = "abstract"

    @abstractmethod
    def complete(self, system: str, user: str, *, max_tokens: int = 800) -> Optional[str]:
        """Return generated text, or ``None`` on any failure."""


class TemplateBackend(LLMBackend):
    """Sentinel backend that returns ``None`` so the caller uses its template.

    We model the "no LLM" path as a backend so the selection / wiring code
    stays uniform — there's exactly one decision tree, not two.
    """

    name = "template"

    def complete(self, system: str, user: str, *, max_tokens: int = 800) -> Optional[str]:
        return None


class AnthropicBackend(LLMBackend):
    name = "anthropic"

    def __init__(self, model: str = "claude-haiku-4-5-20251001",
                 api_key: Optional[str] = None):
        self.model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None  # lazy

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        import anthropic  # raises if SDK missing

        self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def complete(self, system: str, user: str, *, max_tokens: int = 800) -> Optional[str]:
        try:
            client = self._ensure_client()
            resp = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                # Cache the (long, stable) system prompt so re-generations are cheap.
                system=[{"type": "text", "text": system,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
            )
            parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
            text = "".join(parts).strip()
            return text or None
        except Exception as exc:
            log.info("Anthropic backend failed: %s", exc)
            return None


class OllamaBackend(LLMBackend):
    name = "ollama"

    def __init__(self, base_url: str = "http://192.168.1.253:11434",
                 model: str = "llama3.2:3b", timeout_s: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def complete(self, system: str, user: str, *, max_tokens: int = 800) -> Optional[str]:
        try:
            import httpx  # raises if dep missing

            with httpx.Client(timeout=self.timeout_s) as c:
                r = c.post(f"{self.base_url}/api/generate", json={
                    "model": self.model,
                    "system": system,
                    "prompt": user,
                    "stream": False,
                    "options": {"num_predict": max_tokens, "temperature": 0.7},
                })
                r.raise_for_status()
                text = (r.json().get("response") or "").strip()
                return text or None
        except Exception as exc:
            log.info("Ollama backend failed: %s", exc)
            return None


def _ollama_reachable(base_url: str, timeout_s: float = 1.5) -> bool:
    try:
        import httpx

        with httpx.Client(timeout=timeout_s) as c:
            return c.get(f"{base_url.rstrip('/')}/api/tags").status_code == 200
    except Exception:
        return False


def make_backend(backend: Optional[str] = None) -> LLMBackend:
    """Pick an LLM backend by name, env, or capability — never raises."""
    choice = (backend or os.environ.get("PEEKY_STORY_BACKEND") or "auto").lower()

    if choice in {"template", "off", "none"}:
        return TemplateBackend()

    if choice == "anthropic":
        return _try_anthropic() or TemplateBackend()

    if choice == "ollama":
        return _try_ollama() or TemplateBackend()

    # auto: prefer Anthropic if a key is set, else Ollama if reachable, else template.
    if os.environ.get("ANTHROPIC_API_KEY"):
        b = _try_anthropic()
        if b is not None:
            return b
    b = _try_ollama()
    if b is not None:
        return b
    log.info("No LLM backend available; using deterministic template fallback.")
    return TemplateBackend()


def _try_anthropic() -> Optional[LLMBackend]:
    try:
        import anthropic  # noqa: F401
    except Exception as exc:
        log.info("Anthropic SDK not installed (%s); skipping.", exc)
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.info("ANTHROPIC_API_KEY not set; skipping Anthropic backend.")
        return None
    model = os.environ.get("PEEKY_LLM_MODEL", "claude-haiku-4-5-20251001")
    return AnthropicBackend(model=model)


def _try_ollama() -> Optional[LLMBackend]:
    base = os.environ.get("PEEKY_OLLAMA_URL", "http://192.168.1.253:11434")
    if not _ollama_reachable(base):
        log.info("Ollama not reachable at %s; skipping.", base)
        return None
    model = os.environ.get("PEEKY_OLLAMA_MODEL", "llama3.2:3b")
    return OllamaBackend(base_url=base, model=model)
