"""Lazy wrapper around the gemma-4 reason stack (transformers-based).

Engine choice: **HuggingFace transformers** rather than vLLM.

Why transformers and not vLLM here:

* gemma-4-E4B-it is multimodal (text + image + audio in, text out) with a
  dedicated audio encoder (~300M params). The multimodal ``AutoModelFor*``
  path is best-supported in transformers; vLLM's gemma-4 multimodal path is
  still landing in main and the MTP-drafter pattern (assistant_model) is
  only well-documented against the transformers ``generate`` API.
* The MTP drafter is small (~79M params, text-only). vLLM would also need a
  custom speculative-decoding config; transformers wires it via
  ``model.generate(..., assistant_model=drafter)`` in one call.
* Keeps the service's dep surface small (no vLLM engine, no CUDA graphs at
  the server tier); we can swap to vLLM later by replacing the
  :class:`GemmaReasonWrapper` implementation behind the same
  :meth:`reason`/:meth:`ensure_loaded` interface.

What the wrapper does:

* Lazily loads the target (``AutoModelForMultimodalLM``) and drafter
  (``AutoModelForCausalLM``) on first request (or eagerly on
  :meth:`ensure_loaded` if ``PEEKY_EAGER_LOAD=1``).
* Loads a :class:`transformers.AutoProcessor` for tokenization + audio
  feature extraction.
* Builds the request in the model-template's chat format with an audio
  attachment, calls ``target.generate(..., assistant_model=drafter)``,
  and returns the decoded text.
* Never imports ``transformers`` at module import time — the service
  starts and ``/healthz`` works even with no GPU / no model on disk.

This file is a *pure loader + inference helper*. It does not know about
HTTP, FastAPI, JSON parsing, or the 30 s audio cap — those live in
:mod:`gemma_service.server`.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger("peeky.gemma.wrap")

DEFAULT_TARGET_MODEL = "google/gemma-4-E4B-it"
DEFAULT_DRAFTER_MODEL = "google/gemma-4-E4B-it-assistant"

# gemma-4-E4B-it's audio encoder is capped at 30 s of input. The server
# enforces this *before* calling the wrapper so we never waste GPU on a
# clip we know we can't encode.
MAX_AUDIO_SECONDS = 30.0

# Default prompt — overridable via env or per-request. Asks for a single
# JSON object and nothing else; the server parses it.
DEFAULT_PROMPT = (
    "You are a careful audio analyst for a baby monitor. Listen to the audio "
    "clip and respond with EXACTLY ONE JSON object, no markdown, no prose, "
    "no code fences. Use this schema:\n"
    '{"event": "silence"|"speech"|"baby_cry"|"dog"|"other",\n'
    ' "reason": "hungry"|"tired"|"discomfort"|"pain"|"burping"|"unknown",\n'
    ' "confidence": <float 0-1>,\n'
    ' "transcription": "<short literal transcript or empty string>"\n'
    "}\n"
    "Rules: pick ONE event. Set reason to \"unknown\" unless event is "
    "\"baby_cry\". Confidence is your own certainty in the event, not the "
    "reason. Output ONLY the JSON object."
)


class GemmaUnavailable(RuntimeError):
    """Raised when gemma-4 can't be loaded (no GPU, missing dep, OOM, ...)."""


@dataclass
class ReasonRequest:
    audio_wav_b64: str
    sample_rate: int = 16000
    prompt: Optional[str] = None  # overrides DEFAULT_PROMPT
    max_new_tokens: int = 256
    timeout_s: float = 30.0


@dataclass
class ReasonResult:
    text: str  # the model's raw decoded text (may be malformed JSON)


class GemmaReasonWrapper:
    """Lazy-loaded gemma-4 reason wrapper with MTP speculative decoding.

    The target model is multimodal (audio + text in, text out). The
    drafter is text-only and is passed as ``assistant_model`` to
    ``target.generate(...)`` for ~3x speedup. Both load together in
    :meth:`ensure_loaded`.
    """

    def __init__(
        self,
        target_id: str = DEFAULT_TARGET_MODEL,
        drafter_id: str = DEFAULT_DRAFTER_MODEL,
    ) -> None:
        self.target_id = target_id
        self.drafter_id = drafter_id
        self._target = None
        self._drafter = None
        self._processor = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._load_error: Optional[str] = None
        self._device: Optional[str] = None
        self._dtype: Optional[object] = None

    @property
    def is_loaded(self) -> bool:
        return self._target is not None and self._processor is not None

    @property
    def last_load_error(self) -> Optional[str]:
        return self._load_error

    def ensure_loaded(self) -> None:
        """Eagerly load target + drafter + processor. Safe to call multiple times."""
        if self.is_loaded:
            return
        with self._lock:
            if self.is_loaded:
                return
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoProcessor

                # Pick dtype / device from env, default to bf16 on CUDA else float32.
                use_cuda = torch.cuda.is_available()
                self._device = "cuda" if use_cuda else "cpu"
                self._dtype = (
                    torch.bfloat16
                    if use_cuda and os.environ.get("PEEKY_GEMMA_FORCE_FP32", "0") != "1"
                    else torch.float32
                )
                log.info(
                    "loading gemma-4 target=%s drafter=%s device=%s dtype=%s",
                    self.target_id, self.drafter_id, self._device, self._dtype,
                )

                # Processor is shared: it tokenizes the text prompt and
                # extracts audio features (the multimodal "first-class"
                # inputs are {input_ids, attention_mask, pixel_values?,
                # input_features?}; gemma-4 uses input_features for audio).
                self._processor = AutoProcessor.from_pretrained(self.target_id)

                self._target = AutoModelForMultimodalLM.from_pretrained(
                    self.target_id,
                    torch_dtype=self._dtype,
                ).to(self._device)
                # The drafter is text-only Causal LM; the processor's
                # tokenizer is reused for it.
                self._tokenizer = self._processor.tokenizer
                self._drafter = AutoModelForCausalLM.from_pretrained(
                    self.drafter_id,
                    torch_dtype=self._dtype,
                ).to(self._device)

                # Eval mode for inference.
                self._target.eval()
                self._drafter.eval()
                log.info("gemma-4 loaded (target + drafter)")
            except Exception as exc:  # noqa: BLE001
                self._load_error = repr(exc)
                log.exception("failed to load gemma-4: %s", exc)
                raise GemmaUnavailable(self._load_error) from exc

    def reason(self, req: ReasonRequest) -> ReasonResult:
        """Run one reason pass. Caller is responsible for the 30 s audio cap.

        Returns the raw decoded text. JSON parsing happens in
        :mod:`gemma_service.server`.
        """
        self.ensure_loaded()
        import torch

        from .server import wav_b64_to_samples  # local import to avoid cycle

        samples, sr = wav_b64_to_samples(req.audio_wav_b64)
        if sr != req.sample_rate:
            # Trust the request's stated sample_rate if it differs from
            # the WAV header — caller knows what they resampled to.
            log.debug("WAV header sr=%d differs from request sr=%d; using request",
                      sr, req.sample_rate)

        prompt_text = req.prompt or DEFAULT_PROMPT

        # Build the chat-format message with an audio attachment. We let
        # the processor build {input_ids, attention_mask, input_features}.
        # The drafter gets the *text* part only (assistant_model decodes
        # token-by-token; the audio features are the target's job).
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": samples.astype(np.float32), "sample_rate": req.sample_rate},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(  # type: ignore[attr-defined]
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.inference_mode():
            out_ids = self._target.generate(  # type: ignore[union-attr]
                **inputs,
                assistant_model=self._drafter,
                max_new_tokens=req.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
            )
        # Strip the prompt tokens — generate returns the full sequence.
        input_len = inputs["input_ids"].shape[1]
        new_ids = out_ids[0, input_len:]
        text = self._tokenizer.decode(new_ids, skip_special_tokens=True)  # type: ignore[union-attr]
        return ReasonResult(text=text.strip())


def make_wrapper_from_env() -> GemmaReasonWrapper:
    """Read target/drafter model ids from env (with sane defaults)."""
    return GemmaReasonWrapper(
        target_id=os.environ.get("PEEKY_GEMMA_TARGET", DEFAULT_TARGET_MODEL),
        drafter_id=os.environ.get("PEEKY_GEMMA_DRAFTER", DEFAULT_DRAFTER_MODEL),
    )
