"""Thin wrapper around the VoxCPM2 model so the FastAPI layer never imports
``voxcpm`` directly. Makes the model lazy-loadable and the importable at
process start without a GPU present (so ``/healthz`` works during boot)."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("peeky.gpu.voxwrap")

DEFAULT_MODEL = "openbmb/VoxCPM2"


@dataclass
class SynthRequest:
    text: str
    reference_id: Optional[str] = None
    language: str = "en"
    sample_rate: int = 48000  # VoxCPM2 native rate


class VoxCPMUnavailable(RuntimeError):
    """Raised when VoxCPM2 can't be loaded (no GPU, missing dep, OOM, ...)."""


class VoxCPMWrapper:
    """Lazy-loaded VoxCPM2 wrapper. The model itself is loaded on first call to
    :meth:`synthesize` (or eagerly on :meth:`ensure_loaded` if you want it
    warm at boot).

    Reference audio lives in ``references_dir`` as ``<id>.wav`` plus an
    optional ``<id>.txt`` transcript for "ultimate cloning" mode. The
    client only ever sends ``reference_id``; the wrapper resolves the files
    on the GPU box — caregiver audio never leaves the LAN."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        references_dir: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.references_dir = Path(references_dir) if references_dir else None
        self._model = None
        self._lock = threading.Lock()
        self._load_error: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def last_load_error(self) -> Optional[str]:
        return self._load_error

    def ensure_loaded(self) -> None:
        """Eagerly load the model. Safe to call multiple times."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                import voxcpm  # type: ignore

                log.info("loading VoxCPM2 model %s ...", self.model_id)
                self._model = voxcpm.VoxCPM.from_pretrained(self.model_id)
                log.info("VoxCPM2 loaded")
            except Exception as exc:  # noqa: BLE001
                self._load_error = repr(exc)
                log.exception("failed to load VoxCPM2: %s", exc)
                raise VoxCPMUnavailable(self._load_error) from exc

    def list_references(self) -> list[str]:
        if not self.references_dir or not self.references_dir.exists():
            return []
        return sorted(p.stem for p in self.references_dir.glob("*.wav"))

    def _resolve_reference(self, reference_id: str) -> tuple[Optional[Path], Optional[Path]]:
        if not self.references_dir:
            return None, None
        wav = self.references_dir / f"{reference_id}.wav"
        txt = self.references_dir / f"{reference_id}.txt"
        return (wav if wav.exists() else None), (txt if txt.exists() else None)

    def synth(self, req: SynthRequest) -> np.ndarray:
        """Run synthesis. Returns mono float32 at ``req.sample_rate``."""
        self.ensure_loaded()
        ref_wav: Optional[Path] = None
        ref_txt: Optional[Path] = None
        if req.reference_id:
            ref_wav, ref_txt = self._resolve_reference(req.reference_id)
            if ref_wav is None:
                raise FileNotFoundError(
                    f"reference {req.reference_id!r} not found in {self.references_dir}"
                )

        # VoxCPM2's API: VoxCPM has a generation method; we try a few common
        # names so this doesn't break if the upstream package renames things.
        synth = self._model
        samples: Optional[np.ndarray] = None
        for method_name in ("generate", "synthesize", "tts", "infer"):
            method = getattr(synth, method_name, None)
            if method is None:
                continue
            try:
                kwargs: dict = {"text": req.text}
                if ref_wav is not None:
                    kwargs.setdefault("reference_audio", str(ref_wav))
                if ref_txt is not None:
                    kwargs.setdefault("reference_transcript", ref_txt.read_text())
                if req.sample_rate:
                    kwargs.setdefault("sample_rate", req.sample_rate)
                if req.language:
                    kwargs.setdefault("language", req.language)
                out = method(**kwargs)
                samples = _to_mono_f32(out)
                break
            except TypeError:
                # Method exists but signature differs; try the next one.
                continue
        if samples is None:
            raise VoxCPMUnavailable(
                "VoxCPM2 has no generate/synthesize/tts/infer method we can call"
            )
        return samples.astype(np.float32)


def _to_mono_f32(samples) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=-1)
    return arr.reshape(-1)


def make_wrapper_from_env() -> VoxCPMWrapper:
    """Read ``VOXCPM_MODEL`` and ``PEEKY_REFERENCES_DIR`` from the env."""
    return VoxCPMWrapper(
        model_id=os.environ.get("VOXCPM_MODEL", DEFAULT_MODEL),
        references_dir=os.environ.get("PEEKY_REFERENCES_DIR"),
    )
