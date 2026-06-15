"""Runtime configuration with environment-variable overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields


def _env(name: str, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


@dataclass
class Config:
    # Audio capture
    sample_rate: int = 16000
    frame_ms: int = 96
    input_channels: int = 1

    # Detection thresholds
    vad_threshold: float = 0.5
    cry_score_threshold: float = 0.55
    # A cry must persist this long (seconds) before we act, to avoid reacting to
    # a single yelp or a door slam.
    sustain_seconds: float = 3.0
    # After soothing, stay quiet this long before responding again.
    cooldown_seconds: float = 30.0

    # Robustness
    classify_window_seconds: float = 1.0
    smoothing_window: int = 5          # frames voted over to stabilize the class
    min_snr_db: float = 3.0            # ignore faint, far-off sounds below this SNR
    calibration_seconds: float = 2.0   # ambient sampling at startup
    use_ensemble: bool = True          # soft-vote heuristic + ML members
    prefer_ml_vad: bool = True
    prefer_ml_classifier: bool = True

    # Feature flags
    reason_hint_enabled: bool = False

    # Remote model services (LAN GPU boxes)
    #   turing (192.168.1.220) -> baby-cry classification
    #   spark  (192.168.1.253) -> VoxCPM2 voice clone
    cry_service_url: str = "http://192.168.1.220:8080"   # turing
    cry_service_timeout_s: float = 5.0
    use_remote_cry: bool = False                          # set true in production/on-robot
    # Default to the systemd-managed port (per ops/infra.md). :8080 on spark
    # is taken by the pre-existing nginx-llama-proxy (do not collide with it).
    voice_clone_url: str = "http://192.168.1.253:8090"   # spark
    voice_clone_timeout_s: float = 20.0
    enrollment_dir: str = "enrollment"

    # Gemma-4 reason service (ml-engineer T27 pt 2). Frozen 2026-06-15.
    # :8082 on turing — distinct from cry (:8081) and anuj's llama-swap (:8080).
    use_remote_gemma: bool = False
    gemma_reason_url: str = "http://192.168.1.220:8082"
    gemma_timeout_s: float = 10.0

    # Soothing
    assets_dir: str = "assets/soothing"
    language: str = "en"

    @classmethod
    def from_env(cls) -> "Config":
        kwargs = {}
        for f in fields(cls):
            env_name = "PEEKY_" + f.name.upper()
            kwargs[f.name] = _env(env_name, f.default)
        return cls(**kwargs)

    @property
    def frame_size(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000)
