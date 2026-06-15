# Peeky gemma-4 reason service (turing)

Wraps `google/gemma-4-E4B-it` (multimodal: text+image+**audio** in, text
out) with the `google/gemma-4-E4B-it-assistant` MTP drafter for
speculative decoding. The pipeline calls it when a baby cry is detected
and needs a *flagged, low-confidence* reason hint.

Engine: **HuggingFace transformers** (not vLLM). See the long comment at
the top of `gemmawrap.py` for the trade-offs.

## Endpoints (contract — frozen 2026-06-15)

```
GET  /healthz
  → 200 {"ok": bool,
         "model_loaded": bool,
         "target": "google/gemma-4-E4B-it",
         "drafter": "google/gemma-4-E4B-it-assistant"}

POST /reason
  body: {"audio_wav_b64": "<base64 mono PCM16 wav, ≤30s>",
         "sample_rate": 16000,            # optional, default 16000
         "prompt": "<optional override>"} # optional; default is JSON-only
  → 200 {"event": "silence"|"speech"|"baby_cry"|"dog"|"other",
         "reason": "hungry"|"tired"|"discomfort"|"pain"|"burping"|"unknown",
         "confidence": 0.0-1.0,
         "transcription": "<short string>",
         "raw_text": "<exact model output>"}
  → 400 {"detail": "..."}   # bad base64, >35 s (hard cap)
  → 500 {"detail": "..."}   # real model error
  → 503 {"detail": "..."}   # model not loaded / OOM
```

Parse miss → `200` with `event="other"`, `reason="unknown"`,
`confidence=0.0`, and the raw text in `raw_text`. **Never a 5xx for
malformed JSON** — that's a model formatting bug, not a service fault.

Server-side caps (overridable via env):
- `PEEKY_GEMMA_MAX_NEW_TOKENS` (default `256`)
- `PEEKY_GEMMA_TIMEOUT_S` (default `30`)

## Run

```bash
# from the repo checked out on turing
python3 -m venv .venv-gemma
source .venv-gemma/bin/activate
pip install -U pip wheel
pip install -e ".[ml]"
pip install -r gemma_service/requirements.txt
PEEKY_EAGER_LOAD=1 \
  uvicorn gemma_service.server:app --host 0.0.0.0 --port 8082
```

Or via Docker (the image is large; first build pulls gemma-4 weights):

```bash
docker build -f gemma_service/Dockerfile -t peeky-gemma .
docker run --gpus all -p 8082:8082 peeky-gemma
```

## Point Peeky at it

```bash
export PEEKY_USE_REMOTE_GEMMA=true
export PEEKY_GEMMA_REASON_URL=http://192.168.1.220:8082
```

## Smoke test

```bash
# health
curl -s http://192.168.1.220:8082/healthz
# expect: {"ok":true,"model_loaded":true,"target":"google/gemma-4-E4B-it","drafter":"google/gemma-4-E4B-it-assistant"}

# reason on the synthetic cry
python - <<'PY'
import base64, io, wave, numpy as np, httpx, os
SR=16000
t=np.arange(SR*5)/SR
sig=(0.5*np.sin(2*np.pi*450*t)).astype(np.float32)
buf=io.BytesIO()
with wave.open(buf,'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((sig*32767).astype('<i2').tobytes())
print(httpx.post("http://192.168.1.220:8082/reason",
      json={"audio_wav_b64": base64.b64encode(buf.getvalue()).decode(),
            "sample_rate": SR}, timeout=30).json())
PY
# expect: {"event":"other", "reason":"unknown", "confidence":0.0, ...}
#         (5 s of pure tone is not a baby cry; we only test that the
#          pipeline returns 200 with the safe-default shape)
```

## Threat model

The service is bound to `0.0.0.0:8082` on turing. **LAN-trust only** —
no auth, no TLS. Do not expose beyond the LAN. Audio is never logged
or persisted on the GPU box (decoded from the request, fed to the
model, discarded).

## Disk + VRAM budget (turing, RTX 5090)

- Weights: target ≈ 16 GB BF16 + drafter ≈ 0.3 GB → **≥20 GB free disk**
  in `~/.cache/huggingface` (or the mounted volume).
- VRAM: target ≈ 16 GB reserved at load, drafter ≈ 0.3 GB. The
  RTX 5090 has 32 GB so we have ~16 GB headroom for KV cache + a
  concurrent request. Two concurrent long clips may OOM — add a
  request semaphore at the FastAPI tier if this becomes a problem.
- Cold load: ~60-90 s on the RTX 5090 (first download + GPU load).
  `PEEKY_EAGER_LOAD=1` warms the model at boot; plan systemd
  `TimeoutStartSec=600`.
