# Peeky GPU service (VoxCPM2)

FastAPI wrapper around the [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)
zero-shot voice-clone model. Deployed on the local LAN GPU boxes (turing /
spark). The core `peeky_reachy` package never imports this — it talks to it
over HTTP via `httpx`.

## What it runs

- `POST /synthesize` — body `{text, reference_id?, language?, sample_rate?}`
  → `audio/wav` (PCM16 mono, requested rate).
- `GET  /healthz` — `{ok, model_loaded, model}`.
- `GET  /references` — `["id1", "id2", ...]`.
- `POST /references` — `{id, audio_wav_b64, transcript?, overwrite?}` registers
  a caregiver voice. The reference stays on the GPU box; the client only ever
  sends `reference_id`.

## Deploy

### Quick (local dev box)

```bash
cd /path/to/peeky_reachy
python -m venv .venv-gpu && source .venv-gpu/bin/activate
pip install -r gpu_service/requirements.txt
PEEKY_REFERENCES_DIR=$HOME/peeky-enrollment \
PEEKY_EAGER_LOAD=1 \
uvicorn gpu_service.server:app --host 0.0.0.0 --port 8080
```

### Docker (the deployment shape on turing/spark)

```bash
docker build -f gpu_service/Dockerfile -t peeky-voxcpm2 .
docker run --gpus all --rm -p 8080:8080 \
    -v /srv/peeky/enrollment:/srv/peeky/enrollment:ro \
    -e PEEKY_REFERENCES_DIR=/srv/peeky/enrollment \
    -e PEEKY_EAGER_LOAD=1 \
    peeky-voxcpm2
```

## Config (env)

| Var                      | Default                | Notes                                          |
|--------------------------|------------------------|------------------------------------------------|
| `VOXCPM_MODEL`           | `openbmb/VoxCPM2`      | HF model id                                    |
| `PEEKY_REFERENCES_DIR`   | _unset_                | dir of `<id>.wav` (+ optional `<id>.txt`) refs |
| `PEEKY_EAGER_LOAD`       | `0`                    | set to `1` to load the model on boot           |
| `PEEKY_LOG_LEVEL`        | `INFO`                 |                                                |

## Smoke test

```bash
curl -s http://localhost:8080/healthz
# {"ok":true,"model_loaded":true,"model":"openbmb/VoxCPM2"}

curl -s -X POST http://localhost:8080/synthesize \
    -H 'content-type: application/json' \
    -d '{"text":"hi baby, I am right here","reference_id":"dad"}' \
    --output /tmp/peeky.wav
```

## What it does NOT do

- No authentication — assumes LAN trust. If you ever expose it, put it behind
  a reverse proxy with mTLS or basic auth.
- No request authn on `/synthesize` either. A bad actor on the LAN could
  synthesize in anyone's voice. That's the threat model the project is OK
  with for an MVP; revisit before going past LAN.
- No audio effects. VoxCPM2 outputs the raw synthesis.
