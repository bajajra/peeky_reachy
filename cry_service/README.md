# Peeky cry-classification service (turing)

Runs baby/pet cry classification off the robot/dev box, on **turing
(192.168.1.220)**. The Peeky client (`RemoteEventClassifier`) calls it and falls
back to a local classifier if it's unreachable, so the monitor never stops.

## Endpoints
- `GET  /healthz` → `{"ok": bool, "model_loaded": bool, "model": str}`
- `POST /classify` body `{"audio_wav_b64": "<base64 mono PCM16 wav>"}`
  → `{"event": "baby_cry|dog|speech|silence|other", "score": 0.0-1.0}`

## Run
```bash
# from the repo checked out on turing
pip install -e ".[ml]"                          # classifier
pip install -r cry_service/requirements.txt
pip install "tensorflow>=2.13" "tensorflow-hub>=0.15"   # YAMNet path (optional)
PEEKY_CRY_PREFER_ML=1 PEEKY_EAGER_LOAD=1 \
  uvicorn cry_service.server:app --host 0.0.0.0 --port 8080
# or: docker build -f cry_service/Dockerfile -t peeky-cry . && docker run -p 8080:8080 peeky-cry
```

## Point Peeky at it
```bash
export PEEKY_USE_REMOTE_CRY=true
export PEEKY_CRY_SERVICE_URL=http://192.168.1.220:8080
```

## Smoke test
```bash
curl -s http://192.168.1.220:8080/healthz
python - <<'PY'
import base64, io, wave, numpy as np, httpx
t=np.arange(16000)/16000; sig=(0.5*np.sin(2*np.pi*900*t)).astype(np.float32)
buf=io.BytesIO()
with wave.open(buf,'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes((sig*32767).astype('<i2').tobytes())
print(httpx.post("http://192.168.1.220:8080/classify",
      json={"audio_wav_b64": base64.b64encode(buf.getvalue()).decode()}).json())
PY
```

Without `tensorflow`/`tensorflow-hub` the service runs on the numpy-heuristic
fallback (`model_loaded=false`); install them for the real YAMNet model.
