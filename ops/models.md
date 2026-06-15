# Peeky model-services runbook (T27 — ml-engineer)

Operational guide for the two LAN GPU model services that back Peeky:

| Service                | Host   | IP            | Code           | Port (intended / actual) | Status |
|------------------------|--------|---------------|----------------|--------------------------|--------|
| Cry classification     | turing | 192.168.1.220 | `cry_service/` | 8080 / —                 | BLOCKED — host down at deploy time |
| VoxCPM2 voice clone    | spark  | 192.168.1.253 | `gpu_service/` | 8080 / **8081**          | Deployed (venv), model warmed |

The Peeky client (`peeky_reachy.config.Config`) defaults to
`PEEKY_CRY_SERVICE_URL=http://192.168.1.220:8080` and
`PEEKY_VOICE_CLONE_URL=http://192.168.1.253:8080`. See "Port conflict" below —
the voice URL currently needs `:8081`.

---

## 0. Known issues / hand-offs to infra-engineer

1. **turing (192.168.1.220) is DOWN** — `ssh ... : Host is down` and `ping`/22
   unreachable at deploy time. The cry service could NOT be deployed. All cry
   artifacts are prepared + validated locally (clean-venv install + endpoint
   smoke). When infra brings turing up, follow section 1.
2. **Port 8080 on spark is taken** by a pre-existing `nginx-llama-proxy`
   container (returns 502; not ours — leave it alone). The voice service is
   therefore bound to **8081**. To restore the `:8080` default either (a) point
   the client at `:8081` (`PEEKY_VOICE_CLONE_URL=http://192.168.1.253:8081`), or
   (b) have infra free 8080 / add an nginx upstream `8080 -> 127.0.0.1:8081`,
   then change `ops/peeky-voice.service` back to `--port 8080`.
3. **GB10 / aarch64 / CUDA 13.** spark is an NVIDIA GB10 (Grace-Blackwell,
   `aarch64`), driver CUDA 13.0, system nvcc 12.0, Python 3.12. voxcpm 2.0.3
   pulls `torch==2.12.0+cu130` / `torchaudio==2.11.0+cu130` which work and see
   the GPU (`torch.cuda.is_available() == True`, device "NVIDIA GB10"). The
   `gpu_service/Dockerfile` pins `nvidia/cuda:12.4.1` for an x86 box — it was NOT
   used here; we used a venv. If you containerise on spark, rebase the image on a
   CUDA-13 aarch64 base and re-pin.

---

## 1. Cry-classification service (turing) — PREPARED, awaiting host

### Versions
- App: `cry_service/server.py` (FastAPI 0.1.0), reuses
  `peeky_reachy.detect.classifier` (single source of truth with the local fallback).
- Model: **YAMNet** (`google/yamnet`) via `tensorflow` + `tensorflow-hub`,
  downloaded from TF-Hub on first load (~17 MB SavedModel). CPU-friendly; GPU
  optional. Without TF the service serves the **numpy-heuristic** fallback
  (`model_loaded=false`) — still returns correct `baby_cry` on clear cries.
- Pinned deps: `ops/requirements.cry.lock.txt`.

### Deploy (run on turing once SSH is up)
```bash
ssh bajajra@192.168.1.220
cd ~/workspace/peeky_reachy           # rsync the repo here first (see section 3)
python3 -m venv .venv-cry
.venv-cry/bin/pip install -U pip
.venv-cry/bin/pip install -e ".[ml]"
.venv-cry/bin/pip install -r ops/requirements.cry.lock.txt
# real YAMNet (optional but recommended; large):
.venv-cry/bin/pip install "tensorflow>=2.13" "tensorflow-hub>=0.15"
# run (foreground sanity, then use systemd):
PEEKY_CRY_PREFER_ML=1 PEEKY_EAGER_LOAD=1 \
  .venv-cry/bin/uvicorn cry_service.server:app --host 0.0.0.0 --port 8080
```
Then install the unit: `ops/peeky-cry.service` (see header for commands).

### Env vars
| Var                 | Value | Notes |
|---------------------|-------|-------|
| `PEEKY_CRY_PREFER_ML` | `1`   | prefer YAMNet over heuristic |
| `PEEKY_EAGER_LOAD`    | `1`   | load model at boot (warmup) |
| `PEEKY_LOG_LEVEL`     | `INFO`| |

### Health + smoke (from this Mac, once turing is up)
```bash
curl -s http://192.168.1.220:8080/healthz
# expect {"ok":true,"model_loaded":true,"model":"google/yamnet"}  (false+heuristic if no TF)

# classify a generated cry (expect event=baby_cry):
.venv/bin/python - <<'PY'
import base64, httpx
b = open("ops/sample_cry.wav","rb").read()
print(httpx.post("http://192.168.1.220:8080/classify",
      json={"audio_wav_b64": base64.b64encode(b).decode()}, timeout=20).json())
PY
```
**Validated locally** in a clean throwaway venv (artifact check): `/healthz` 200,
`/classify` -> `{"event":"baby_cry","score":~0.78}`, bad base64 -> 400.

---

## 2. VoxCPM2 voice-clone service (spark) — DEPLOYED

### Versions / model
- App: `gpu_service/server.py` (FastAPI 0.1.0), `gpu_service/voxwrap.py` lazy loader.
- Model: **`openbmb/VoxCPM2`** (Apache-2.0, ~2B params). `generate` API present
  (the wrapper probes `generate/synthesize/tts/infer` — first match `generate`).
  Pulls a modelscope denoiser `iic/speech_zipenhancer_ans_multiloss_16k_base`
  (~19 MB) as part of its pipeline.
- Runtime: `torch==2.12.0+cu130`, `torchaudio==2.11.0+cu130`, `voxcpm==2.0.3`,
  `fastapi==0.137.1`, `uvicorn==0.49.0`, `pydantic==2.13.4`. Full freeze:
  `ops/requirements.voxcpm.lock.txt`.

### Sizes / VRAM budget (measured on spark GB10)
- **VRAM on load: ~5.5 GB allocated / 6.3 GB reserved** (within the ~8 GB budget
  in PLAN.md). Add headroom for generation; spark has the GB10's large unified
  pool, so plenty of room. Note an unrelated `zoom_sitter` service (port 8088)
  already uses ~3 GB on the same GPU — leave it alone.
- Model load (cold, incl. CUDA graph warmup): **~136 s**. Plan systemd
  `TimeoutStartSec=600`.
- venv on disk: ~a few GB (torch + CUDA 13 wheels dominate).

### Deploy (already done; reproduce with)
```bash
ssh bajajra@192.168.1.253
cd ~/workspace/peeky_reachy           # rsync'd from the Mac (section 3)
python3 -m venv .venv-gpu
.venv-gpu/bin/pip install -U pip wheel
.venv-gpu/bin/pip install "fastapi==0.137.1" "uvicorn[standard]==0.49.0" \
    "pydantic==2.13.4" "voxcpm==2.0.3"
mkdir -p ~/peeky-enrollment
```
Run (detached — `setsid` so it survives the SSH session; or use systemd):
```bash
cd ~/workspace/peeky_reachy
setsid env PEEKY_REFERENCES_DIR=$HOME/peeky-enrollment PEEKY_EAGER_LOAD=1 \
    VOXCPM_MODEL=openbmb/VoxCPM2 PEEKY_LOG_LEVEL=INFO \
    .venv-gpu/bin/uvicorn gpu_service.server:app --host 0.0.0.0 --port 8081 \
    > /tmp/peeky-voice.log 2>&1 < /dev/null &
```
Preferred: install `ops/peeky-voice.service` (systemd, auto-restart, eager load).

### Env vars
| Var                   | Value                       | Notes |
|-----------------------|-----------------------------|-------|
| `PEEKY_REFERENCES_DIR`| `/home/bajajra/peeky-enrollment` | where `<id>.wav` (+ `<id>.txt`) refs live |
| `PEEKY_EAGER_LOAD`    | `1`                         | warm the model at boot |
| `VOXCPM_MODEL`        | `openbmb/VoxCPM2`           | HF/modelscope model id |
| `PEEKY_LOG_LEVEL`     | `INFO`                      | |

### Health + smoke (from this Mac)
```bash
curl -s http://192.168.1.253:8081/healthz
# expect {"ok":true,"model_loaded":true,"model":"openbmb/VoxCPM2"}

curl -s http://192.168.1.253:8081/references     # {"references":[...]}

# synth a short phrase WITHOUT a reference (zero-shot default voice):
curl -s -X POST http://192.168.1.253:8081/synthesize \
  -H 'content-type: application/json' \
  -d '{"text":"hi baby, I am right here","sample_rate":48000}' \
  --output /tmp/peeky.wav && file /tmp/peeky.wav    # expect: RIFF (little-endian) WAVE

# with an enrolled caregiver reference:
curl -s -X POST http://192.168.1.253:8081/synthesize \
  -H 'content-type: application/json' \
  -d '{"text":"shhh, mama is here","reference_id":"mom","sample_rate":48000}' \
  --output /tmp/peeky_mom.wav
```

### Enroll a caregiver reference (audio stays on spark)
```bash
# from the Mac, the client uploads via the voice flow; or directly:
curl -s -X POST http://192.168.1.253:8081/references \
  -H 'content-type: application/json' \
  -d '{"id":"mom","audio_wav_b64":"<b64 wav>","transcript":"hush now little one"}'
```

---

## 3. Repo sync to a box
```bash
rsync -az --delete \
  --exclude '.venv' --exclude '.venv-gpu' --exclude '.git' --exclude '.pytest_cache' \
  --exclude '__pycache__' --exclude '*.egg-info' --exclude 'output' --exclude 'enrollment' \
  ./ bajajra@<box>:~/workspace/peeky_reachy/
```

---

## 4. End-to-end check (from the Mac)
```bash
PEEKY_USE_REMOTE_CRY=true \
PEEKY_CRY_SERVICE_URL=http://192.168.1.220:8080 \
PEEKY_VOICE_CLONE_URL=http://192.168.1.253:8081 \
PEEKY_ASSETS_DIR=assets/soothing \
  .venv/bin/peeky-demo --wav ops/sample_cry.wav --voice
```
The pipeline calls turing's `/healthz`; if `model_loaded` it uses the remote
classifier, else it logs "remote cry service down; using local" and falls back to
the local heuristic/ensemble — detection never stops. With `--voice` it posts to
spark's `/synthesize` for the soothing phrase (falls back to a pre-recorded track
if spark is unreachable).

`ops/sample_cry.wav` = 2.5 s silence (for ambient calibration) + 10 s synthetic
baby cry; classifies as `baby_cry` (~0.97) and triggers exactly one soothe.

**Last e2e result (turing down):** `Classifier: remote cry service down; using
local` -> `SOOTHE @ score=0.98 event=baby_cry ... played=True` -> 1 soothe event.
Graceful-fallback path confirmed. Re-run after turing is up to confirm the live
remote-classify path.

---

## 5. Restart / rollback / stop

### systemd (preferred, once units installed)
```bash
sudo systemctl restart peeky-voice      # spark
sudo systemctl restart peeky-cry        # turing
sudo systemctl status  peeky-voice
journalctl -u peeky-voice -n 100 --no-pager
sudo systemctl stop peeky-voice && sudo systemctl disable peeky-voice   # rollback
```

### Manual (current spark deploy is a detached uvicorn, not yet systemd)
```bash
ssh bajajra@192.168.1.253 'pgrep -af "uvicorn gpu_service"'
ssh bajajra@192.168.1.253 'pkill -f "uvicorn gpu_service.server"'    # stop
# restart: re-run the setsid command in section 2.
tail -f /tmp/peeky-voice.log    # logs (manual run)
```

### Rollback to no-GPU operation
Leave `PEEKY_USE_REMOTE_CRY=false` (default) and don't pass `--voice`: Peeky runs
fully on the dev box with numpy fallbacks (energy VAD, heuristic classifier,
pre-recorded soothing tracks). Both remote services are optional by design.

---

## 6. Files in this runbook set
- `ops/models.md` — this file.
- `ops/requirements.cry.lock.txt` — pinned cry-service deps (turing).
- `ops/requirements.voxcpm.lock.txt` — full `pip freeze` from spark's working voice venv.
- `ops/peeky-voice.service` — systemd unit for spark (port 8081).
- `ops/peeky-cry.service` — systemd unit for turing (prepared; port 8080).
- `ops/sample_cry.wav` — generated test cry (silence + baby cry) for smoke/e2e.
