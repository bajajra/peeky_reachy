# Peeky model-services runbook (T27 — ml-engineer)

Operational guide for the two LAN GPU model services that back Peeky:

| Service                | Host   | IP            | Code           | Port (intended / actual) | Status |
|------------------------|--------|---------------|----------------|--------------------------|--------|
| Cry classification     | turing | 192.168.1.220 | `cry_service/` | 8080 / **8081**          | **LIVE** (uvicorn + user-mode systemd, PID 15935) |
| VoxCPM2 voice clone    | spark  | 192.168.1.253 | `gpu_service/` | 8080 / **8081**          | Deployed (venv), model warmed |
| gemma-4 reason         | turing | 192.168.1.220 | `gemma_service/`| 8082 / —                | **PREPARED** — code + tests in repo; deps installing on turing; awaits user ufw allow + systemd unit (infra) |

The Peeky client (`peeky_reachy.config.Config`) defaults to
`PEEKY_CRY_SERVICE_URL=http://192.168.1.220:8081` (cry moved off :8080 — see
section 1 + `ops/infra.md` "turing port map"),
`PEEKY_VOICE_CLONE_URL=http://192.168.1.253:8081`, and the proposed
`PEEKY_GEMMA_REASON_URL=http://192.168.1.220:8082` (see section 7 — not yet
wired by ai-engineer-3; will land in `peeky_reachy/config.py` with the
gemma-4 client in their next pass).

---

## 0. Known issues / hand-offs to infra-engineer

1. **turing (192.168.1.220) is BACK ONLINE** (RTX 5090, x86_64, CUDA 13.2,
   driver 595.71.05). The cry service is **DEPLOYED and RUNNING** on
   **:8081** (uvicorn + user-mode systemd, PID 15935, see section 1 below).
   host :8080 is anuj's `llama-swap` (different user, do NOT kill) — see
   `ops/infra.md` "turing port map" and the kill-eviction option there.
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
4. **turing ufw blocks new ports by default.** `DEFAULT_INPUT_POLICY="DROP"`.
   The user must `sudo ufw allow 8081/tcp` and `sudo ufw allow 8082/tcp`
   before either service is reachable from the dev Mac. Cry is already
   listening on :8081 but unreachable from outside turing until #4 is run.

---

## 1. Cry-classification service (turing) — DEPLOYED on :8081

### Versions
- App: `cry_service/server.py` (FastAPI 0.1.0), reuses
  `peeky_reachy.detect.classifier` (single source of truth with the local fallback).
- Model: **YAMNet** (`google/yamnet`) via `tensorflow` + `tensorflow-hub`,
  downloaded from TF-Hub on first load (~17 MB SavedModel). CPU-friendly; GPU
  optional. Without TF the service serves the **numpy-heuristic** fallback
  (`model_loaded=false`) — still returns correct `baby_cry` on clear cries.
- Pinned deps: `ops/requirements.cry.lock.txt`.
- venv: `~/workspace/peeky_reachy/.venv-cry/` (Python 3.12, torch 2.12.0+cu130).
  Owned + installed by **infra-engineer** (2026-06-15, see `ops/infra.md`).

### Port decision
- host :8080 on turing is **anuj's `llama-swap`** (different user, do NOT
  kill). Cry is on **:8081** instead — see `ops/infra.md` "turing port map".

### Deploy (already done by infra-engineer; reproduce with)
```bash
ssh bajajra@192.168.1.220
cd ~/workspace/peeky_reachy
# (venv already created at .venv-cry/)
source .venv-cry/bin/activate
pip install -e ".[ml]" -r cry_service/requirements.txt
# optional — adds ~1 GB and ~30 s cold-load time, then `model_loaded=true`:
pip install "tensorflow>=2.13" "tensorflow-hub>=0.15"
PEEKY_CRY_PREFER_ML=1 PEEKY_EAGER_LOAD=1 \
  uvicorn cry_service.server:app --host 0.0.0.0 --port 8081
```
Service is wired to **user-mode systemd** (`~/.config/systemd/user/
peeky-cry.service`, source `ops/peeky-cry-turing.service` in the repo).
The canonical multi-user variant is `ops/peeky-cry.service` — install
via `sudo cp` for production auto-restart (infra's call).

### Env vars
| Var                 | Value | Notes |
|---------------------|-------|-------|
| `PEEKY_CRY_PREFER_ML` | `1`   | prefer YAMNet over heuristic |
| `PEEKY_EAGER_LOAD`    | `1`   | load model at boot (warmup) |
| `PEEKY_LOG_LEVEL`     | `INFO`| |

### Health + smoke (from this Mac, **after** `sudo ufw allow 8081/tcp` on turing)
```bash
curl -s http://192.168.1.220:8081/healthz
# expect {"ok":true,"model_loaded":true,"model":"google/yamnet"}  (false+heuristic if no TF)

# classify a generated cry (expect event=baby_cry):
python - <<'PY'
import base64, httpx
b = open("ops/sample_cry.wav","rb").read()
print(httpx.post("http://192.168.1.220:8081/classify",
      json={"audio_wav_b64": base64.b64encode(b).decode()}, timeout=20).json())
PY
```

**Last verified (2026-06-15 ~10:13 by infra-engineer):**
- `curl http://127.0.0.1:8081/healthz` (loopback) → `{"ok":true,
  "model_loaded":false,"model":"numpy-heuristic"}` ✓
- `POST /classify` with `ops/sample_cry.wav` → `{"event":"baby_cry",
  "score":0.9595...}` ✓
- `curl http://192.168.1.220:8081/healthz` from the dev Mac → **TIMEOUT**
  (ufw blocks; expected — see ufw step in `ops/infra.md`).

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

## 7. gemma-4 reason service (turing) — PREPARED, deploy pending

### Versions / model
- App: `gemma_service/server.py` (FastAPI 0.1.0), `gemma_service/gemmawrap.py`
  lazy loader.
- **Model: `google/gemma-4-E4B-it`** (Apache-2.0, 8B total / 4.5B effective PLE,
  42 layers, 128K context, multimodal text+image+**audio** in, text out, audio
  encoder ~300M params, audio cap **30 s**).
- **Drafter: `google/gemma-4-E4B-it-assistant`** (Apache-2.0, ~78.8M params,
  text-only MTP drafter for speculative decoding, ~3x speedup).
- Engine: **HuggingFace transformers** (not vLLM). The multimodal target +
  MTP-drafter pattern is cleanest in transformers' `generate(assistant_model=
  ...)`. The wrapper interface is engine-agnostic; a vLLM backend can be
  dropped in by replacing `GemmaReasonWrapper` if vLLM catches up.
- Weights disk: target BF16 ~16 GB + drafter ~0.3 GB → **≥20 GB free** in
  `~/.cache/huggingface` (default cache root). Cold load on the RTX 5090:
  ~60-90 s. Plan systemd `TimeoutStartSec=600`.
- VRAM: target ~16 GB reserved at load, drafter ~0.3 GB. RTX 5090 has
  32 GB → ~16 GB headroom for KV cache + a concurrent request. Long
  concurrent clips may OOM — add a request semaphore at the FastAPI tier
  if this becomes a problem.

### Port decision
- host :8080 on turing is anuj's `llama-swap` (different user, do NOT kill).
- host :8081 is the cry service (already running, see section 1).
- **gemma is on :8082** — next free port. Matches the free-ports list in
  `ops/infra.md` "turing port map".
- ufw blocks inbound by default; the user must `sudo ufw allow 8082/tcp`
  (one line) before the gemma service is reachable from the dev Mac.

### Deploy (deps install in progress; service start + unit are infra's job)
```bash
# 1) rsync the repo to turing (run from the dev Mac, infra's convention):
rsync -az --delete \
  --exclude '.venv' --exclude '.venv-gpu' --exclude '.venv-gemma' \
  --exclude '.venv-cry' --exclude '.git' --exclude '.pytest_cache' \
  --exclude '__pycache__' --exclude '*.egg-info' --exclude 'output' \
  --exclude 'enrollment' --exclude '.cache' \
  ./ bajajra@192.168.1.220:~/workspace/peeky_reachy/

# 2) install (ml-engineer is doing this; script /tmp/install_gemma_turing.sh):
ssh bajajra@192.168.1.220
cd ~/workspace/peeky_reachy
export PATH="/home/bajajra/.local/bin:$PATH"      # uv 0.9.2 already on disk
uv venv .venv-gemma --python 3.12
source .venv-gemma/bin/activate
uv pip install -U pip wheel
uv pip install -e ".[ml]"
uv pip install -r gemma_service/requirements.txt
# weights download on first /reason call (or warm with PEEKY_EAGER_LOAD=1)

# 3) foreground sanity:
PEEKY_EAGER_LOAD=1 \
  .venv-gemma/bin/uvicorn gemma_service.server:app --host 0.0.0.0 --port 8082

# 4) install systemd unit (infra's job — file in repo at
#    ops/peeky-gemma.service once it's written, model on
#    ops/peeky-gemma-turing.service for the user-mode variant):
sudo cp ops/peeky-gemma-turing.service /etc/systemd/system/peeky-gemma.service
sudo systemctl daemon-reload && sudo systemctl enable --now peeky-gemma
sudo journalctl -u peeky-gemma -f
```

### Env vars
| Var                         | Value                                | Notes |
|-----------------------------|--------------------------------------|-------|
| `PEEKY_GEMMA_TARGET`        | `google/gemma-4-E4B-it`              | HF model id (target) |
| `PEEKY_GEMMA_DRAFTER`       | `google/gemma-4-E4B-it-assistant`    | HF model id (drafter) |
| `PEEKY_EAGER_LOAD`          | `1`                                  | load model at boot |
| `PEEKY_GEMMA_MAX_NEW_TOKENS`| `256`                                | hard server cap |
| `PEEKY_GEMMA_TIMEOUT_S`     | `30`                                 | per-request timeout |
| `PEEKY_GEMMA_FORCE_FP32`    | unset                                | `=1` to override bf16 on CUDA |
| `PEEKY_LOG_LEVEL`           | `INFO`                               | |

### Health + smoke (from the dev Mac, **after** `sudo ufw allow 8082/tcp` on turing)
```bash
curl -s http://192.168.1.220:8082/healthz
# expect: {"ok":true,"model_loaded":true,
#          "target":"google/gemma-4-E4B-it",
#          "drafter":"google/gemma-4-E4B-it-assistant"}

# reason on the synthetic cry (5 s of pure tone — expect 200 with safe-default shape)
python - <<'PY'
import base64, io, wave, numpy as np, httpx
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

# >30 s audio rejection (expect 400):
python - <<'PY'
import base64, io, wave, numpy as np, httpx
SR=16000
sig=(0.5*np.sin(2*np.pi*450*np.arange(SR*40)/SR)).astype(np.float32)
buf=io.BytesIO()
with wave.open(buf,'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((sig*32767).astype('<i2').tobytes())
r=httpx.post("http://192.168.1.220:8082/reason",
      json={"audio_wav_b64": base64.b64encode(buf.getvalue()).decode(),
            "sample_rate": SR}, timeout=30)
print(r.status_code, r.json())
PY
```

### Point Peeky at it
```bash
export PEEKY_USE_REMOTE_GEMMA=true
export PEEKY_GEMMA_REASON_URL=http://192.168.1.220:8082
```
(The first env var is **proposed** — ai-engineer-3 will add it to
`peeky_reachy/config.py` when they wire the client in their next pass;
the existing `cry_service` flag is `PEEKY_USE_REMOTE_CRY=true`.)

### Threat model
- Service is bound to `0.0.0.0:8082` on turing. **LAN-trust only** — no
  auth, no TLS. Do not expose beyond the LAN. Audio is never logged or
  persisted on the GPU box (decoded from the request, fed to the model,
  discarded). The same threat model applies to the cry / voice services.

### Client contract (frozen 2026-06-15)
Sent to ai-engineer-3 in full — see `standup.md` "ml-engineer" section.
Reproduced here for the runbook:
- `GET  /healthz` → `{ok, model_loaded, target, drafter}`
- `POST /reason` body `{audio_wav_b64, sample_rate?, prompt?}` → 200
  `{event, reason, confidence, transcription, raw_text}` | 400 | 500 | 503.
- **No `/classify` alias** — single endpoint keeps the contract clean.
  The remote `RemoteEventClassifier` already uses `/classify` on the
  cry service (different URL); ai-engineer-3's `GemmaReasonClient`
  uses `/reason` here.

### Rollback
```bash
# on turing:
sudo systemctl disable --now peeky-gemma
rm /etc/systemd/system/peeky-gemma.service      # multi-user install
# or:
systemctl --user disable --now peeky-gemma
rm ~/.config/systemd/user/peeky-gemma.service   # user-mode install
sudo systemctl daemon-reload
# remove the venv + weights cache:
rm -rf ~/workspace/peeky_reachy/.venv-gemma
rm -rf ~/.cache/huggingface                     # drops the ~16 GB target + 0.3 GB drafter
# undo the ufw rule (optional):
sudo ufw delete allow 8082/tcp
```
Peeky client falls back to the local heuristic reason hint (off by
default per `peeky_reachy.detect.reason` — see `ROBUSTNESS.md`).

---

## 8. Files in this runbook set
- `ops/models.md` — this file.
- `ops/requirements.cry.lock.txt` — pinned cry-service deps (turing).
- `ops/requirements.voxcpm.lock.txt` — full `pip freeze` from spark's working voice venv.
- `ops/peeky-voice.service` — systemd unit for spark (port 8081).
- `ops/peeky-cry.service` — systemd unit for turing (multi-user, port 8081).
- `ops/peeky-cry-turing.service` — systemd unit for turing (user-mode, port 8081, currently installed).
- `gemma_service/server.py`, `gemma_service/gemmawrap.py` — gemma-4 reason service (turing, port 8082).
- `gemma_service/requirements.txt`, `gemma_service/Dockerfile` (CUDA 12.8 base).
- `ops/sample_cry.wav` — generated test cry (silence + baby cry) for smoke/e2e.
