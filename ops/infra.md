# Peeky — Infra Runbook (turing + spark)

Owner: **infra-engineer**. Covers the two LAN GPU boxes that host Peeky's model
services, how to start/stop/status/log them, ports/firewall, and the manual steps
still required from the user. Companion to `cry_service/README.md` and
`gpu_service/README.md`.

Service allocation (per `PLAN.md` / `status.md`):

| Service | Code | Intended host | Notes |
|---|---|---|---|
| Cry classification | `cry_service/` (FastAPI) | **turing — 192.168.1.220** | YAMNet (TF + tf-hub), CPU-friendly; numpy fallback if TF absent |
| VoxCPM2 voice clone | `gpu_service/` (FastAPI) | **spark — 192.168.1.253** | needs GPU; ~8 GB VRAM min |

> Both service Dockerfiles `EXPOSE 8080` internally. On spark we publish them on
> **alternate host ports** because host :8080 is already taken (see Port map).

---

## Host audit (last re-verified 2026-06-15 by infra-engineer re-spawn)

| Item | turing (192.168.1.220) | spark (192.168.1.253) |
|---|---|---|
| SSH (key-based) | **WORKING** (`team_infra_ed25519`, no password) | **WORKING** (`team_infra_ed25519`, no password) |
| Reachability | ping UP, tcp/22 open | ping UP, tcp/22 open |
| Hostname | `bajajra-turing` | `spark-f82c` |
| OS / arch | **Ubuntu 26.04 LTS ("Resolute Raccoon"), x86_64** | Ubuntu 24.04.4 LTS, **aarch64** |
| Kernel | 7.0.0-22-generic | 6.17.0-1021-nvidia |
| CPU / RAM | **AMD Ryzen 9 9950X3D, 16C/32T, 121 GiB** (118 GiB avail) | 20 cores / 121 GiB (112 GiB avail) |
| GPU | **NVIDIA GeForce RTX 5090, 32 GB VRAM, driver 595.71.05, CUDA 13.2** | **NVIDIA GB10** (Grace-Blackwell / DGX Spark) |
| VRAM | 32 GB discrete | unified memory (GB10) — backed by the 121 GiB system RAM |
| Python | **3.14.4** (system); ensurepip **NOT** bundled (need `apt install python3.14-venv`) | 3.12.3 + pip 24.0 |
| Docker | 29.1.3, nvidia runtime unknown (user not in docker group; `permission denied`); dockerd running as root | 29.2.1, overlay2, **nvidia runtime present** (`nvidia-ctk` 1.19.1, CDI `nvidia.com/gpu=all`) — but ml-engineer did NOT use Docker for the voice service (uses venv+uvicorn directly) |
| Free disk | 1.3 TB free on `/` (2.1 TB, 35% used) | 2.2 TB free on `/` (3.7 TB, 40% used) |
| systemd --user | working; **linger DISABLED** (services die on full logout) | working; **linger ENABLED** (services survive logout / start on boot) |
| ufw | **ENABLED, `DEFAULT_INPUT_POLICY="DROP"`** (allow-list; see port map) | not active (LAN-firewall-open per prior audit) |
| Passwordless sudo | NO (sudo requires interactive password) | NO |
| Pre-existing processes | `llama-swap` (anuj, PID 7105, --listen :8080) | `infra-caddy-1` (:80), `nginx-llama-proxy` (:8080, healthy) — **DO NOT TOUCH** |
| Internet (PyPI) | YES (re-verified 2026-06-15 ~10:09) | YES |
| Cry service | **RUNNING** on `:8081` (user-mode systemd, numpy-heuristic, PID 15935) — see "Cry service on turing" below | not started (contingency only) |
| Voice service | not deployed on turing | **RUNNING** on `:8081` (ml-engineer's detached uvicorn, model_loaded=true) |

### turing port map (re-verified 2026-06-15)

`ss -ltn` on turing shows these bound:

| Port | Owner | Action |
|---|---|---|
| 22 | sshd | keep |
| **8080** | `llama-swap` (anuj's pre-existing, PID 7105, `--listen :8080 -watch-config`, config at `/home/anuj/owlagents/infra/bajara/llama-swap.yaml`) | **do not touch** — pre-existing, owned by another user. See "Kill llama-swap + redeploy on :8080" below if you want to evict it. |
| **8081** | **ml-engineer's/our cry service** (uvicorn, PID 15935, since 2026-06-15 10:13) | **Peeky cry — do not touch** (peer's process) |
| 53, 631, 111, 2049, 34213, 35243, 47705, 49187, 49765, 57053 | local services / NFS / RPC | keep |

Free ports still available: 8090, 8091, 9090. (8081 is now the cry service.)
`:8080` is reachable from the dev Mac (anuj has ufw allowed it for the LAN);
`:8081` is **not yet reachable from the dev Mac** — ufw blocks it (see
"Manual steps still required" #0).

### spark port map (re-verified 2026-06-15)

`ss -ltn` on spark shows these bound:

| Port | Owner | Action |
|---|---|---|
| 22 | sshd | keep |
| 80 | `infra-caddy-1` container (pre-existing) | **do not touch** |
| 8000 | pre-existing | **do not touch** |
| **8080** | `nginx-llama-proxy` container (LLM proxy, up >4 days) | **do not touch** — this is why Peeky services use alternate ports on spark |
| **8081** | **ml-engineer's voice service** (uvicorn, PID 3463795, since 2026-06-15 08:23) | **Peeky voice — do not touch** (peer's process) |
| 8088 | pre-existing (`zoom_sitter`, ~3 GB VRAM) | **do not touch** |
| 11000, 631, 53 | local services | keep |

Free ports still available: 8090, 8091, 9090. (8081 is now used by the live voice
service; the cry fallback on spark was never started, so 8091 is free.)

### Current spark deploy layout (note: two trees)

ml-engineer rsynced the repo to `~/workspace/peeky_reachy/` (their convention,
with a `.venv-gpu` venv) and the prior infra-engineer's rsync lives at
`~/peeky/` (with my user-mode systemd units + env files). **The currently running
voice service is in ml-engineer's tree, not mine.** The two trees are independent
and the running service does not use my env files or my user-mode units.

```bash
# ml-engineer's tree (live voice service)
~/workspace/peeky_reachy/
  .venv-gpu/                                # venv with voxcpm==2.0.3, torch==2.12.0+cu130
  gpu_service/                             # service code
  pyproject.toml, etc.
# detached uvicorn running here (per ml-engineer's setsid invocation):
#   PEEKY_REFERENCES_DIR=$HOME/peeky-enrollment PEEKY_EAGER_LOAD=1
#   VOXCPM_MODEL=openbmb/VoxCPM2 PEEKY_LOG_LEVEL=INFO
#   .venv-gpu/bin/uvicorn gpu_service.server:app --host 0.0.0.0 --port 8081
#   > /tmp/peeky-voice.log 2>&1

# prior infra-engineer's tree (orphaned, kept for rollback)
~/peeky/
  voice.env                                # PEEKY_VOICE_PORT=8090 (NOT used by live service)
  cry.env                                  # PEEKY_CRY_PORT=8091 (NOT used; cry's real home is turing)
  gpu_service/, cry_service/, peeky_reachy/, assets/, enrollment/
  ~/.config/systemd/user/peeky-voice.service   # Docker-based user unit, currently DISABLED
  ~/.config/systemd/user/peeky-cry.service     # Docker-based user unit, DISABLED
```

**Why the user-mode units are disabled (re-spawn, 2026-06-15):** they were set up
against a Docker-based deploy (`docker run ... peeky-voxcpm2`) that never
happened. ml-engineer chose the venv+detached-uvicorn path instead (because the
committed `gpu_service/Dockerfile` pins x86_64/CUDA 12.4 and spark is
aarch64/GB10/CUDA 13). Re-enabling my units would either crash-loop on a missing
image, or `docker rm -f peeky-voice` (a no-op today) and try to start a second
voice process. The committed canonical unit is `ops/peeky-voice.service` (a
`multi-user.target` unit, venv-based, port 8081) — install it via `sudo cp` +
`sudo systemctl enable --now` for production auto-restart (see "Manual steps"
below).

---

## What infra-engineer set up

### On spark (safe, reversible; in `~/peeky` + `~/.config/systemd/user`)

- Synced the service code to `~/peeky/` (`gpu_service/`, `cry_service/`,
  `peeky_reachy/`, `pyproject.toml`, `assets/`) via `rsync` (the live code isn't
  pushed to GitHub yet, so this is the only copy under the prior tree).
- `~/peeky/enrollment/` — voice reference dir (mounted read-only into the voice
  container if/when the Docker-based deploy is used).
- Env files (still present, kept for documentation; not used by the current
  ml-engineer venv-based deploy — see "Current spark deploy layout" above):
  - `~/peeky/voice.env` — `PEEKY_VOICE_IMAGE`, `PEEKY_VOICE_PORT=8090`,
    `PEEKY_REFERENCES_DIR`, `PEEKY_ENROLL_HOST`, `PEEKY_EAGER_LOAD`.
  - `~/peeky/cry.env` — `PEEKY_CRY_IMAGE`, `PEEKY_CRY_PORT=8091`.
- `systemd --user` units (auto-restart `on-failure`, RestartSec=5). Both are
  **disabled** as of 2026-06-15 (re-spawn audit) — the live voice service runs
  via ml-engineer's detached `setsid` uvicorn, not via these user-mode units:
  - `~/.config/systemd/user/peeky-voice.service` → `docker run --gpus all -p
    ${PEEKY_VOICE_PORT}:8080 … peeky-voxcpm2`. DISABLED (would crash-loop on
    missing image; the venv-based detached uvicorn is the active deploy).
  - `~/.config/systemd/user/peeky-cry.service` → `docker run -p 8091:8080
    peeky-cry`. DISABLED. Contingency only — cry's real home is turing.

Nothing was started, no existing container/process was killed, no firewall or
driver was changed. The orphan user-mode units + env files in `~/peeky` are
retained for rollback / future Docker-based deploys.

### On turing (cry service — DEPLOYED, port 8081, user-mode systemd unit)

- rsynced the repo to `~/workspace/peeky_reachy/` (mirrors ml-engineer's
  convention on spark). Excludes: `.venv*`, `.git`, `__pycache__`,
  `*.egg-info`, `output`, `enrollment`, `.claude`.
- Created a venv at `~/workspace/peeky_reachy/.venv-cry/` via
  `python3 -m venv --without-pip` + bootstrapped pip (the bundled
  `ensurepip` is not present on Ubuntu 26.04's Python 3.14 — once the user
  runs `sudo apt install -y python3.14-venv`, `python3 -m venv` works
  without the bootstrap dance).
- Installed into the venv: `pip install -e ".[ml]" -r cry_service/requirements.txt`
  (fastapi 0.137.1, uvicorn 0.49.0, pydantic 2.13.4, numpy 2.4.6, plus
  onnxruntime 1.26.0, silero-vad 6.2.1, torch 2.12.0+cu13 with the
  nvidia-cu13 runtime libs). No `tensorflow`/`tensorflow-hub` — the service
  runs on the **numpy-heuristic** fallback (`model_loaded=false`). To enable
  the real YAMNet path later, `pip install "tensorflow>=2.13" "tensorflow-hub>=0.15"`.
- systemd `--user` unit installed and running: **`peeky-cry.service`** →
  `uvicorn cry_service.server:app --host 0.0.0.0 --port 8081` from the venv
  (PID 15935 since 2026-06-15 10:13). Source-of-truth files in the repo:
  - `ops/peeky-cry-turing.service` — user-mode variant (no sudo needed;
    what's installed today).
  - `ops/peeky-cry.service` — multi-user.target variant (production, needs
    `sudo cp` to `/etc/systemd/system/`).
- **Port conflict resolution:** host :8080 is anuj's `llama-swap` (different
  user, leave alone). Cry is on **:8081** instead.
- **Verifications done:**
  - `curl http://127.0.0.1:8081/healthz` →
    `{"ok":true,"model_loaded":false,"model":"numpy-heuristic"}` ✓
  - `POST /classify` with `ops/sample_cry.wav` →
    `{"event":"baby_cry","score":0.9595...}` ✓
  - `curl http://192.168.1.220:8081/healthz` from the dev Mac → **TIMEOUT**
    (ufw blocks inbound — see Manual step #0 below).

The user-mode unit will stop when no `bajajra` session is active (Linger=no).
For boot-time auto-start without an active session, the user runs once:
`sudo loginctl enable-linger bajajra` (and either switch the install to the
multi-user unit or keep this user-mode one).

---

## Operate the services (run from your Mac or on the box)

`ssh bajajra@<host>` first, or prefix each command with
`ssh bajajra@<host> '<cmd>'`.

### Cry service (turing) — live on :8081, user-mode systemd unit
```bash
ssh bajajra@192.168.1.220

systemctl --user status  peeky-cry      # status
systemctl --user restart peeky-cry      # restart (after editing the unit)
systemctl --user stop    peeky-cry      # stop
journalctl --user -u peeky-cry -f       # live logs
journalctl --user -u peeky-cry -n 100   # last 100 log lines
systemctl --user is-enabled peeky-cry   # boot autostart? (currently: enabled; will die on full logout until Linger is enabled)
```

The unit lives at `~/.config/systemd/user/peeky-cry.service` on turing
(source: `ops/peeky-cry-turing.service` in the repo). It's a `default.target`
unit, not `multi-user.target`, because turing has no passwordless sudo —
the user-mode install is the only one reachable without root.

> The service is **bound to 0.0.0.0:8081** but **ufw on turing blocks inbound
> 8081** (DEFAULT_INPUT_POLICY=DROP, no allow rule for 8081). From the dev
> Mac the connection times out. From inside turing (loopback or LAN if you
> SSH-tunnel) it works. **Manual step #0 in this runbook** opens 8081.

### Voice service (spark) — live on :8081, ml-engineer's deploy

### Voice service (spark) — live on :8081, ml-engineer's deploy
```bash
# Process check
ssh bajajra@192.168.1.253 'pgrep -af "uvicorn gpu_service.server"'

# Live logs (ml-engineer's detached uvicorn logs here):
tail -f /tmp/peeky-voice.log

# Stop (do not run unless you're sure you want to):
ssh bajajra@192.168.1.253 'pkill -f "uvicorn gpu_service.server"'

# Restart (re-run the setsid line from "Repro the live voice service" above)
```

> **The user-mode `systemctl --user peeky-voice` is disabled.** If/when we
> move the voice service to a `multi-user.target` systemd unit (the committed
> `ops/peeky-voice.service` in the repo is the canonical one), install with:
> ```bash
> sudo cp ops/peeky-voice.service /etc/systemd/system/peeky-voice.service
> sudo systemctl daemon-reload && sudo systemctl enable --now peeky-voice
> journalctl -u peeky-voice -f
> ```

### Cry service (contingency on spark :8091; primary = turing :8080)
```bash
# User-mode unit (Docker-based, currently DISABLED):
systemctl --user enable --now peeky-cry   # only if you also have the peeky-cry image built
systemctl --user status  peeky-cry
journalctl --user -u peeky-cry -f
systemctl --user disable --now peeky-cry  # turn the fallback back off
```
Cry is **not** running on spark today; the only path to a live cry service is
**turing :8080** (ml-engineer is the deploy owner, see `ops/models.md`).

### Health / smoke tests (from the dev Mac, over LAN)
```bash
# Voice — live on spark :8081 (ml-engineer):
curl -s http://192.168.1.253:8081/healthz
# expect: {"ok":true,"model_loaded":true,"model":"openbmb/VoxCPM2"}
curl -s http://192.168.1.253:8081/references    # {"references":[...]}
curl -s -X POST http://192.168.1.253:8081/synthesize \
  -H 'content-type: application/json' \
  -d '{"text":"hi baby, I am right here","sample_rate":48000}' \
  --output /tmp/peeky.wav && file /tmp/peeky.wav    # expect: RIFF (little-endian) WAVE

# Cry — turing :8081 (UP from inside; LAN blocked until ufw allows):
#   from the dev Mac (will TIMEOUT until ufw allows 8081):
curl -s -m 5 http://192.168.1.220:8081/healthz
#   from inside turing (works now):
ssh bajajra@192.168.1.220 'curl -s http://127.0.0.1:8081/healthz'
# expect: {"ok":true,"model_loaded":false,"model":"numpy-heuristic"}

# Cry fallback (spark :8091) — not started
curl -s http://192.168.1.253:8091/healthz
```

### Point Peeky at the services
```bash
# Cry (once turing :8081 is reachable from the Mac; today it is ufw-blocked):
export PEEKY_USE_REMOTE_CRY=true
export PEEKY_CRY_SERVICE_URL=http://192.168.1.220:8081   # port 8081, not 8080
# Voice (note the non-default port on spark):
export PEEKY_VOICE_CLONE_URL=http://192.168.1.253:8081   # current live port
```

---

## Manual steps still required (cannot/should not be done by infra-engineer)

> The cry service is **already running on turing :8081** (uvicorn PID 15935,
> user-mode systemd). What it needs from the user is just two allow-rules and
> (optionally) enabling boot-time auto-restart. None of this is destructive —
> all of it is just `ufw` + `systemctl` knobs.

0. **Open turing :8081 inbound (ufw).** ufw on turing has
   `DEFAULT_INPUT_POLICY=DROP` and the cry service on :8081 is currently
   unreachable from the dev Mac. Run on turing:
   ```bash
   sudo ufw allow 8081/tcp comment "peeky cry service (uvicorn)"
   ```
   Verify from the dev Mac: `curl -s -m 5 http://192.168.1.220:8081/healthz`
   should return `{"ok":true,"model_loaded":false,"model":"numpy-heuristic"}`.

1. **(Recommended) Enable turing linger for boot-time auto-start of the cry
   unit.** Currently the user-mode unit stops when no `bajajra` session is
   active (Linger=no on turing). For always-on, run once on turing:
   ```bash
   sudo loginctl enable-linger bajajra
   ```
   After this, `systemctl --user enable --now peeky-cry` will keep the service
   running across reboots and full logouts. (It is already enabled; the linger
   is what is missing.)

2. **(Optional, recommended) Install `python3.14-venv` for future-proofing.**
   The current `.venv-cry` was bootstrapped by dropping pip into a
   `python3 -m venv --without-pip` directory. A cleaner future setup uses:
   ```bash
   sudo apt install -y python3.14-venv
   ```
   Then `python3 -m venv` works directly. Not blocking — only matters if we
   ever blow away `.venv-cry` and want to rebuild from scratch.

3. **(Optional) Switch the cry unit to a multi-user.target install.** Today
   we use `~/.config/systemd/user/peeky-cry.service` (no sudo needed). The
   `ops/peeky-cry.service` in the repo is the multi-user variant. To install:
   ```bash
   sudo cp ops/peeky-cry.service /etc/systemd/system/peeky-cry.service
   sudo systemctl daemon-reload && sudo systemctl disable --now peeky-cry --user
   sudo systemctl enable --now peeky-cry
   sudo journalctl -u peeky-cry -f
   ```
   Then the service survives `bajajra` logout without needing linger. Skip
   unless you want the production-style install.

4. **(Optional) Build the GPU Docker image for spark.** **Not needed for the
   current deploy** — ml-engineer shipped the voice service on :8081 via
   venv+uvicorn (not Docker). The committed `gpu_service/Dockerfile` base
   `nvidia/cuda:12.4.1-…-ubuntu22.04` is **x86_64 / CUDA 12.4** and **will not
   run** on spark (aarch64 / GB10 / CUDA 13). If/when we move to a Docker-based
   voice deploy, fix the Dockerfile to a CUDA-13 aarch64 base + matching PyTorch
   + `pip install voxcpm`, then:
   ```bash
   ssh bajajra@192.168.1.253
   cd ~/peeky && docker build -f gpu_service/Dockerfile -t peeky-voxcpm2 .
   sudo systemctl enable --now peeky-voice   # canonical multi-user unit, not the user one
   journalctl -u peeky-voice -f
   ```

5. **(Optional) Cry fallback on spark.** If turing stays unavailable (e.g.
   `anuj` reboots the box), ml-engineer can build the cry image and start
   the user-mode unit on spark :8091. See "Cry service (contingency on
   spark :8091; primary = turing :8080)" in the spark section above.

6. **(Optional) Kill llama-swap + redeploy cry on turing :8080.** If we
   want cry on the default port (matches `cry_service/{Dockerfile,README,server.py}`
   docstrings), anuj's `llama-swap` on :8080 needs to be evicted. **Destructive
   op, NOT recommended without anuj's sign-off.** Proposed step:
   ```bash
   # on turing, as the user owning llama-swap (anuj) — NOT bajajra:
   pkill -f llama-swap   # or: systemctl --user stop ...  if it has a unit
   # then on turing, as bajajra, edit ~/.config/systemd/user/peeky-cry.service
   # change --port 8081 to --port 8080, then:
   systemctl --user daemon-reload && systemctl --user restart peeky-cry
   ```
   No infrastructure change — just one process dies and our unit rebinds.
   The cry client default `PEEKY_CRY_SERVICE_URL` then matches the
   committed `cry_service/` docstrings with no env override needed.

---

## Rollback (undo everything infra-engineer did)

### On spark
```bash
ssh bajajra@192.168.1.253

# (already done on 2026-06-15 re-spawn audit)
systemctl --user disable peeky-voice peeky-cry   # disable user-mode units
# If we ever want to remove them:
# rm ~/.config/systemd/user/peeky-voice.service ~/.config/systemd/user/peeky-cry.service
# systemctl --user daemon-reload

# Remove the prior infra-engineer's tree (does NOT contain the live voice service;
# the live service lives in ml-engineer's ~/workspace/peeky_reachy/ tree):
rm -rf ~/peeky
# Optional: revert boot-time user services
loginctl disable-linger "$USER"
```
**The live voice service in `~/workspace/peeky_reachy/` is ml-engineer's; do
not delete or restart that tree from this runbook.** If you must stop the
voice service, use `pkill -f "uvicorn gpu_service.server"` (or restart it
via ml-engineer's setsid command). No system packages, drivers, firewall
rules, or pre-existing containers were touched, so there is nothing else to
revert.

### On turing
```bash
ssh bajajra@192.168.1.220

# Stop and disable the cry service
systemctl --user disable --now peeky-cry
rm ~/.config/systemd/user/peeky-cry.service
systemctl --user daemon-reload

# Remove the repo + venv (also drops the manual ufw rule, if you added one)
rm -rf ~/workspace/peeky_reachy
# Optional: undo the ufw allow
# sudo ufw delete allow 8081/tcp

# Optional: undo the linger change
# sudo loginctl disable-linger bajajra
```
No system packages were installed (we used `python3 -m venv --without-pip`
+ bootstrap, so no `apt install` happened on turing). Nothing was killed —
we left anuj's `llama-swap` on :8080 alone. Drivers, kernels, and other
pre-existing services are untouched.

---

## Networking / firewall notes
- **spark**: ufw is not active (per prior audit). :8080 is reachable from the
  dev Mac; :8081 (current voice) follows the same openness. If you spin up
  the cry fallback on :8091, expect it to also be reachable from the LAN.
  (If a service is up but unreachable from the Mac, check ufw first:
  `ssh bajajra@192.168.1.253 'sudo ufw status'` and add an `allow` if
  needed.)
- **turing**: ufw is **ENABLED** with `DEFAULT_INPUT_POLICY="DROP"`. Only
  ports the user has explicitly allowed (currently :22 and :8080 for
  anuj's `llama-swap`) are reachable from the LAN. The cry service on
  :8081 is **NOT yet reachable from the dev Mac** — see Manual step #0.
  After the user runs `sudo ufw allow 8081/tcp`, :8081 is reachable.
- Both hosts are also on a Tailscale tailnet (100.x / fd7a: addresses
  present); LAN IPs above are what Peeky uses.
- No service has authentication — LAN-trust only (same threat model
  documented in `gpu_service/README.md`). Do not expose these ports beyond
  the LAN.
