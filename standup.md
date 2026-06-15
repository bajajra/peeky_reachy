# Peeky — Agent Standup / Shared Comms

A shared log so the agents working on this repo stay coordinated. Append a new
entry under your name when you start, finish, or hand off work. Claim tickets in
`status.md` (Owner column) before starting. Keep entries short.

Active agents:
- **claude-main (team-lead)** — coordinating the build.
- **ai-engineer** — owns the client application (`peeky_reachy/**`, `tests/**`).
- **ai-engineer-2** — owns LLM story/lullaby generator (`peeky_reachy/generate/**`).
- **ai-engineer-3** — owns the client services layer (`detect/remote_classifier.py`, `voice/clone_client.py`, their tests, and live e2e against turing/spark).
- **infra-engineer** — owns turing + spark hosts (`ops/**`).
- (external, off-team) **ml-engineer** — model service deploys; coord via `standup.md`.

---

## claude-main

**Scope:** Building the entire MVP per `PLAN.md`, e2e, with numpy fallbacks so it
runs without robot/GPU. Tracking every ticket in `status.md`.

**Done:** T0 scaffold/config, T5 types, T1 audio I/O, T2 VAD, T3 classifier,
T4 reason hint, T6 soothe controller, T7 motion, T8 responses, T9 enrollment +
store (consent gate, encrypted-at-rest), T10 voice-clone client (built to your
T11 contract).

**Note on T11:** acknowledged — you own it. I will NOT write
`gpu_service/voxcpm_server.py`. My `voice/clone_client.py` already speaks your
`/healthz`, `/references`, `/synthesize` contract verbatim.

**Doing next (claiming):** T12 pipeline, T13 ReachyMiniApp, T14 CLI, core-pipeline
tests (T15), T17 verification — PLUS a new robustness workstream (user ask):
T18 preprocessing/SNR, T19 temporal smoothing + hysteresis, T20 ensemble +
abstain, T21 ambient calibration, T22 robust mood/reason aggregation, T23
robustness strategy doc. All new files live under `peeky_reachy/detect/` and
`peeky_reachy/` root — clear of your `gpu_service/` + GPU tests.

**DONE (all claude-main tickets):** core pipeline e2e complete — T0–T10, T12–T23.
`pytest -q` = **39 passed** (my 24 + your 15). `peeky-demo` triggers a real
soothe on a cry clip; silence/speech → 0 false soothes. Added robustness layer
(`detect/preprocess.py`, `smoothing.py`, `ensemble.py`, episode mood aggregation)
+ `ROBUSTNESS.md`. Flipped T11 → Complete in `status.md` per your note; your
contract matched my client 1:1, no rework. Nothing of yours edited.

**T24 added (user ask): Gradio v6 web app** — `peeky_reachy/webapp.py` (gradio
6.18) with Monitor / Enroll / Soothe-preview / About tabs over the pipeline;
`[web]` extra + `peeky-web` script. Added `ArrayAudioIO` + `on_window` callback +
`SootheEvent.audio` to feed the UI. Builds, serves HTTP 200, analyze triggers a
soothe. Full suite now **43 passed**. The web app's "Soothe preview" + Monitor
"voice clone" toggle both call your GPU `/synthesize` via `VoiceCloneClient`.

**T25 + spawned engineers (user ask):** Baby-cry classification is now a REMOTE
service on **turing** — new `cry_service/` (FastAPI `/healthz` + `/classify`,
reuses the package classifier) + `detect/remote_classifier.py`
(`RemoteEventClassifier`, falls back to local on any error) + pipeline wiring
(`use_remote_cry`). Allocation: **turing = cry (`cry_service/`, :8080)**,
**spark = voice (`gpu_service/`, :8080)**; I moved `voice_clone_url` default to
spark (192.168.1.253), cry → turing (192.168.1.220). Suite now **48 passed**.
Spawned **infra-engineer** (T26: hosts/SSH/GPU/systemd/networking → `ops/infra.md`)
and **ml-engineer** (T27: deploy cry→turing + voice→spark, health/smoke/versions
→ `ops/models.md`). They'll add their own sections below.

**Interfaces published (safe to build against):**
- `peeky_reachy.config.Config` — central config + env overrides.
- `peeky_reachy.detect.events` — `SoundEvent`, `CryReason`, `DetectionResult`.
- `peeky_reachy.audio.io.AudioIO` — `start/stop/read()->frame/play(samples, sr)`;
  backends `LocalAudioIO`, `ReachyAudioIO`, `FileAudioIO`, `NullAudioIO`.
- `peeky_reachy.detect.vad.make_vad(...)` and `classifier.make_classifier(...)`.

**Asks / coordination:**
- If you (agent-2) want a ticket I've claimed, ping here and flip Owner in
  `status.md`; I'll skip it. To avoid churn, please take *unclaimed* tickets
  first, or grab T15 tests / T11 GPU service which are well isolated.
- Heads up: I'm editing files across `peeky_reachy/`. If you're also writing
  there, name the exact files in your entry to avoid collisions.

---

## agent-2

**Scope:** Standalone deliverables that don't depend on claude-main's
in-progress core (no edits under `peeky_reachy/audio|detect|soothe/` or to
`config.py` / `__init__.py` / `status.md`).

**Claiming:**
- **T11 — VoxCPM2 GPU service** (`gpu_service/voxcpm_server.py` + Dockerfile
  + `pyproject.toml` extras). FastAPI wrapper, `POST /synthesize`
  (text + reference_id) → 48 kHz audio. Deployed on turing/spark. Matches
  the `voice_clone_url`/`voice_clone_timeout_s` already in `Config`.
- **T15 (partial) — tests for the GPU service** (HTTP layer only, mocked
  VoxCPM2). Will NOT touch test files for the core pipeline yet — those need
  claude-main's modules to stabilize.

**Published interface for T10 client to build against:**
- `POST {voice_clone_url}/synthesize` body `{"text": str, "reference_id": str,
  "language": str, "sample_rate": int}` → `audio/wav` bytes (PCM16, the
  requested `sample_rate`).
- `GET /healthz` → `{"ok": bool, "model_loaded": bool, "model": "openbmb/VoxCPM2"}`.
- `GET /references` → `["ref1", "ref2", ...]`.
- `POST /references` body `{"id": str, "audio_wav_b64": str, "transcript":
  str}` → registers a reference (used when T9 enroll uploads).
- All errors return `{"error": str, "detail": str}` with non-200.

**Will not edit:** anything in `peeky_reachy/audio|detect|soothe/voice`,
`peeky_reachy/config.py`, `peeky_reachy/__init__.py`, `status.md`. If
something there needs a tweak, I'll flag here first.

**Status:** T11 + T15 (GPU service slice) **DONE**.

**Shipped:**
- `gpu_service/__init__.py`, `gpu_service/voxwrap.py` (lazy model loader,
  reference resolution), `gpu_service/server.py` (FastAPI app, 4 endpoints,
  stdlib WAV encoder so the service only needs `voxcpm` on the GPU box).
- `gpu_service/requirements.txt`, `gpu_service/Dockerfile`
  (nvidia/cuda:12.4.1, listens on :8080, mounts enrollment dir).
- `gpu_service/README.md` — env vars, run commands, smoke-test curls, threat
  model note.
- `tests/test_gpu_service.py` — 15 tests, all passing (`pytest -q` in
  `.venv` with fastapi/httpx/pydantic installed; no GPU, no voxcpm — model
  is mocked via a `_FakeVoxCPM` that subclasses `VoxCPMWrapper`).

**Verified locally:** `pytest -q tests/test_gpu_service.py` → 15 passed.

**Not edited (per scope):** `peeky_reachy/**`, `pyproject.toml`, `status.md`.
claude-main should now flip T11 to Complete in `status.md`.

**Takeover plan if claude-main dies** (per user instruction):
- **Trigger:** no standup update for ≥5 min AND no file mtime advance in
  `peeky_reachy/` AND the process holding session `4dd5f72f-...` is gone
  (`pgrep -f 4dd5f72f || lsof | grep 4dd5f72f`).
- **Step 1** read every file under `peeky_reachy/` to learn the current shape
  (read `audio/io.py`, `detect/{events,vad,classifier,reason}.py`,
  `soothe/{controller,motion,responses}.py`, `config.py`).
- **Step 2** run `git status` to see uncommitted work — DON'T `git reset`,
  just review the diff and continue.
- **Step 3** resume from the first ticket in `status.md` whose Status is
  `In progress` or `Not picked up` and that isn't already implemented in
  the working tree. Skip the ones that are clearly done based on file
  presence.
- **Step 4** update `status.md` Owner column to `agent-2` for any ticket I
  take, then append a new "Takeover" entry to this standup explaining what
  I picked up.
- **Step 5** for tests (T15), do a smoke pass over each module with a
  minimal happy-path test (don't try to write a comprehensive suite in
  takeover mode — cover the seams).
- **Do not** delete or rewrite claude-main's files. Treat their code as
  read-only and add to it / around it.

---

## Heartbeats (agent-2 → status check)

- 22:43 — agent-2 checked: claude-main ALIVE (PID 27807, session
  4dd5f72f), mtime advance 3 min ago (preprocess/smoothing/ensemble/reason
  — robustness workstream). T11 still "In progress" in their board;
  will wait for them to flip to Complete. No takeover.

- 22:51 — agent-2 checked: claude-main ALIVE (PID 27807), mtime advance
  within last 2 min (vad.py + classifier.py just touched at 22:49 — looks
  like post-completion polish). All tickets T0–T23 are now Complete in
  status.md; 39 pytest passing. T11 flipped to Complete. Nothing to do.

- 23:01 — agent-2 checked: claude-main ALIVE (PID 27807), no mtime advance
  in last 12 min (last edits 22:49 — looks idle, but process still
  running). All T0–T23 still Complete. Nothing to do.

- 23:11 — agent-2 checked: claude-main ALIVE (PID 27807), still no mtime
  advance since 22:49 (~22 min idle). All T0–T23 still Complete. Nothing
  to do.

- 23:21 — agent-2 checked: claude-main ALIVE (PID 27807), ~32 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 23:31 — agent-2 checked: claude-main ALIVE (PID 27807), ~42 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 23:41 — agent-2 checked: claude-main ALIVE (PID 27807), ~52 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 23:51 — agent-2 checked: claude-main ALIVE (PID 27807), ~62 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 00:01 — agent-2 checked: claude-main ALIVE (PID 27807), ~72 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 00:11 — agent-2 checked: claude-main ALIVE (PID 27807), ~82 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 00:21 — agent-2 checked: claude-main ALIVE (PID 27807), ~92 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 00:31 — agent-2 checked: claude-main ALIVE (PID 27807), ~102 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 00:41 — agent-2 checked: claude-main ALIVE (PID 27807), ~112 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 00:51 — agent-2 checked: claude-main ALIVE (PID 27807), ~122 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 01:01 — agent-2 checked: claude-main ALIVE (PID 27807), ~132 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 01:11 — agent-2 checked: claude-main ALIVE (PID 27807), ~142 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 01:21 — agent-2 checked: claude-main ALIVE (PID 27807), ~152 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 01:31 — agent-2 checked: claude-main ALIVE (PID 27807), ~162 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 01:41 — agent-2 checked: claude-main ALIVE (PID 27807), ~172 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 01:51 — agent-2 checked: claude-main ALIVE (PID 27807), ~182 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 02:02 — agent-2 checked: claude-main ALIVE (PID 27807), ~193 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 02:12 — agent-2 checked: claude-main ALIVE (PID 27807), ~203 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 02:21 — agent-2 checked: claude-main ALIVE (PID 27807), ~212 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 02:31 — agent-2 checked: claude-main ALIVE (PID 27807), ~222 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 02:41 — agent-2 checked: claude-main ALIVE (PID 27807), ~232 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 02:51 — agent-2 checked: claude-main ALIVE (PID 27807), ~242 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 03:01 — agent-2 checked: claude-main ALIVE (PID 27807), ~252 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 03:11 — agent-2 checked: claude-main ALIVE (PID 27807), ~262 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 03:21 — agent-2 checked: claude-main ALIVE (PID 27807), ~272 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 03:31 — agent-2 checked: claude-main ALIVE (PID 27807), ~282 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 03:41 — agent-2 checked: claude-main ALIVE (PID 27807), ~292 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 03:51 — agent-2 checked: claude-main ALIVE (PID 27807), ~302 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 04:01 — agent-2 checked: claude-main ALIVE (PID 27807), ~312 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 04:11 — agent-2 checked: claude-main ALIVE (PID 27807), ~322 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 04:21 — agent-2 checked: claude-main ALIVE (PID 27807), ~332 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 04:31 — agent-2 checked: claude-main ALIVE (PID 27807), ~342 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 04:41 — agent-2 checked: claude-main ALIVE (PID 27807), ~352 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 04:51 — agent-2 checked: claude-main ALIVE (PID 27807), ~362 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 05:01 — agent-2 checked: claude-main ALIVE (PID 27807), ~372 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 05:11 — agent-2 checked: claude-main ALIVE (PID 27807), ~382 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 05:21 — agent-2 checked: claude-main ALIVE (PID 27807), ~392 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 05:31 — agent-2 checked: claude-main ALIVE (PID 27807), ~402 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 05:41 — agent-2 checked: claude-main ALIVE (PID 27807), ~412 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 05:51 — agent-2 checked: claude-main ALIVE (PID 27807), ~422 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 06:01 — agent-2 checked: claude-main ALIVE (PID 27807), ~432 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 06:11 — agent-2 checked: claude-main ALIVE (PID 27807), ~442 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 06:21 — agent-2 checked: claude-main ALIVE (PID 27807), ~452 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 06:31 — agent-2 checked: claude-main ALIVE (PID 27807), ~462 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 06:41 — agent-2 checked: claude-main ALIVE (PID 27807), ~472 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 06:51 — agent-2 checked: claude-main ALIVE (PID 27807), ~482 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 07:01 — agent-2 checked: claude-main ALIVE (PID 27807), ~492 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 07:12 — agent-2 checked: claude-main ALIVE (PID 27807), ~503 min idle
  (mtimes unchanged since 22:49). All T0–T23 still Complete. Nothing
  to do.

- 07:22 — agent-2 checked: claude-main ALIVE (PID 27807), mtimes advanced
  ~07:19–07:21 — they woke up. New `webapp.py` (11k), edits to
  `audio/io.py` and `pipeline.py`. No new tickets in `status.md` yet
  (T0–T23 still all Complete). No takeover; flagging in case they want
  to claim webapp work.

- 07:31 — agent-2 checked: claude-main ALIVE (PID 27807), latest mtime
  07:21:51 (`webapp.py`) — they finished the T24 Gradio webapp work then
  went idle. All T0–T23 still Complete; T24 (webapp) is in their standup
  but not yet in `status.md` board. No takeover.

- 07:41 — agent-2 checked: claude-main ALIVE (PID 27807), still idle
  (mtimes unchanged since 07:21). All T0–T23 Complete. No takeover.

- 07:51 — agent-2 checked: claude-main ALIVE (PID 27807), still idle
  (mtimes unchanged since 07:21). All T0–T23 Complete. No takeover.

- 08:03 — agent-2 checked: claude-main ALIVE (PID 27807), still idle
  (mtimes unchanged since 07:21). All T0–T23 Complete. No takeover.

- 08:11 — agent-2 checked: claude-main ALIVE (PID 27807), woke up again.
  mtimes 08:08–08:09: new `detect/remote_classifier.py` (2.3k) + edits
  to `pipeline.py` and `config.py`. T24 (Gradio webapp) now in
  `status.md` Complete. New work looks like T25 (remote classifier
  client?). No takeover.

- 08:21 — agent-2 checked: claude-main ALIVE (PID 27807), idle
  (mtimes unchanged since 08:09). Two new In-progress tickets:
  **T26 infra-engineer (turing OFFLINE — needs user power-on,
  spark READY per `ops/infra.md`)** and **T27 ml-engineer (deploy
  cry→turing, voice→spark)**. Not my tickets; flagging in case the
  user wants to act on the turing power-on. No takeover.

---

## infra-engineer (re-spawned 2026-06-15)

**Re-spawned by team-lead** to pick up where the prior infra-engineer left off
(spark READY, turing OFFLINE). Same scope and guardrails as before — T26 only.
Read `ops/infra.md` for the full audit, then re-verify spark is still healthy
and chase the turing power-on. For risky ops, write the step into `ops/infra.md`
and let the user run it.

**SSH status:**
- **spark (192.168.1.253): READY.** Key-based SSH works (`team_infra_ed25519`,
  no password). Full audit done.
- **turing (192.168.1.220): OFFLINE.** No ping, tcp/22 timeout, ARP incomplete
  — the box is powered off / off the LAN. Cannot configure remotely. **USER
  ACTION:** power it on, then `ssh bajajra@192.168.1.220 hostname`; if it asks
  for a password, run `ssh-copy-id -i ~/.ssh/team_infra_ed25519.pub
  bajajra@192.168.1.220` (do not guess passwords).

**spark audit:** Ubuntu 24.04.4, aarch64, 20c/121 GiB, 2.2 TB free, Python
3.12.3, Docker 29.2.1 w/ nvidia runtime. GPU = **NVIDIA GB10 (Grace-Blackwell /
DGX Spark)**, driver 580.159.03, CUDA 13.0, unified memory (>>8 GB). systemd
--user works, linger enabled.

**Port conflict found:** host :8080 on spark is already taken by a pre-existing
`nginx-llama-proxy` container (do NOT disturb). So Peeky services use alt ports
on spark: **voice :8090**, cry-fallback :8091. LAN firewall is open.

**Set up (safe, reversible; on spark):** rsynced service code to `~/peeky`;
env files `voice.env`/`cry.env`; `systemd --user` units `peeky-voice`
(enabled, inactive — awaiting GPU image) and `peeky-cry` (contingency only,
disabled — cry's real home is turing). Nothing started, nothing killed.

**Runbook:** `ops/infra.md` — host specs, port map, start/stop/status/log per
service, smoke tests, rollback, and the manual steps below.

**For ml-engineer:** spark SSH + host are READY now. Blockers before voice can
start: the committed `gpu_service/Dockerfile` targets x86_64/CUDA 12.4 but spark
is aarch64/GB10/CUDA 13 — you must build an arm64+Blackwell `peeky-voxcpm2`
image (then `systemctl --user start peeky-voice` on spark; unit auto-restarts).
Cry service is blocked on turing being powered on; spark :8091 fallback unit is
ready if you want cry up immediately (`docker build -t peeky-cry`, then
`systemctl --user enable --now peeky-cry`). Details + commands in `ops/infra.md`.

### infra-engineer (2nd re-spawn, 2026-06-15 ~09:00) — current pass

**Re-verified spark.** Same OS/arch/CPU/RAM/GPU/driver as before; nothing
drifted. `systemd --user` linger still on, env files in place, repo tree in
`~/peeky/` intact.

**Major finding — my prior setup is orphaned:**
ml-engineer shipped the voice service to **spark :8081** as a detached uvicorn
in `~/workspace/peeky_reachy/.venv-gpu/`, NOT via Docker. `ops/models.md`
documents the deploy; their `systemd` unit `ops/peeky-voice.service`
(multi-user, port 8081) is in the repo but **not yet installed** (needs
`sudo cp` to `/etc/systemd/system/`, which ml-engineer didn't run).

`curl http://192.168.1.253:8081/healthz` →
`{"ok":true,"model_loaded":true,"model":"openbmb/VoxCPM2"}`. The model is loaded
and warm.

**voxcpm wrapper bug confirmed on the live service:** ml-engineer's voice log
(`/tmp/peeky-voice.log`) shows a `/synthesize` call returned `503` at 08:26:42
with the warning: `model unavailable: VoxCPM2 has no generate/synthesize/tts/
infer method we can call`. This is the **voxcpm 2.0.3 API mismatch** flagged in
T27 / `ops/models.md` section 2. The voice service is up but synthesize is
broken until ml-engineer fixes `gpu_service/voxwrap.py` (their code — I do not
edit it). Health still 200, model loaded, but no successful synth yet.

**Actions taken (safe, reversible):**
- Disabled both user-mode units: `systemctl --user disable peeky-voice
  peeky-cry`. The units are still on disk; the symlinks in
  `default.target.wants/` are removed. Won't auto-start on next boot, won't
  `docker rm -f peeky-voice` on next login.
- Updated `ops/infra.md` to reflect the live port (8081, not 8090), the
  ml-engineer tree in `~/workspace/peeky_reachy/`, the disabled user units, and
  the new health/smoke URLs.
- Updated `status.md` T26 row (notes column) to point at the current state.

**Did NOT touch:** the running voice uvicorn (PID 3463795), `~/workspace/
peeky_reachy/`, the `nginx-llama-proxy` / `infra-caddy-1` containers, any
firewall rules, turing (still offline), `cry_service/` / `gpu_service/`
code, `peeky_reachy/`, ml-engineer's systemd unit files in the repo.

**For ml-engineer:** the orphan user units are no longer the source of truth
for spark. The canonical production unit is the one in `ops/peeky-voice.service`
(multi-user, port 8081); install it via `sudo cp ops/peeky-voice.service
/etc/systemd/system/peeky-voice.service && sudo systemctl enable --now
peeky-voice` when you want auto-restart. Synthesize is still broken on the
live process — see your own `ops/models.md` and the 08:26:42 503 in
`/tmp/peeky-voice.log`.

**Still blocked:** turing is still OFFLINE. **USER ACTION unchanged:** power
on turing (192.168.1.220), then `ssh bajajra@192.168.1.220 hostname`; if it
prompts for a password, `ssh-copy-id -i ~/.ssh/team_infra_ed25519.pub
bajajra@192.168.1.220`. Once reachable, I'll audit turing, rsync the cry
service, build `.venv-cry` (per `ops/models.md` section 1), and install
`ops/peeky-cry.service` (the multi-user unit, port 8080).

---

## ai-engineer

**Scope:** Own the client application (`peeky_reachy/`) + dev testing. Picking
up where claude-main left off; NOT editing `gpu_service/**`, `cry_service/**`,
or `ops/**` (other agents' surfaces).

**Verified (2026-06-15):**
- `pytest -q` baseline: **48 passed**.
- `peeky-demo --wav <synthetic cry.wav>` (silence prefix + harmonic-rich cry):
  triggers a soothe (score 0.98), plays the fallback hum track, logs comfort
  motion. Silence / pure-speech / tiny clips: **0** false soothes.
- Gradio webapp: built via `build_app()`, launched headless on :7891, served
  HTTP 200. Direct calls to `analyze`/`enroll`/`preview` all behave correctly
  (cry→soothe with audio; empty→friendly message; consent gate works;
  preview falls back to the default hum track when GPU is unreachable).

**Tests added (`tests/test_remote_integration.py` + `tests/test_webapp_edges.py`):**
- `RemoteEventClassifier` happy path via `httpx.MockTransport` (asserts
  `/healthz` + `/classify` JSON contract; tests `model_loaded=false`,
  fallback on 500, short-window encoding).
- `VoiceCloneClient` happy path (register-once-then-synth, skip-register if
  already known, 503→None, no-enrollment→None, network error→`available=False`).
- Pipeline glued to the mocked remote classifier: e2e cry triggers soothe via
  the remote path; if remote is unhealthy the pipeline silently falls back
  to local.
- Webapp edge cases: int16-stereo mono-down, zero-length input, very short
  clips, consent + name/transcript validation, enroll happy path,
  preview-with-cloned-voice (mocked GPU) + fallback-track path + nothing-at-all
  path, `analyze(use_voice=True)` against a mocked spark.
- **New total: 72 passed (was 48; +24 new).** All offline, deterministic, no
  real network.

**Live integration:**
- spark voice service @ `http://192.168.1.253:8090/healthz`: connection
  refused (port closed). Per infra-engineer's runbook the `peeky-voice`
  systemd unit is enabled-but-inactive, waiting on ml-engineer to build the
  arm64/Blackwell `peeky-voxcpm2` image. **Cannot e2e the cloned voice yet.**
- spark :8080 = pre-existing `nginx-llama-proxy` (do not disturb), confirmed
  502 from peeky's perspective. The client code already handles this; the
  unit test `test_voice_clone_returns_none_when_service_500s` covers the
  shape.
- turing cry service @ `http://192.168.1.220:8080`: timed out (turing still
  powered off). Client falls back to local heuristic, verified.

**Findings / non-blocking suggestions (no fix applied — flagging for owner):**
- `peeky_reachy/soothe/controller.py:42` — `in_cooldown` returns True forever
  after the first action (returns `self._last_action_at is not None`). It's
  unused (`cooldown_remaining` is the real gate), so this is dead code rather
  than a live bug. Either delete or fix to `cooldown_remaining(now) > 0`.
- `peeky_reachy/webapp.py:160` — `preview()` uses `pick_phrase.__doc__` as a
  fallback when the user-entered phrase is blank; `pick_phrase` has no
  docstring, so it actually falls through to the literal `"Shhh, it's okay."`.
  Works, but reads like a bug. Suggest `pick_phrase(SoundEvent.BABY_CRY)`.
- `peeky_reachy/pipeline.py:132-135` — per-frame `DetectionResult.reason` is
  always UNKNOWN; the real reason is filled in `_execute` from the episode
  aggregator. Intentional, but a one-line comment would prevent future
  "is this a bug?" reads.

**Heads-up for ai-engineer-2:** I see you're working stories/lullabies and
plan to send a `build_story_tab()` wiring patch into the webapp. The webapp
is fine to extend; I haven't touched any of its existing entry points.
My new tests (`test_webapp_edges.py`) only call `analyze` / `enroll` /
`preview` and `_to_float_mono` / `_to_gradio_audio`, so a new tab should
not interfere.

---

## agent-2 heartbeats

- 08:32 — checked, claude-main alive (PID 27807, ~20h uptime). T28 still in flight (ai-engineer-2 just wrote 7 files in generate/). No action.
- 08:41 — checked, claude-main alive (PID 27807, 20h23m uptime). No new file activity since last check. No action.
- 08:51 — checked, claude-main alive (PID 27807, 20h33m uptime). No new file activity since last check. No action.
- 09:02 — checked, claude-main alive (PID 27807, 20h43m uptime). No new file activity since last check. No action.
- 09:11 — checked, claude-main alive (PID 27807, 20h53m uptime). infra-engineer active on T26 (`ops/infra.md` updated). No action.
- 09:21 — checked, claude-main alive (PID 27807, 21h03m uptime). ai-engineer-3 active on `detect/remote_classifier.py` and `voice/clone_client.py` tests. No action.
- 10:02 — checked, claude-main alive (PID 27807, 21h43m uptime). All quiet, no file activity in last 10 min. No action.
- 10:11 — checked, claude-main alive (PID 27807, 21h53m uptime). Quiet. No action.
- 10:22 — checked, claude-main alive (PID 27807, 22h03m uptime). infra-engineer active on T26 — wrote `ops/peeky-cry.service` and `ops/peeky-cry-turing.service` systemd units. No action.
- 10:31 — checked, claude-main alive (PID 27807, 22h13m uptime). Quiet. No action.
- 10:42 — checked, claude-main alive (PID 27807, 22h23m uptime). No file activity. No action.
- 10:51 — checked, claude-main alive (PID 27807, 22h33m uptime). Quiet. No action.
- 11:01 — checked, claude-main alive (PID 27807, 22h43m uptime). ai-engineer-3 added `benchmarks/donateacry-corpus/` (cry eval dataset, likely for live e2e). No action.

---

## ai-engineer-3

**Scope:** client-services layer only — `peeky_reachy/detect/remote_classifier.py`,
`peeky_reachy/voice/clone_client.py`, `peeky_reachy/config.py` (URL/timeout/flag
fields), plus `tests/test_remote_classifier.py`, `tests/test_cry_service.py`,
and a new `tests/test_voice_clone.py`. Not editing pipeline/app/cli/voice-store/
cry-service/gpu-service code.

**Live reachability (2026-06-15):**
- `curl -m 3 http://192.168.1.220:8080/healthz` → turing OFFLINE (no route; turing
  is powered off per infra-engineer, awaiting user power-on).
- `curl -m 3 http://192.168.1.253:8090/healthz` → connection refused (the systemd
  unit is installed but inactive; ml-engineer still owes the arm64/Blackwell
  `peeky-voxcpm2` image).
- `curl -m 3 http://192.168.1.253:8081/healthz` →
  `{"ok":true,"model_loaded":true,"model":"openbmb/VoxCPM2"}` — the live
  ml-engineer uvicorn is on :8081, not :8090. Synthesize is broken on the live
  process (voxcpm 2.0.3 API mismatch, per `/tmp/peeky-voice.log` and infra-engineer
  T27 note). So even when reachable, `peeky-demo` will fall back to the hum
  track — same fallback path the client already handles cleanly.

**Test suite status:** `pytest -q` → **115 passed** (was 89; +26 new client-service
tests, all offline, deterministic, no real network).

**New / changed files:**
- `peeky_reachy/config.py` — `voice_clone_url` default flipped from
  `http://192.168.1.253:8080` → `http://192.168.1.253:8090` (the systemd unit
  port; :8080 on spark is the pre-existing nginx-llama-proxy per
  `ops/infra.md`). Picked up orphaned ticket #12. Live process is on :8081; if
  the lead wants :8081 as the new default I can flip it, but :8090 is the
  documented systemd port and matches the runbook.
- `peeky_reachy/detect/remote_classifier.py` — hardened `_wav_b64`:
  1. **Stereo → mono downmix** (was broadcast-flattening, doubling the duration
     the server would see for a 2-channel input).
  2. **int16 passthrough** (was re-quantizing int16→float→int16, trimming
     1 LSB off the top end and adding rounding error — caught by
     `test_wav_b64_int16_passthrough`).
  3. **Empty / 1-sample / out-of-range inputs** now encode cleanly.
  4. Rounding via `np.round` so the float→int16 conversion is symmetric.
- `tests/test_remote_classifier.py` — rewrote + 11 new tests: encoder edge
  cases (float32 mono, int16 passthrough, stereo downmix, empty, clamp),
  JSON contract (`available()` true/false/5xx), happy-path asserts the
  **local fallback does NOT fire** when remote returns a confident score,
  4xx→fallback, 5xx→fallback, real-network timeout→fallback, single-sample
  window, `classify()` does NOT also hit `/healthz` (i.e. it stays a single
  round-trip — no double-`/healthz`-then-`/classify`).
- `tests/test_voice_clone.py` — **new file** (didn't exist). 9 tests:
  `available()` happy + `model_loaded=False` + connection-error, register-only-
  once-per-session, no-POST-when-already-on-server, register-4xx-doesn't-cache,
  synth happy path (asserts the returned audio actually has energy, not just
  HTTP 200), 5xx→None, synth-404→None (stale-cache scenario), register-500→None
  (synth must NOT be hit), explicit `speaker_id` overrides default.

**Not in my scope (flagged, not fixed):**
- The orphaned `peeky-voice` systemd unit + :8081 live uvicorn mismatch —
  infra-engineer / ml-engineer territory. Standup note already covers it.
- `voice/clone_client.py:79` `ensure_reference` caches "known" in-process only;
  if the server is restarted and its `references/` dir is wiped, the next
  `synth()` will 404. The new `test_voice_clone_returns_none_on_synthesize_404`
  covers the failure mode (returns None, never raises), but a real fix would
  be to clear the in-process cache on a 404 from `/synthesize` and retry
  `ensure_reference` once. Want me to ship that, or leave it as a known
  limitation since the spa spark references dir is durable across restarts?

**Still blocked on live e2e:** turing cry service (host offline) + spark
voice service (image missing / synthesize 503). Once those are green I'll
run the live `peeky-demo --wav <clip>` and `peeky-enroll` checks per the
brief and report here.

**Proposed next (waiting for lead approval):**
- (a) `clone_client.py` re-register on synth 404 (see above).
- (b) Add a `test_remote_classifier.py` test that asserts the local
  `classify()` is NOT called on a confident remote score, using a counting
  fallback subclass — already done (`test_classify_happy_path_uses_remote_score`).
- (c) Once live e2e is possible: a script under `tests/live/` that does the
  full e2e against real turing/spark behind a `--live` flag, skipped by
  default.

### infra-engineer (3rd pass, 2026-06-15 ~10:20) — turing is back, cry is up

**User said: "turing is back online."** Confirmed. Key audit (full table in
`ops/infra.md`):

- **turing = the RTX 5090 box** (per team-lead's confirmation + my
  `nvidia-smi -L`: "NVIDIA GeForce RTX 5090, 32607 MiB"). x86_64, Ubuntu
  26.04 LTS ("Resolute Raccoon"), kernel 7.0.0, AMD Ryzen 9 9950X3D
  16C/32T, 121 GiB RAM, 1.3 TB free, Python 3.14.4. No passwordless sudo.
  systemd --user works (3 bajajra sessions, Linger=no).
- **Turing has internet** (PyPI=200, github=200) — re-verified after
  team-lead's note. No proxy config, just default route.
- **`:8080` is owned by `anuj` (different user), running `llama-swap`
  PID 7105, config `/home/anuj/owlagents/infra/bajara/llama-swap.yaml
  --listen :8080 -watch-config`.** Not in Docker. **DO NOT KILL** — that
  user owns it. The user's "kill other services for now" message was
  about freeing :8080 generally; the safe path is to use a different port
  (8081), which I did.
- **Turing `ufw` is ENABLED with `DEFAULT_INPUT_POLICY="DROP"`** and the
  user-rules files are empty — so only the ports anuj already opened
  (:22, :8080) are reachable from the LAN. **My new :8081 is NOT yet
  reachable from the dev Mac until the user adds a ufw allow.**

**What I did (safe, reversible):**
- rsynced repo to `~/workspace/peeky_reachy/` on turing (mirrors
  ml-engineer's spark convention).
- Created `.venv-cry` via `python3 -m venv --without-pip` + bootstrap
  pip (Ubuntu 26.04's `python3.14` ships without `ensurepip`).
- Installed in the venv: `pip install -e ".[ml]" -r cry_service/requirements.txt`.
  Resolved to fastapi 0.137.1, uvicorn 0.49.0, pydantic 2.13.4, numpy
  2.4.6, onnxruntime 1.26.0, silero-vad 6.2.1, torch 2.12.0+cu13 (with
  nvidia-cu13 runtime libs). **No tensorflow** — service runs on the
  numpy-heuristic fallback.
- Wrote `ops/peeky-cry-turing.service` (user-mode systemd, no sudo, no
  linger requirement; binds 0.0.0.0:8081). Installed as
  `~/.config/systemd/user/peeky-cry.service` on turing.
- Updated `ops/peeky-cry.service` to point at the venv + port 8081 too
  (the multi-user-target production variant).
- `systemctl --user enable --now peeky-cry` — service RUNNING
  (PID 15935).
- **Verified end-to-end:**
  - `curl http://127.0.0.1:8081/healthz` →
    `{"ok":true,"model_loaded":false,"model":"numpy-heuristic"}` ✓
  - `POST /classify` with `ops/sample_cry.wav` →
    `{"event":"baby_cry","score":0.9595...}` ✓ (numpy-heuristic
    correctly classifies)
  - `curl http://192.168.1.220:8081/healthz` from the dev Mac →
    **TIMEOUT** (ufw blocks; expected)

**Did NOT touch:** turing's ufw (no `sudo ufw` rule added), anuj's
llama-swap, any pre-existing service, the system Python, the kernel, the
GPU, turing's user (still `bajajra`, no group changes). ml-engineer's
code in `cry_service/` / `gpu_service/` / `peeky_reachy/` was not edited.
`peeky_reachy/` is also untouched (per scope).

**Runbook updates:** `ops/infra.md` now has the turing host audit row
populated (no longer "OFFLINE"), a turing port map, a "On turing" setup
section, a turing cry service in "Operate the services", and 6 manual
steps (the most important is #0: `sudo ufw allow 8081/tcp`). Rollback
section split into spark + turing. Status.md T26 row updated.

**On the user's "free up resource on spark" message:** Spark has plenty
of headroom (2.2 TB free disk, 121 GiB RAM, GB10 GPU mostly idle), and
the priority service is the **cry service on turing** (which is now up).
Spark is **not** a blocker for anything I'm doing. I made no changes to
spark in this pass. Flagging in case the user meant to free resources on
**turing** for vllm (separate workstream, ml-engineer's domain — the
RTX 5090's 32 GB VRAM is the right target for that, but the cry venv
only uses ~1-2 GB RAM and 0 VRAM so there's no actual conflict).

**Still blocked for the LAN path:** the user must run `sudo ufw allow
8081/tcp` on turing (one line). After that, the cry service is
fully LAN-reachable from the dev Mac and the existing
`PEEKY_CRY_SERVICE_URL=http://192.168.1.220:8081` env override (or the
new `cry_service_url` default) just works.

**For ml-engineer (separate from infra):** the synth bug on the spark
voice service is still there — `/tmp/peeky-voice.log` showed `503
... voxcpm 2.0.3 has no generate/synthesize/tts/infer method we can call`
yesterday. Not my code; flagging in case you missed it.
