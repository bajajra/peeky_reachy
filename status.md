# Peeky — Implementation Status

Ticket board for the Peeky baby/pet monitor build (see `PLAN.md` for the design,
`ROBUSTNESS.md` for the classification-robustness strategy).

**Tickets at a glance:** Complete: 25 · In progress: 6 · Not picked up: 0
Next unowned work: none — all 31 tickets are either Complete or In progress.
Active in-flight: **T26** (infra: turing cry on :8081, ufw still blocks), **T27**
(ml: voice on spark :8081, gemma install green on turing, deploy pending),
**T28** (LLM story/lullaby), **T29** (3D Reachy sidebar), **T30** (program-mgr audit).

**Status legend:** `Not picked up` · `In progress` · `Complete`
**Owner:** which session/agent handled the ticket.

| ID  | Ticket                                   | Status   | Owner       | Notes |
|-----|------------------------------------------|----------|-------------|-------|
| T0  | Scaffold package + pyproject + config    | Complete | claude-main | entry points + `config.py` |
| T1  | Audio I/O abstraction                    | Complete | claude-main | `audio/io.py` Local/Reachy/File/Null |
| T2  | VAD (Silero + energy fallback)           | Complete | claude-main | `detect/vad.py` asymmetric floor |
| T3  | Event classifier (YAMNet + heuristic)    | Complete | claude-main | `detect/classifier.py` |
| T4  | Reason hint (flagged, weak signal)       | Complete | claude-main | `detect/reason.py`, off by default |
| T5  | Detection events/types                   | Complete | claude-main | `detect/events.py` |
| T6  | Soothe controller                        | Complete | claude-main | `soothe/controller.py` |
| T7  | Comfort motion routines                  | Complete | claude-main | `soothe/motion.py` |
| T8  | Soothing responses + assets              | Complete | claude-main | `soothe/responses.py` + `assets/soothing/` |
| T9  | Voice enrollment + store                 | Complete | claude-main | consent gate + encrypted-at-rest |
| T10 | Voice clone client                       | Complete | claude-main | `voice/clone_client.py` |
| T11 | VoxCPM2 GPU service                       | Complete | agent-2     | `gpu_service/` FastAPI, 15 tests pass |
| T12 | Pipeline orchestrator                    | Complete | claude-main | `pipeline.py` |
| T13 | ReachyMiniApp + run loop                 | Complete | claude-main | `app.py` |
| T14 | CLI (run/enroll/demo)                    | Complete | claude-main | `cli.py` |
| T15 | Tests                                    | Complete | split       | core=claude-main (24) · GPU=agent-2 (15) |
| T16 | status.md + run docs                     | Complete | claude-main | this file |
| T17 | e2e demo run + verification              | Complete | claude-main | `pytest` 39 passed; `peeky-demo` triggers a soothe |
| T18 | Audio preprocessing + SNR (robustness)   | Complete | claude-main | `detect/preprocess.py` |
| T19 | Temporal smoothing + hysteresis          | Complete | claude-main | `detect/smoothing.py` |
| T20 | Ensemble classifier + abstain            | Complete | claude-main | `detect/ensemble.py` |
| T21 | Ambient calibration                      | Complete | claude-main | `pipeline.calibrate()` |
| T22 | Robust mood/reason aggregation           | Complete | claude-main | `EpisodeReasonAggregator` |
| T23 | Robustness strategy doc                  | Complete | claude-main | `ROBUSTNESS.md` |
| T24 | Gradio v6 web app                         | Complete | claude-main | `webapp.py` (gradio 6.18); Monitor/Enroll/Preview/About |
| T29 | 3D Reachy in Gradio left sidebar          | Complete | claude-main | Fixed-width left sidebar with a procedural three.js Reachy (`peeky_reachy/reachy3d.py` + `assets/reachy_3d/{states.json,ATTRIBUTION.md}`); `analyze()` maps pipeline run→state, hidden `gr.Textbox` + 200 ms JS poll → `window.peekyReachy.setState` (idle/listening/alert/comfort). `tests/test_webapp_3d.py` (12). Suite **184 passed**. Live on dev Mac http://127.0.0.1:7860. Procedural geometry instead of vendored URDF meshes — see ATTRIBUTION.md. |
| T30 | Program manager: docs + ticket board      | In progress | program-manager | owns `status.md`, `standup.md`, `PLAN.md`, `ROBUSTNESS.md`. Read-only on code. Does NOT commit/push. First pass: audit board, archive >24h heartbeats, add "Tickets at a glance" summary, flag doc↔code drift. |
| T31 | Gradio caregiver voice enrollment (priority) | In progress | ai-engineer | voice-from-mic as primary path; move Enroll tab to position #1; "Test this voice" button calls `VoiceCloneClient.synthesize`; end-to-end test in `test_webapp_enroll.py`. **BLOCKED**: ai-engineer is `backendType: in-process` (not a persistent tmux teammate) — messages sit in inbox until team-lead spawns them. |
| T32 | Fix voxwrap + re-deploy voice on spark (priority) | In progress | ml-engineer | `gpu_service/voxwrap.py` calls a non-existent method (voxcpm 2.0.3 API mismatch, /synthesize returns 503). Confirm real API, update wrapper + tests, pkill old uvicorn, git pull, relaunch on :8081, /synthesize smoke test → `/tmp/peeky-synth.wav`. |
| T33 | Audit + propose cleanup of running services (priority) | Complete | claude-main | Re-audit done (read-only, no kills) — see `ops/infra.md` "T33 re-audit" table. **No Peeky orphans: cry/gemma/voice all KEEP.** Corrected prior audit: spark :8088 is the user's separate `zoom_sitter` project (OFF-LIMITS, not a Peeky `server.app` orphan); gemma :8082 is KEEP not kill. |
| T25 | Remote cry-classification service (turing)| Complete | claude-main | `cry_service/` + `RemoteEventClassifier` + pipeline wiring |
| T26 | Infra: manage turing + spark             | In progress | infra-engineer | spark READY (voice on :8081 by ml-engineer); turing BACK ONLINE (RTX 5090, 32 GB, x86_64, Ubuntu 26.04). **Cry service RUNNING on turing :8081** (user-mode systemd) — now on **owlgebra-ai/babycry** (`PEEKY_CRY_MODEL=owlgebra`, `model_loaded:true`); `/classify` real cry → `baby_cry` 0.79, silence → `other`. **Blocker for LAN access: turing ufw blocks 8081 — user must `sudo ufw allow 8081/tcp`**. See `ops/infra.md` "Manual steps still required" #0 |
| T27 | ML: manage model services                | In progress | ml-engineer | voice→spark DEPLOYED (:8081, VoxCPM2 ~5.5GB VRAM); cry→turing BLOCKED (host offline); runbook `ops/models.md`. NOTE: voxwrap bug flagged in standup |
| T28 | LLM-powered bedtime story + lullaby gen  | In progress | ai-engineer-2 | `peeky_reachy/generate/` — Anthropic/Ollama/template fallback, voice-clone glue, Gradio tab factory |
| T34 | Autonomous live-streaming soothing mode  | Complete    | claude-main  | Per `vision.md`: pivot Gradio app from "upload + Analyze clip" to **live monitor as primary**. New `peeky_reachy/streaming.py` `StreamingSession` (thread-safe in-memory frame buffer + Pipeline worker + explicit sound-type→action map). Rewrote `webapp.py`: `🔴 Live monitor` tab is now Tab #1 with Start/Stop, live status Markdown, rolling 50-window timeline Dataframe, soothe Audio, gr.Timer(0.2) poll driving both the status and the existing 3D Reachy `reachy_state` textbox. Old upload-analyze flow demoted to "🛠 Debug / Analyze clip". New `tests/test_webapp_live.py` (13) covers `_LiveMonitor` lifecycle with a mocked `LocalAudioIO`, poll shape, and the `build_app()` vision-alignment structure (Live monitor is the first tab, has Start/Stop but no Analyze, drives the 3D state bridge). Suite **206 passed**. Lead taking this (ai-engineer is in-process, not running). |

**Model-service allocation:** turing (192.168.1.220) = baby-cry classification
(`cry_service/`, port **8081** — :8080 on turing is anuj's `llama-swap`, not
ours); spark (192.168.1.253) = VoxCPM2 voice clone (`gpu_service/`, port
**8081** — :8080 on spark is a pre-existing `nginx-llama-proxy`).
Enable remote cry with `PEEKY_USE_REMOTE_CRY=true`.

**Core tickets complete.** MVP runs e2e on the dev machine with numpy fallbacks
(no robot/GPU); YAMNet/Silero/VoxCPM2 slot into the same seams. Gradio v6 UI
(`peeky-web`) wraps the whole pipeline.

## Verification (T17)
- `.venv/bin/python -m pytest -q` → **43 passed** (core pipeline + GPU service + web app).
- `PEEKY_ASSETS_DIR=assets/soothing peeky-demo --wav <cry.wav>` → detects a
  sustained baby cry (score 0.98), notifies caregiver, plays the soothing track,
  and runs the comfort motion. Silence/speech inputs produce **0** false soothes.
- Gradio app builds, serves **HTTP 200** on `/`, and `analyze()` triggers a soothe
  on a cry clip (verified headless via `prevent_thread_lock`).

## How to run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . soundfile cryptography pytest      # core + test deps (no robot/GPU)
pytest -q
peeky-demo --wav <some_cry.wav>                     # file-driven e2e
peeky-enroll --name "Mom" --transcript "hush now little one" --wav mom.wav --i-consent
peeky run                                           # live laptop mic; add --robot for sim/hardware motion
pip install -e ".[web]" && peeky-web                 # Gradio v6 UI at http://127.0.0.1:7860
```
Optional extras: `pip install -e ".[audio,voice,ml,robot,web]"`.
GPU voice service: see `gpu_service/README.md` (runs on turing/spark).

## Environment notes
- Dev machine: macOS, sim-only (no physical Reachy Mini).
- Heavy models (VoxCPM2 voice clone) run on LAN GPU boxes turing/spark.
- Core pipeline degrades gracefully to numpy fallbacks (energy VAD, heuristic
  classifier, logged motion, fallback-track soothing) so the full flow runs and
  is testable on the dev machine.
- Safety: Peeky is a soothing **companion**, not a medical/SIDS/safety monitor;
  cry-reason inference is advisory only (see `ROBUSTNESS.md`).
