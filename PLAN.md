# Peeky — AI Baby/Pet Monitor on Reachy Mini (MVP plan)

## Context

Build an AI-powered baby (and pet) monitor that runs as a **Reachy Mini app**. It
continuously listens to a room, detects when a baby or pet is crying, makes a
*best-effort* guess at the reason, and soothes the child by speaking calming
phrases **in the caregiver's cloned voice** plus gentle, expressive robot motion.

Decisions locked with the user:
- **Sim-only for now** (no physical robot). MuJoCo `--sim` gives head/antenna
  motion + camera but **no microphone/speaker** — so the audio pipeline runs
  against the **laptop's mic/speaker** during dev, behind an abstraction so the
  same code targets the real 4-mic robot later.
- **MVP scope: detect + caregiver voice cloning.** AI stories/lullabies are a
  later phase, but the code is structured to drop them in.
- **Heavy models run on the local GPU boxes** turing (192.168.1.220) and spark
  (192.168.1.253), reached over LAN (see `ssh-setup-local-gpu-machines` skill).
  Lightweight always-on detectors run on the dev machine. No cloud — audio stays
  on the LAN for privacy.

**Safety positioning (non-negotiable, baked into UX & docs):** Peeky is a
*soothing companion*, NOT a medical / SIDS / safety-monitoring device and must
never be relied on as one. Cry-*reason* classification is scientifically weak
(trained nurses ~33% accurate), so reasons are surfaced as low-confidence hints
with the caregiver kept in the loop — never as fact.

## Architecture

A single `ReachyMiniApp` whose `run(reachy_mini, stop_event)` loop drives a
pipeline, with I/O hidden behind interfaces so sim/laptop today == robot later.

```
mic ─► AudioIO ─► VAD (Silero) ─► EventClassifier (YAMNet) ─► SootheController
                                   (cry? dog? speech? silence)        │
                            optional ReasonHint (weak, flagged) ──────┤
                                                                      ▼
                                          VoiceCloneClient (HTTP→GPU)  + Motion
                                                      │                   │
                                                   speaker ◄── AudioIO    SDK
```

Key abstractions:
- **`AudioIO`** interface — `LocalAudioIO` (laptop, via `sounddevice`) for dev;
  `ReachyAudioIO` (wraps `mini.media.start_recording`/`get_audio_sample`/
  `push_audio_sample`/`play_sound`) for hardware. App depends only on the interface.
- **Motion** uses the Reachy SDK directly (`goto_target` / `set_target` /
  `create_head_pose`) — identical calls work in `--sim` and on hardware.
- **`VoiceCloneClient`** — thin HTTP client to the VoxCPM2 service on
  turing/spark; falls back to pre-recorded soothing tracks if the GPU is unreachable.

## Tech stack
- `reachy-mini[mujoco]` (SDK + sim), launched on macOS via
  `mjpython -m reachy_mini.daemon.app.main --sim`.
- `sounddevice` + `numpy` for laptop audio I/O (dev path).
- **Silero VAD** (`silero-vad`, ~2 MB, CPU) — always-on "is there sound" gate.
- **YAMNet** for event classification (baby cry / dog / speech / silence),
  Apache-2.0, tiny. Run via TF-Hub or an ONNX export; HF AST is the fallback if
  TF on macOS is painful.
- **VoxCPM2** (`openbmb/VoxCPM2`, Apache-2.0, 2B params, ~8 GB VRAM, CUDA ≥12 /
  PyTorch ≥2.5) for caregiver voice cloning — **primary engine**. Zero-shot clone
  from a short reference; best fidelity in "ultimate cloning" mode using the
  reference audio **+ its transcript**. 48 kHz output, 30 languages, real-time
  (RTF ~0.13–0.30 on a 4090). Served on turing/spark behind FastAPI; client via
  `httpx`. `pip install voxcpm`; `VoxCPM.from_pretrained("openbmb/VoxCPM2")`.
  *(OpenVoice v2, MIT/~2–3 GB, is the lighter fallback if VRAM is tight.)*
- *(Optional, flagged off by default)* a Dunstan/`donateacry` cry-reason model
  from HF for the weak reason hint.
- **LLM reason hints via gemma-4** (`gemma_service/` on turing:8082, FastAPI
  `/healthz` + `/reason`; lazy-loaded via `transformers` with a drafter for
  speculative decoding). Optional pipeline stage; falls back to the local
  heuristic on any 4xx/5xx/network failure. Pair with `RemoteEventClassifier`
  on turing:8081 (cry) and `VoiceCloneClient` → spark:8081 (VoxCPM2) for the
  full LAN deployment. See `ops/infra.md` + `ops/models.md` for the host
  topology, port map, and systemd units.
- **Robustness benchmark (donateacry).** Real-model eval output lives at
  `benchmarks/results/donateacry_real_models.json`; corpus at
  `benchmarks/donateacry-corpus/`. Used to tune the ensemble/abstain layer's
  confidence thresholds, not as the production model itself.

## Repo layout (greenfield — dir is empty)
```
peeky_reachy/
  pyproject.toml                 # deps + [project.entry-points."reachy_mini_apps"]
  peeky_reachy/
    app.py                       # ReachyMiniApp subclass + run loop
    config.py                    # thresholds, GPU host, feature flags
    audio/io.py                  # AudioIO interface + LocalAudioIO + ReachyAudioIO
    audio/vad.py                 # Silero wrapper
    detect/classifier.py         # YAMNet wrapper -> {cry,dog,speech,silence}+score
    detect/reason.py             # optional weak cry-reason hint (flagged)
    soothe/controller.py         # decision logic: what to do on a detection
    soothe/motion.py             # comforting head/antenna routines
    voice/clone_client.py        # HTTP client to GPU VoxCPM2 service
    voice/enroll.py              # consent gate + caregiver sample + transcript (CLI)
  gpu_service/voxcpm_server.py   # FastAPI wrapper around VoxCPM2 (runs on GPU box)
  assets/soothing/               # fallback pre-recorded calming tracks
  tests/                         # sample WAVs + threshold/pipeline tests
```

## Reuse / references
- Reachy SDK primitives: `ReachyMini`, `ReachyMiniApp`, `create_head_pose`,
  `mini.media.*`, `mini.goto_target`.
- **`pollen-robotics/reachy_mini_conversation_demo`** — closest reference for the
  always-on audio loop + app lifecycle; mirror its structure.
- `ssh-setup-local-gpu-machines` skill for deploying the VoxCPM2 service.

## Phased build
- **Phase 0 — Scaffold + sim hello.** Create package + `pyproject.toml` entry
  point. Connect to `--sim` daemon, run a simple head/antenna idle animation.
- **Phase 1 — Listening pipeline (laptop audio).** `LocalAudioIO` capture →
  Silero VAD → YAMNet → log `"baby cry detected (0.87)"`, `"dog"`, etc. Tune
  thresholds against sample clips.
- **Phase 2 — Soothe with fallback.** On a sustained cry, `SootheController`
  plays a pre-recorded calming track from `assets/` and triggers a comforting
  motion routine in the sim. Add cooldown/debounce + a caregiver notification log.
- **Phase 3 — Voice cloning (the MVP differentiator).** `enroll.py` captures a
  consented caregiver sample **plus its transcript** (for VoxCPM2 ultimate
  cloning), stored locally, encrypted-at-rest, never uploaded. Stand up
  `voxcpm_server.py` on turing/spark (`pip install voxcpm`, load
  `openbmb/VoxCPM2`). `VoiceCloneClient` POSTs phrase text + reference id and gets
  back 48 kHz audio; play through `AudioIO`. Falls back to Phase-2 tracks if GPU
  unreachable.
- **Phase 4 — Reason hint + polish.** Wire the flagged cry-reason model as a
  low-confidence hint shown in notifications/logs (off by default). Polish motion,
  debounce, config.
- **Future (out of MVP):** LLM-generated bedtime stories & lullaby lyrics →
  voice-clone TTS → cached playback (the pipeline already has the seam for it).

## Verification
- **Sim motion:** start `mjpython ... --sim`, run the app, confirm head/antenna
  motion in the 3D viewer.
- **Detection:** feed labeled WAVs (baby cry, dog, speech, silence) directly into
  the classifier and assert correct top class + threshold; also a live test
  playing a cry clip through the laptop speaker into the mic.
- **Unit tests:** VAD trigger and classifier thresholds against `tests/` clips.
- **Voice clone:** verify turing/spark meet VoxCPM2 reqs (CUDA ≥12, ~8 GB free
  VRAM); enroll a sample + transcript, hit the GPU service, play the 48 kHz
  result, listen for likeness + intelligibility; verify graceful fallback when GPU
  is down.
- **End-to-end (sim):** play a cry clip → app detects → speaks a cloned soothing
  line → robot runs the comfort motion in the viewer; verify cooldown prevents
  spamming.

## Open items to confirm during implementation
- Exact `mini.media.*` method names/signatures against the installed SDK version
  (the audio API is inferred from docs/repo and should be checked on first run).
- Whether `--sim` truly blocks audio; if it routes to host devices we can simplify
  the dev path — either way the `AudioIO` abstraction covers it.
- macOS GStreamer segfault workaround from the sim docs may be needed.
