# Robustness Strategy — Source & Mood Audio Classification

A nursery is not a lab. Sounds arrive at varying distance and volume, over HVAC
hum, TV chatter, sibling noise and reverb. Two very different problems hide under
"classify the audio", and we treat them differently:

- **Source classification** (is this a *baby cry* vs dog / speech / silence) —
  solvable and reliable.
- **Mood / reason classification** (is the baby *hungry* vs *tired* vs *in pain*) —
  scientifically weak (trained nurses ~33%). We make it *more* robust but never
  trust it; it stays an advisory, abstaining hint.

The guiding trade for a baby device: **precision over recall**. A missed cry is
recovered on the next one a few seconds later; a false soothe (talking at a quiet
sleeping baby) erodes trust. Every layer below is tuned to drop false positives.

## Layered pipeline (defense in depth)

```
mic ─► [1] preprocess ─► [2] VAD gate ─► [3] window + classify(ensemble)
                                            │
        [4] temporal smoothing + hysteresis ┘
                                            │
        [5] SNR + abstain gates ─► [6] sustain/cooldown ─► [7] mood aggregation
```

1. **Front-end conditioning** (`detect/preprocess.py`). DC/low-cut removes rumble;
   RMS loudness-normalization makes the classifier *distance/volume invariant*
   (a cry across the room and one in the crib look the same to the model); a live
   SNR estimate vs an adaptive noise floor lets us ignore faint, far-off sounds.
   *Why:* models trained on close-mic clips degrade on quiet, noisy home audio —
   normalize the input distribution instead of hoping the model generalizes.

2. **Two-tier gating** (`detect/vad.py`). A cheap always-on VAD (Silero, energy
   fallback) decides *something is happening* before the heavier classifier runs.
   The energy fallback adapts its floor **only on quiet frames** (asymmetric:
   falls faster than it rises) so a sustained loud cry can't inflate the floor and
   gate itself out. *Why:* cuts compute ~100× when idle and suppresses silence
   false-fires.

3. **Ensemble + abstain** (`detect/ensemble.py`). Soft-vote several diverse
   members (numpy heuristic + YAMNet, optionally CLAP) and **abstain to OTHER**
   when confidence/agreement is low. *Why:* no single model is reliable across all
   rooms; disagreement is signal — when members disagree, don't act.

4. **Temporal smoothing + hysteresis** (`detect/smoothing.py`). Vote the class
   over a short sliding window; latch the cry state with separate enter/exit
   thresholds. *Why:* per-window predictions flicker (a cough spikes one frame).
   Hysteresis stops chattering on the boundary.

5. **SNR + confidence gates** (`pipeline.py`). A cry must be voiced **and** clear
   the room floor by `min_snr_db` **and** survive hysteresis before it counts.

6. **Sustain + cooldown** (`soothe/controller.py`). Require the cry to persist
   `sustain_seconds` (debounces yelps/door slams) and rate-limit actions with a
   cooldown. *Why:* turns a noisy per-frame stream into a few deliberate actions.

7. **Episode-level mood aggregation** (`detect/reason.py`
   `EpisodeReasonAggregator`). Reason hints are accumulated across the *whole* cry
   (not per frame), require enough votes and clear agreement, take a *weak,
   explicit* context prior (late-night → slightly more "tired"), keep confidence
   **capped**, and abstain to UNKNOWN when thin. Surfaced as "possible: hungry
   (~30%, low confidence)" — never as fact, caregiver always in the loop.

## Calibration & personalization
- **Ambient calibration at startup** (`pipeline.calibrate`) samples the room's
  noise floor so thresholds adapt per environment, not per global constant.
- **DoA gating (hardware)** — Reachy's 4-mic array gives direction of arrival;
  ignore sounds from the wrong direction (hallway, TV) and orient toward the crib.
- **Caregiver feedback loop (roadmap)** — let the caregiver mark a soothe as
  right/wrong to tune thresholds and, for reason, personalize per child. This is
  the only credible path to better-than-chance reason inference.

## For the trained models (when YAMNet / a cry-reason model is plugged in)
- **Augment for the home**: train/eval with added background noise, reverb,
  random gain and distance simulation, time/pitch shift — the gap between lab and
  living room is mostly distribution shift.
- **Calibrate probabilities** (temperature scaling) so the abstain thresholds
  mean something; raw softmax scores are overconfident.
- **Evaluate the right metric**: precision @ fixed low false-alarm rate, measured
  on *held-out home recordings*, not clean-clip accuracy.
- **No silent caps**: log when we abstain or gate, so misses are visible and
  tunable rather than looking like clean coverage.

## What is implemented now
Layers 1–7, ambient calibration, asymmetric VAD floor, and the abstaining
ensemble/aggregator are implemented with numpy fallbacks and covered by tests
(`tests/test_signal_detection.py`, `test_smoothing.py`, `test_controller.py`,
`test_reason.py`, `test_pipeline.py`). YAMNet/CLAP/Silero and a HF cry-reason
model slot into the same seams via the `make_*` factories without pipeline
changes. DoA gating and the feedback loop are roadmap.

## Real-model benchmark (donateacry)
The abstain thresholds in layers 3, 5, and 7 were tuned against the
**donateacry** corpus with real models plugged in (numpy heuristic +
YAMNet + the HF cry-reason model, plus a gemma-4 LLM reason hint
where applicable). Results live at
[`benchmarks/results/donateacry_real_models.json`](benchmarks/results/donateacry_real_models.json);
the corpus itself is at `benchmarks/donateacry-corpus/`. The benchmark
measures precision @ fixed low false-alarm rate (the metric that matters for
a baby device) on held-out home-style clips — see
"Evaluate the right metric" above. Re-run when changing the ensemble
composition or the abstain thresholds.
