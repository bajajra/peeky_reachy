---
title: Peeky
emoji: 🐣
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.18.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: AI baby/pet monitor + soothing companion on Reachy Mini
---

# Peeky — AI Baby / Pet Monitor (Reachy Mini)

Peeky is a soothing companion for babies and pets. It runs as a Gradio
app, listens continuously through your browser's microphone, classifies
each sound (cry / dog / speech / silence), and auto-soothes on a
sustained cry in a **caregiver's cloned voice** (or a fallback track)
with a gentle robot motion. A 3D Reachy Mini in the sidebar reacts in
real time.

> **Safety:** Peeky is a *soothing companion*, not a medical, SIDS, or
> safety monitor. Cry-reason guesses are advisory and low-confidence.
> Always keep a caregiver in the loop.

## How the Space talks to your hardware

The Space runs the **Gradio client** only. The heavy models (voice
clone, cry classification, gemma-4 reason hint) live on **your LAN GPU
boxes** so baby audio never leaves your network.

Set the following **Space secrets** (`Settings → Variables and secrets`)
to point the Space at your boxes:

| Secret | Example | Purpose |
|---|---|---|
| `PEEKY_CRY_URL` | `https://turing.lan:8081` | Cry classification (`owlgebra-ai/babycry`) |
| `PEEKY_VOICE_CLONE_URL` | `https://spark.lan:8081` | Caregiver voice clone (VoxCPM2) |
| `PEEKY_GEMMA_URL` | `https://turing.lan:8082` | Optional cry-reason hint (gemma-4) |

The easiest way to expose the boxes publicly is **Tailscale Funnel**
(`tailscale funnel 8081` on turing, etc.) — gives you a stable
`https://<node>.ts.net:8081` URL with no home-router port-forwarding.

If the secrets are unset, the Space runs against the local numpy
fallbacks so the UI still works end-to-end (no cloned voice, no gemma
hints, but the live monitor and soothing flow still function).

## Run it locally

```bash
git clone https://github.com/bajajra/peeky_reachy
cd peeky_reachy
pip install -e .[web]
python app.py     # or: peeky-web
```

## Vendored assets

The 3D Reachy Mini sidebar ships the **canonical Apache-2.0 URDF and
STL meshes** from
[`pollen-robotics/reachy-mini-desktop-app`](https://github.com/pollen-robotics/reachy-mini-desktop-app),
vendored under `assets/reachy_mini/`. The Space serves them at
`/gradio_api/file=assets/...` so the in-browser
[`urdf-loader`](https://github.com/gkjohnson/urdf-loaders) (MIT, CDN)
can resolve the meshes.

## License

Apache-2.0. The vendored Reachy Mini URDF + meshes are © Pollen Robotics
under the same license (see `assets/reachy_3d/ATTRIBUTION.md`).
