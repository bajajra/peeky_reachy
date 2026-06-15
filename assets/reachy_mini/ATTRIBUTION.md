# Reachy Mini 3D sidebar — assets & attribution

The Peeky web UI renders a 3D companion in the left sidebar that reacts to the
detection pipeline (`idle` / `listening` / `alert` / `comfort`).

## What's here

- `states.json` — animation presets (motion amplitudes, glow, colors, poll
  interval). Loaded by `peeky_reachy/reachy3d.py` and injected into the
  three.js scene. Edit this to retune the look/feel; no code change needed.
- `ATTRIBUTION.md` — this file.

## Geometry: real Reachy Mini URDF (vendored)

The figure is now the **actual Reachy Mini robot** loaded via the standard
three.js [`urdf-loader`](https://github.com/gkjohnson/urdf-loaders) (MIT,
CDN-loaded) from the vendored Apache-2.0 URDF + STL meshes under
`../reachy_mini/`.

Source (Apache-2.0):
- `pollen-robotics/reachy-mini-desktop-app` →
  `src/assets/robot-3d/reachy-mini.urdf` + `src/assets/robot-3d/meshes/*.stl`

The vendored copy lives at:
- `peeky_reachy/../assets/reachy_mini/reachy_mini.urdf`
- `peeky_reachy/../assets/reachy_mini/meshes/*.stl` (41 visual + structural
  STL files, ~17 MB)

The `peeky_reachy/reachy3d.py` builder loads the URDF at runtime in the
browser via CDN-hosted `urdf-loader`, then maps the four companion states
(`idle` / `listening` / `alert` / `comfort`) to URDF joint poses:

- `left_antenna`, `right_antenna` — antenna waggle
- `yaw_body` — body rotation (alert shake)
- `head_frame` — pitch/yaw nod

## Asset serving

Gradio 6 doesn't expose a public mount point for arbitrary `assets/` paths,
so the loader is given an absolute path to the URDF (which `urdf-loader` will
fetch from the Gradio dev server's `file=` route) and a relative `meshes/`
prefix. For production (HF Space, Docker) we serve `assets/reachy_mini/`
via `gradio`'s `gr.themes` / `gr.Blocks(allowed_paths=...)` mechanism so the
URDF can resolve its mesh references.

The loader transparently falls back to the procedural figure if the URDF
fails to load (e.g. off-grid asset host, no CDN access) — see
`reachy3d._loader_mode()`.

## References

- Reachy Mini URDF + meshes (Apache-2.0):
  <https://github.com/pollen-robotics/reachy-mini-desktop-app>
  — vendored into `../reachy_mini/` on 2026-06-15.
- `urdf-loader` (MIT, CDN): <https://github.com/gkjohnson/urdf-loaders>
- State-bridge pattern: inspired by `Gaurav-Gosain/small-talk`.

three.js itself is loaded from a CDN at runtime (MIT) and is not vendored.
