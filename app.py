"""Hugging Face Space entry point for Peeky.

HF Spaces run the file named ``app.py`` (or whatever ``sdk: gradio`` points
at). This thin shim imports the actual Gradio Blocks from the installed
``peeky_reachy`` package and launches it.

Why not just point HF at ``peeky_reachy.webapp:main``? The Gradio Spaces
runtime expects a top-level ``app.py`` whose return value (or whose
``demo``/``app`` global) is a Gradio Blocks/FastAPI app. We expose a
``demo`` global so HF can also pick it up via
``gr.mount_gradio_app`` patterns, and we still call ``.launch()`` for
local dev parity.

LAN / cloud architecture:
- The Space runs the **client** (Gradio UI + WebRTC mic capture + a
  Pipeline). Heavy models stay on your LAN boxes (turing / spark). The
  Space's :7860 talks to those services over HTTPS via the URLs the
  user sets as Space secrets.
- If the secrets are unset, the Space degrades to the local numpy
  fallback (so it still runs end-to-end, just without the cloned voice
  / gemma reason hint).
"""

from __future__ import annotations

import os

# Surface the HF Space secrets into PEEKY_* env vars before any peeky_reachy
# module reads Config.from_env().
for _name in (
    "PEEKY_CRY_URL",
    "PEEKY_VOICE_CLONE_URL",
    "PEEKY_GEMMA_URL",
    "PEEKY_HF_TOKEN",
    "PEEKY_ASSETS_DIR",
    "PEEKY_ENROLLMENT_DIR",
    "PEEKY_USE_REMOTE_CRY",
    "PEEKY_USE_GEMMA_REASON",
):
    _val = os.environ.get(_name)
    if _val:
        os.environ.setdefault(_name, _val)

from peeky_reachy import reachy3d  # noqa: E402
from peeky_reachy.webapp import build_app  # noqa: E402

# Mirror the `allowed_paths` + `head` wiring from peeky_reachy.webapp.main
# so the 3D Reachy sidebar can serve the vendored URDF + meshes on the HF
# Space too. On Spaces the runtime imports this file and uses `demo`
# directly (skipping the __main__ branch), so the launch kwargs are baked
# into the demo instance via the same call.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_URDF_URL = "/gradio_api/file=assets/reachy_mini/reachy_mini.urdf"

# ``demo`` is the symbol HF's Spaces runtime looks for when the entry
# is structured as "import demo; demo.launch()". We build it with the
# same head+css wiring as the local CLI so the 3D sidebar renders.
demo = build_app()


def _launch():
    """Launch with the 3D companion head script + whitelisted assets.

    Idempotent: re-using an already-launched Blocks returns early.
    """
    if getattr(demo, "_launched", False):
        return
    demo.launch(
        server_name=os.environ.get("PEEKY_WEB_HOST", "0.0.0.0"),
        server_port=int(os.environ.get("PEEKY_WEB_PORT", "7860")),
        css=reachy3d.SIDEBAR_CSS,
        head=reachy3d.head_html(urdf_url=_URDF_URL),
        allowed_paths=[os.path.join(_REPO_ROOT, "assets")],
    )


if __name__ == "__main__":
    # Local dev parity: ``python app.py`` runs the same as the CLI
    # ``peeky-web`` script. On HF Spaces the runtime imports this file
    # and uses ``demo`` directly, so this branch is skipped there.
    _launch()
