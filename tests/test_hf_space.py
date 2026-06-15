"""T40 — Hugging Face Space entry point.

These tests guard the contract for the HF Space at
``huggingface.co/spaces/thebajajra/peeky_reachy``:

- Top-level ``app.py`` exposes a ``demo`` Gradio Blocks instance.
- The Space has the files the runtime expects (``app.py``,
  ``requirements.txt``, ``README.md`` with HF metadata).
- ``app.py`` wires the same 3D Reachy head+assets setup as
  ``peeky_reachy.webapp.main`` so the URDF sidebar renders on Spaces too.

If the Space goes back to ``NO_APP_FILE``, the SDK metadata test is the
first thing to check.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


# -------------------- File presence --------------------


def test_top_level_app_py_exists():
    p = REPO_ROOT / "app.py"
    assert p.is_file(), (
        f"HF Space needs {p} at the repo root. Without it the Space "
        "runtime is NO_APP_FILE."
    )


def test_requirements_txt_exists_and_lists_gradio():
    p = REPO_ROOT / "requirements.txt"
    assert p.is_file(), f"Space needs {p} at the repo root"
    text = p.read_text(encoding="utf-8")
    # The Space runtime installs from this file; gradio is mandatory.
    assert re.search(r"^gradio\s*[<>=~]", text, re.MULTILINE), (
        f"{p} must pin gradio for the Space SDK to import it"
    )


def test_readme_has_hf_metadata_header():
    """HF Spaces parse a YAML front-matter at the top of README.md. Without
    ``sdk: gradio`` the runtime won't know which SDK to boot."""
    p = REPO_ROOT / "README.md"
    assert p.is_file(), f"Space needs {p} at the repo root"
    text = p.read_text(encoding="utf-8")
    # The front-matter is delimited by ``---`` lines.
    assert text.startswith("---"), (
        f"{p} must start with a YAML front-matter block delimited by '---'"
    )
    end = text.find("\n---", 3)
    assert end != -1, f"{p} front-matter is not closed with a second '---' line"
    fm = text[3:end]
    assert re.search(r"^sdk:\s*gradio\s*$", fm, re.MULTILINE), (
        f"{p} front-matter must declare 'sdk: gradio' so the Space runtime "
        "knows to boot the Gradio SDK. Current front-matter:\n" + fm
    )
    assert re.search(r"^app_file:\s*app\.py\s*$", fm, re.MULTILINE), (
        f"{p} must set 'app_file: app.py' so the runtime imports our shim "
        "(not the in-package peeky_reachy.webapp:main)."
    )


# -------------------- app.py behavior --------------------


def test_app_py_exposes_demo_blocks():
    """HF's Spaces runtime imports ``app.py`` and looks for a Gradio Blocks
    instance named ``demo`` (or ``app``) at module scope."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "peeky_app_entry", REPO_ROOT / "app.py"
    )
    mod = importlib.util.module_from_spec(spec)
    # Don't actually launch; just exercise the import + global wiring.
    # The module calls build_app() at import time which is cheap (no
    # network, no .launch). We catch SystemExit just in case.
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except SystemExit:
        pass
    assert hasattr(mod, "demo"), "app.py must expose a top-level `demo`"
    # The demo must be a Gradio Blocks (or FastAPI subclass).
    gr = pytest.importorskip("gradio")
    assert isinstance(mod.demo, gr.blocks.Blocks), (
        f"app.demo must be a Gradio Blocks instance; got {type(mod.demo)!r}"
    )


def test_app_py_wires_3d_reachy_assets():
    """The Space's `demo.launch(...)` must whitelist the vendored URDF +
    meshes via `allowed_paths`, and the head script must point at the
    Gradio static-file route for the URDF. Without this, the 3D sidebar
    returns 403 on the meshes and falls back to procedural."""
    src = (REPO_ROOT / "app.py").read_text(encoding="utf-8")
    assert "allowed_paths" in src, (
        "app.py must call demo.launch(allowed_paths=[...]) so the URDF "
        "and meshes are whitelisted for Gradio's static-file route."
    )
    assert "gradio_api/file=" in src, (
        "app.py must point the in-browser urdf-loader at "
        "/gradio_api/file=assets/reachy_mini/reachy_mini.urdf (Gradio 6's "
        "static-file route; the bare /file=... path 404s)."
    )
    assert "head=" in src, (
        "app.py must call demo.launch(head=reachy3d.head_html(...)) so the "
        "inline scene script (which dynamically loads three.js + urdf-loader) "
        "is injected. Gradio 6.18 escapes external <script src=...> tags "
        "in head= into the gradio_config JSON, so we use the inline-loader "
        "trick (R._loadScript)."
    )


# -------------------- Space runtime sanity (network) --------------------


@pytest.mark.network
def test_hf_space_api_reports_runtime():
    """Smoke-test: the Space exists and the API responds. Skipped if offline.
    When the Space is in NO_APP_FILE, this is the first place to look."""
    import httpx

    url = "https://huggingface.co/api/spaces/thebajajra/peeky_reachy"
    try:
        r = httpx.get(url, timeout=5.0)
    except httpx.HTTPError as exc:
        pytest.skip(f"HF API unreachable: {exc}")
    assert r.status_code == 200, (
        f"HF API returned {r.status_code} for {url}: {r.text[:200]}"
    )
    data = r.json()
    assert data.get("sdk") in ("gradio", "Gradio"), data
    # When the deploy is healthy, runtime.stage is "RUNNING" or "BUILDING".
    # We don't *assert* a particular value here because this test is
    # network-dependent; we just print it for visibility.
    stage = data.get("runtime", {}).get("stage", "UNKNOWN")
    print(f"\n  Space runtime.stage = {stage!r}")
