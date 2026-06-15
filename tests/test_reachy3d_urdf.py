"""T36 — Real Reachy Mini URDF in the 3D sidebar (small-talk style).

These tests cover the vendored URDF + meshes and the four-state pose table
that drives the in-browser urdf-loader scene. They are pure-Python — the
in-browser three.js + urdf-loader is exercised manually (the user opens the
Gradio app and confirms).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from peeky_reachy import reachy3d
from tests.conftest import SR, baby_cry, silence


# -------------------- Vendored assets --------------------


def test_urdf_is_vendored_under_assets_reachy_mini():
    """The canonical Apache-2.0 URDF from pollen-robotics must be on disk."""
    urdf = reachy3d.URDF_PATH
    assert urdf.is_file(), f"URDF not found at {urdf}"
    # Sanity: it's XML, declares the robot name, and references at least
    # one mesh under `meshes/`.
    text = urdf.read_text(encoding="utf-8")
    assert text.startswith("<?xml"), "URDF must be an XML document"
    assert '<robot' in text, "URDF must declare <robot>"
    assert 'name="' in text
    assert re.search(r'filename="meshes/[^"]+\.stl"', text), "URDF must reference STL meshes"


def test_urdf_uses_canonical_joint_names():
    """The animation contract depends on these specific joint names."""
    text = reachy3d.URDF_PATH.read_text(encoding="utf-8")
    for joint_name in [
        reachy3d.URDF_JOINT_PITCH,    # "head_frame"
        reachy3d.URDF_JOINT_YAW,      # "yaw_body"
        reachy3d.URDF_JOINT_L_ANT,    # "left_antenna"
        reachy3d.URDF_JOINT_R_ANT,    # "right_antenna"
    ]:
        assert f'name="{joint_name}"' in text, (
            f"URDF must declare a joint named {joint_name!r}; "
            "the JS scene's setJointValue calls would no-op without it"
        )


def test_all_urdf_meshes_are_vendored():
    """Every mesh referenced by the URDF must be present on disk (no
    runtime fetch of remote assets — the HF Space must be self-contained)."""
    text = reachy3d.URDF_PATH.read_text(encoding="utf-8")
    referenced = set(re.findall(r'filename="meshes/([^"]+\.stl)"', text))
    assert referenced, "URDF must reference at least one mesh"
    on_disk = {p.name for p in (reachy3d.REACHY_MINI_DIR / "meshes").iterdir()}
    missing = referenced - on_disk
    assert not missing, f"URDF references meshes not on disk: {sorted(missing)}"


def test_urdf_total_mesh_count_sane():
    """We expect ~40 visual+structural meshes (screws / hardware skipped
    intentionally to keep the bundle small)."""
    text = reachy3d.URDF_PATH.read_text(encoding="utf-8")
    referenced = set(re.findall(r'filename="meshes/([^"]+\.stl)"', text))
    # Canonical pollen-robotics URDF ships 40 distinct meshes; allow ±2 slack.
    assert 38 <= len(referenced) <= 45, (
        f"expected ~40 meshes; got {len(referenced)}. "
        "Did the upstream URDF change shape?"
    )


def test_attribution_documents_urdf_source():
    """Both ATTRIBUTION.md files must credit pollen-robotics + Apache-2.0."""
    for path in [reachy3d.REACHY_MINI_DIR / "ATTRIBUTION.md",
                 reachy3d.REACHY_3D_DIR / "ATTRIBUTION.md"]:
        assert path.is_file(), f"missing {path}"
        text = path.read_text(encoding="utf-8")
        assert "pollen-robotics" in text, f"{path} must credit pollen-robotics"
        assert "Apache-2.0" in text, f"{path} must mention the Apache-2.0 license"
        assert "urdf-loader" in text, f"{path} must credit urdf-loader"


# -------------------- State poses --------------------


def test_state_poses_define_all_four_states():
    """The pose table is the source of truth for what the URDF joints do
    in each companion state; every state must be present."""
    for name in reachy3d.STATE_NAMES:
        assert name in reachy3d.STATE_POSES, f"missing pose for {name!r}"
        pose = reachy3d.STATE_POSES[name]
        for key in ("head_pitch", "body_yaw", "l_antenna", "r_antenna",
                    "sway", "nod", "shake", "eyeScale", "glow", "speed",
                    "body_color", "antenna_color"):
            assert key in pose, f"pose {name!r} missing key {key!r}"


def test_state_pose_values_are_finite_numbers_or_hex_colors():
    for name, pose in reachy3d.STATE_POSES.items():
        for key in ("head_pitch", "body_yaw", "l_antenna", "r_antenna",
                    "sway", "nod", "shake", "eyeScale", "glow", "speed"):
            v = pose[key]
            assert isinstance(v, (int, float)), f"{name}.{key} not numeric: {v!r}"
            assert -10.0 <= float(v) <= 10.0, f"{name}.{key} out of range: {v}"
        for key in ("body_color", "antenna_color"):
            v = pose[key]
            assert isinstance(v, str) and v.startswith("#") and len(v) == 7, (
                f"{name}.{key} not a #rrggbb hex color: {v!r}"
            )


def test_alert_pose_distinguishes_antennae():
    """The alert state should fan the antennae out (opposite signs) — this
    is the visual 'I heard something!' signal."""
    p = reachy3d.STATE_POSES["alert"]
    assert p["l_antenna"] > 0.0 and p["r_antenna"] < 0.0, (
        f"alert should have L>0, R<0 antenna; got L={p['l_antenna']}, R={p['r_antenna']}"
    )
    # shake is the rapid head-shake visual; must be non-zero in alert.
    assert p["shake"] > 0.0


def test_comfort_pose_nods_more_than_idle():
    p_idle = reachy3d.STATE_POSES["idle"]
    p_comfort = reachy3d.STATE_POSES["comfort"]
    assert p_comfort["nod"] > p_idle["nod"], (
        "comfort state must nod more than idle to read as 'soothing'"
    )


# -------------------- JS bridge (head_html) --------------------


def test_head_html_default_uses_urdf_loader():
    """When the URDF is vendored, head_html must load the urdf-loader CDN
    and inject the URDF URL into the scene."""
    html = reachy3d.head_html()
    assert "urdf-loader" in html, "head_html must reference the urdf-loader CDN"
    assert "reachy_mini.urdf" in html, "head_html must point at the vendored URDF"
    # Gradio 6's static-file route is /gradio_api/file=<path>; the loader
    # must fetch the URDF from this route (the dev server is configured with
    # `allowed_paths=[<repo>/assets]`).
    assert "gradio_api/file=" in html, (
        "head_html must point at /gradio_api/file=... so the URDF resolves "
        "under Gradio 6's static-file route"
    )
    # Loader mode is embedded in the scene script as a JSON string
    assert '"urdf"' in html or "'urdf'" in html


def test_head_html_procedural_fallback_emits_no_urdf_loader():
    """Forcing procedural mode must NOT load the urdf-loader CDN
    (saves bandwidth + avoids a broken promise if it's blocked)."""
    html = reachy3d.head_html(loader_mode="procedural", urdf_url=None)
    assert "urdf-loader" not in html, "procedural fallback must not load urdf-loader"
    # The four state names must still appear in the scene config.
    for name in reachy3d.STATE_NAMES:
        assert name in html, f"procedural scene must mention {name!r}"


def test_head_html_injects_pose_table():
    """STATE_POSES must be embedded in the head script so the JS scene
    can lerp joint values per state."""
    html = reachy3d.head_html()
    for name in reachy3d.STATE_NAMES:
        # Each state should appear as a JSON object key
        assert f'"{name}"' in html, f"head_html missing state pose for {name!r}"


def test_head_html_uses_canonical_joint_names():
    """The JS scene must call setJointValue on the same joint names
    the URDF declares — this is the cross-language contract."""
    html = reachy_reachy_check = reachy3d.head_html()
    assert reachy3d.URDF_JOINT_PITCH in html
    assert reachy3d.URDF_JOINT_YAW in html
    assert reachy3d.URDF_JOINT_L_ANT in html
    assert reachy3d.URDF_JOINT_R_ANT in html


# -------------------- Pipeline-state integration --------------------


def test_reachy_state_from_run_unchanged():
    """T36 must not break the existing pipeline→state mapping."""
    # Cry in the timeline → alert
    timeline = [[0.0, "baby_cry", 0.9, 12.0, True]]
    assert reachy3d.reachy_state_from_run(timeline, []) == "alert"
    # Soothe event → comfort
    assert reachy3d.reachy_state_from_run([], ["event"]) == "comfort"
    # Voiced non-cry → listening
    timeline = [[0.0, "speech", 0.5, 5.0, True]]
    assert reachy3d.reachy_state_from_run(timeline, []) == "listening"
    # Empty → idle
    assert reachy3d.reachy_state_from_run([], []) == "idle"


# -------------------- Loader-mode selector --------------------


def test_loader_mode_default_is_urdf_when_vendored():
    """When the vendored URDF is present, the default loader mode is URDF.
    This guards against accidentally shipping a code change that reverts
    to procedural-only."""
    if reachy3d.URDF_PATH.is_file():
        assert reachy3d._loader_mode() == "urdf"
    else:
        assert reachy3d._loader_mode() == "procedural"
