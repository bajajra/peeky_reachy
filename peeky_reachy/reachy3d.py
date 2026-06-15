"""3D Reachy companion for the Gradio sidebar (T29 + T36).

A small, pipeline-driven three.js figure that reacts to the detection pipeline
with four states: ``idle`` / ``listening`` / ``alert`` / ``comfort``.

Bridge design (kept deliberately simple so it runs anywhere):
- ``webapp.analyze`` maps a pipeline run (``on_window`` timeline + ``on_soothe``
  events) to one state string via :func:`reachy_state_from_run`.
- That string is written into a hidden ``gr.Textbox`` (``elem_id`` =
  :data:`STATE_ELEM_ID`).
- A 200 ms JS poll reads the textbox value and calls
  ``window.peekyReachy.setState(state)``, which lerps the scene toward the
  preset defined in ``assets/reachy_3d/states.json``.

Geometry (T36):
- **Real Reachy Mini URDF** (vendored from pollen-robotics, Apache-2.0) loaded
  via the standard three.js ``urdf-loader`` (MIT, CDN) at runtime.
- Falls back to a procedural figure if the URDF can't be loaded (offline
  asset host, blocked CDN, etc.) — see :data:`LOADER_MODE` and
  :func:`_loader_mode`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

STATE_ELEM_ID = "peeky-reachy-state"
CANVAS_ELEM_ID = "peeky-reachy-canvas"
STATUS_ELEM_ID = "peeky-reachy-status"
SIDEBAR_ELEM_ID = "peeky-sidebar"

STATE_NAMES = ("idle", "listening", "alert", "comfort")
DEFAULT_STATE = "idle"
DEFAULT_POLL_MS = 200

# Mirrors SoundEvent.BABY_CRY.value (avoids importing detect/ here).
_CRY_EVENT = "baby_cry"

REPO_ROOT = Path(__file__).resolve().parent.parent
REACHY_3D_DIR = REPO_ROOT / "assets" / "reachy_3d"
REACHY_MINI_DIR = REPO_ROOT / "assets" / "reachy_mini"
_STATES_PATH = REACHY_3D_DIR / "states.json"
URDF_PATH = REACHY_MINI_DIR / "reachy_mini.urdf"

# Which loader the JS scene should use. 'urdf' is the default; 'procedural' is
# the fallback when the URDF or its meshes aren't shippable. Exposed for
# testing — set to 'procedural' if the vendored URDF is absent.
LOADER_MODE = "urdf" if URDF_PATH.is_file() else "procedural"

# URDF joint names (canonical pollen-robotics names) we animate per state.
# See `assets/reachy_mini/ATTRIBUTION.md` for the kinematic tree.
URDF_JOINT_PITCH = "head_frame"      # the visible head pitch axis
URDF_JOINT_YAW = "yaw_body"          # the body-relative yaw
URDF_JOINT_L_ANT = "left_antenna"
URDF_JOINT_R_ANT = "right_antenna"

# State → joint-pose table. The JS scene lerps between these; values are
# radians, antennae are the (limited) rotation of the antenna joints.
STATE_POSES = {
    "idle": {
        "head_pitch": 0.0,
        "body_yaw": 0.0,
        "l_antenna": 0.0,
        "r_antenna": 0.0,
        "sway": 0.10,
        "nod": 0.05,
        "shake": 0.0,
        "eyeScale": 1.0,
        "glow": 0.30,
        "speed": 0.8,
        "body_color": "#6f8bbf",
        "antenna_color": "#9ad0ff",
    },
    "listening": {
        "head_pitch": 0.12,
        "body_yaw": 0.20,
        "l_antenna": 0.30,
        "r_antenna": 0.30,
        "sway": 0.30,
        "nod": 0.12,
        "shake": 0.0,
        "eyeScale": 1.18,
        "glow": 0.55,
        "speed": 1.4,
        "body_color": "#41bfb0",
        "antenna_color": "#6fffe9",
    },
    "alert": {
        "head_pitch": 0.0,
        "body_yaw": 0.0,
        "l_antenna": 0.55,
        "r_antenna": -0.55,
        "sway": 0.0,
        "nod": 0.0,
        "shake": 0.6,
        "eyeScale": 1.32,
        "glow": 0.95,
        "speed": 3.2,
        "body_color": "#e0584f",
        "antenna_color": "#ff9d6b",
    },
    "comfort": {
        "head_pitch": -0.08,
        "body_yaw": 0.0,
        "l_antenna": 0.10,
        "r_antenna": -0.10,
        "sway": 0.20,
        "nod": 0.42,
        "shake": 0.0,
        "eyeScale": 0.92,
        "glow": 0.70,
        "speed": 0.6,
        "body_color": "#e8bd66",
        "antenna_color": "#ffe9b0",
    },
}

# Fallback used if the vendored JSON is missing (e.g. non-editable install).
_DEFAULT_STATES = {
    "poll_interval_ms": DEFAULT_POLL_MS,
    "default": DEFAULT_STATE,
    "states": {
        "idle": {"label": "Idle", "body": "#6f8bbf", "emissive": "#16243f",
                 "antenna": "#9ad0ff", "sway": 0.16, "nod": 0.0, "shake": 0.0,
                 "eyeScale": 1.0, "glow": 0.2, "speed": 0.8},
        "listening": {"label": "Listening", "body": "#41bfb0", "emissive": "#0f3f39",
                      "antenna": "#6fffe9", "sway": 0.34, "nod": 0.12, "shake": 0.0,
                      "eyeScale": 1.18, "glow": 0.55, "speed": 1.4},
        "alert": {"label": "Alert", "body": "#e0584f", "emissive": "#4a1311",
                  "antenna": "#ff9d6b", "sway": 0.0, "nod": 0.0, "shake": 0.6,
                  "eyeScale": 1.32, "glow": 0.95, "speed": 3.2},
        "comfort": {"label": "Comforting", "body": "#e8bd66", "emissive": "#4a3410",
                    "antenna": "#ffe9b0", "sway": 0.22, "nod": 0.5, "shake": 0.0,
                    "eyeScale": 0.9, "glow": 0.7, "speed": 0.6},
    },
}


def load_states() -> dict:
    """Load the vendored animation presets, falling back to the built-in copy."""
    try:
        with _STATES_PATH.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return _DEFAULT_STATES


def _loader_mode() -> str:
    """Decide which scene loader to use. Public alias of LOADER_MODE for tests."""
    return LOADER_MODE


def reachy_state_from_run(
    timeline: Sequence[Sequence],
    events: Iterable,
) -> str:
    """Map a pipeline run to a single companion state.

    ``timeline`` rows are ``[t, event_value, score, snr, voiced]`` (as built by
    ``webapp.analyze``); ``events`` is the list of soothe events.
    """
    if events:
        return "comfort"
    rows = list(timeline)
    if any(len(r) > 1 and r[1] == _CRY_EVENT for r in rows):
        return "alert"
    if any(len(r) > 4 and bool(r[4]) for r in rows):
        return "listening"
    return "idle"


SIDEBAR_CSS = f"""
#{SIDEBAR_ELEM_ID} {{
  flex: 0 0 300px !important;
  max-width: 320px !important;
  min-width: 250px !important;
}}
.peeky-reachy-icons {{ display: flex; justify-content: space-around; margin-top: 10px; font-size: 1.1em; opacity: 0.5; }}
.peeky-reachy-icons .icon {{ padding: 4px 8px; border-radius: 8px; transition: all 0.2s; }}
.peeky-reachy-icons .icon.is-active {{ opacity: 1.0; background: rgba(255,255,255,0.08); transform: scale(1.15); }}
#{CANVAS_ELEM_ID} {{
  width: 100%;
  height: 420px;
  border-radius: 14px;
  overflow: hidden;
  background: radial-gradient(circle at 50% 32%, #1b2740 0%, #0b1020 80%);
}}
#{CANVAS_ELEM_ID} canvas {{ display: block; }}
.peeky-reachy-title {{ font-weight: 600; opacity: 0.75; margin-bottom: 6px; }}
#{STATUS_ELEM_ID} {{
  text-align: center; margin-top: 8px; font-weight: 600; letter-spacing: 0.04em;
}}
.peeky-st-idle {{ color: #7fb3ff; }}
.peeky-st-listening {{ color: #5ad1c4; }}
.peeky-st-alert {{ color: #ff6b6b; }}
.peeky-st-comfort {{ color: #ffd479; }}
.peeky-reachy-hint {{ font-size: 0.78em; opacity: 0.6; margin-top: 6px; text-align: center; }}
.peeky-reachy-icons {{
  display: flex; justify-content: space-around; margin-top: 10px;
  font-size: 1.1em; opacity: 0.5; transition: opacity 0.2s;
}}
.peeky-reachy-icons .icon {{ padding: 4px 8px; border-radius: 8px; }}
.peeky-reachy-icons .icon.is-active {{
  opacity: 1.0; background: rgba(255,255,255,0.08); transform: scale(1.15);
}}
.peeky-reachy-icons .icon[data-state="idle"]      .ic {{ color: #7fb3ff; }}
.peeky-reachy-icons .icon[data-state="listening"] .ic {{ color: #5ad1c4; }}
.peeky-reachy-icons .icon[data-state="alert"]     .ic {{ color: #ff6b6b; }}
.peeky-reachy-icons .icon[data-state="comfort"]   .ic {{ color: #ffd479; }}
"""


def sidebar_html() -> str:
    """HTML for the sidebar canvas container + live status caption."""
    return f"""
<div id="peeky-reachy-sidebar-inner">
  <div class="peeky-reachy-title">🤖 Reachy</div>
  <div id="{CANVAS_ELEM_ID}"></div>
  <div id="{STATUS_ELEM_ID}" class="peeky-st-idle">Idle</div>
  <div class="peeky-reachy-icons" id="peeky-reachy-icons">
    <span class="icon is-active" data-state="idle"><span class="ic">💤</span> idle</span>
    <span class="icon" data-state="listening"><span class="ic">👂</span> listening</span>
    <span class="icon" data-state="alert"><span class="ic">🚨</span> alert</span>
    <span class="icon" data-state="comfort"><span class="ic">🤗</span> comfort</span>
  </div>
  <div class="peeky-reachy-hint">Reacts to the monitor: idle · listening · alert · comforting</div>
</div>
"""


_THREE_CDN = "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"
_URDF_LOADER_CDN = "https://cdn.jsdelivr.net/npm/urdf-loader@0.12.6/dist/umd/URDFLoader.js"


# Scene + state-bridge. Placeholders are substituted in head_html() so we don't
# fight Python str.format over the JS braces.
_SCENE_JS = """
window.peekyReachy = window.peekyReachy || {};
(function (R) {
  R.config = __STATES_JSON__;
  R.poses = __POSES_JSON__;
  R.loaderMode = __LOADER_MODE__;
  R.urdfUrl = __URDF_URL__;
  R.urdfJointPitch = __JOINT_PITCH__;
  R.urdfJointYaw = __JOINT_YAW__;
  R.urdfJointLAnt = __JOINT_L_ANT__;
  R.urdfJointRAnt = __JOINT_R_ANT__;
  R.states = R.config.states || {};
  R.pollMs = R.config.poll_interval_ms || 200;
  R.current = R.config.default || "idle";
  R.target = R.current;
  R._ready = false;
  R._polling = false;

  R.setState = function (name) {
    if (!R.poses[name]) return;
    R.target = name;
    var s = document.getElementById("__STATUS_ID__");
    if (s) {
      s.textContent = (R.states[name] && R.states[name].label) || name;
      s.className = "peeky-st-" + name;
    }
    var icons = document.querySelectorAll("#peeky-reachy-icons .icon");
    icons.forEach(function (el) {
      if (el.getAttribute("data-state") === name) el.classList.add("is-active");
      else el.classList.remove("is-active");
    });
  };

  R._init = function () {
    if (R._ready) return;
    var THREE = window.THREE;
    var URDFLoader = window.URDFLoader && window.URDFLoader.URDFLoader;
    var host = document.getElementById("__CANVAS_ID__");
    if (!THREE || !host) return;
    R._ready = true;

    var w = host.clientWidth || 280, h = host.clientHeight || 340;
    var scene = new THREE.Scene();
    var camera = new THREE.PerspectiveCamera(40, w / h, 0.1, 100);
    camera.position.set(0.0, 0.45, 1.5);
    camera.lookAt(0, 0.1, 0);
    var renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    host.innerHTML = "";
    host.appendChild(renderer.domElement);

    var key = new THREE.DirectionalLight(0xffffff, 1.15); key.position.set(2, 3, 4); scene.add(key);
    scene.add(new THREE.AmbientLight(0x8899bb, 0.7));

    var root = new THREE.Group(); scene.add(root);
    var robot = null;          // URDF root
    var joints = {};           // joint name -> THREE.Object3D
    var materialCache = {};    // material name -> THREE.Material (for tinting)
    var usedLoader = R.loaderMode;

    if (R.loaderMode === "urdf" && URDFLoader && R.urdfUrl) {
      try {
        var loader = new URDFLoader();
        loader.loadMeshCb = function (mesh, file, done) {
          // Light tint so the figure matches the per-state color
          mesh.castShadow = false;
          mesh.receiveShadow = false;
          done(mesh);
        };
        loader.load(R.urdfUrl, function (urdf) {
          robot = urdf;
          root.add(robot);
          // Look up joint objects for animation
          (urdf.joints || []).forEach(function (j) {
            if (j.name) joints[j.name] = j;
          });
          // Tighten the framing: URDF is in meters; camera placed for 0.4m wide
          var box = new THREE.Box3().setFromObject(robot);
          var size = box.getSize(new THREE.Vector3());
          var maxDim = Math.max(size.x, size.y, size.z);
          camera.position.set(0, size.y * 0.45, maxDim * 2.6);
          camera.lookAt(0, size.y * 0.40, 0);
        }, undefined, function (err) {
          console.warn("[peeky-reachy] URDF load failed, falling back:", err);
          if (!robot) R._buildProcedural(THREE, root);
        });
      } catch (e) {
        console.warn("[peeky-reachy] URDF init failed, falling back:", e);
        R._buildProcedural(THREE, root);
      }
    } else {
      R._buildProcedural(THREE, root);
    }

    var clock = new THREE.Clock();
    var cur = Object.assign({}, R.poses[R.current]);
    var lastUpdate = 0;
    var updateInterval = 80; // ms between joint writes (smoother than rAF)

    function step(name, k) {
      var t = R.poses[name]; if (!t) return;
      for (var key in t) {
        if (typeof t[key] === "number") {
          cur[key] = lerp(cur[key] != null ? cur[key] : t[key], t[key], k);
        }
      }
    }

    function applyPose() {
      if (!robot) return;
      if (joints[R.urdfJointPitch] && cur.head_pitch != null) {
        joints[R.urdfJointPitch].setJointValue(cur.head_pitch);
      }
      if (joints[R.urdfJointYaw] && cur.body_yaw != null) {
        joints[R.urdfJointYaw].setJointValue(cur.body_yaw);
      }
      if (joints[R.urdfJointLAnt] && cur.l_antenna != null) {
        joints[R.urdfJointLAnt].setJointValue(cur.l_antenna);
      }
      if (joints[R.urdfJointRAnt] && cur.r_antenna != null) {
        joints[R.urdfJointRAnt].setJointValue(cur.r_antenna);
      }
    }

    function lerp(a, b, t) { return a + (b - a) * t; }

    function animate() {
      requestAnimationFrame(animate);
      var dt = clock.getDelta();
      var el = clock.getElapsedTime();
      step(R.target, Math.min(1, dt * 4));
      var sp = cur.speed || 1;
      // Subtle global sway (small amplitude so we don't fight the URDF poses)
      var swayAmp = (cur.sway || 0) * 0.05;
      root.rotation.y = Math.sin(el * sp) * swayAmp + (cur.body_yaw || 0) * (cur.shake ? Math.sin(el * 22) * 0.3 : 0);
      if (robot) {
        var now = performance.now();
        if (now - lastUpdate > updateInterval) {
          applyPose();
          lastUpdate = now;
        }
      }
      renderer.render(scene, camera);
    }
    animate();

    window.addEventListener("resize", function () {
      var nw = host.clientWidth || w, nh = host.clientHeight || h;
      camera.aspect = nw / nh; camera.updateProjectionMatrix(); renderer.setSize(nw, nh);
    });

    R.setState(R.current);
    R._startPoll();
  };

  R._buildProcedural = function (THREE, root) {
    var bodyMat = new THREE.MeshStandardMaterial({ color: 0x6f8bbf, emissive: 0x16243f, roughness: 0.5, metalness: 0.2 });
    var body = new THREE.Mesh(new THREE.CylinderGeometry(0.85, 1.0, 1.3, 40), bodyMat);
    body.position.y = -0.45; root.add(body);
    var head = new THREE.Group(); head.position.y = 0.55; root.add(head);
    var headMat = new THREE.MeshStandardMaterial({ color: 0xf2f4fa, emissive: 0x223044, roughness: 0.35, metalness: 0.1 });
    var skull = new THREE.Mesh(new THREE.SphereGeometry(0.7, 40, 32), headMat);
    skull.scale.set(1.0, 0.82, 0.9); head.add(skull);
    var eyeMat = new THREE.MeshStandardMaterial({ color: 0x10141c, emissive: 0x3a6dff, emissiveIntensity: 0.4 });
    function eye(x) {
      var e = new THREE.Mesh(new THREE.SphereGeometry(0.12, 24, 24), eyeMat);
      e.position.set(x, 0.02, 0.62); head.add(e); return e;
    }
    var eyeL = eye(-0.24), eyeR = eye(0.24);
    var antMat = new THREE.MeshStandardMaterial({ color: 0x9ad0ff, emissive: 0x6fffe9, emissiveIntensity: 0.6, roughness: 0.3 });
    function antenna(x) {
      var g = new THREE.Group();
      var rod = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.5, 12),
                               new THREE.MeshStandardMaterial({ color: 0x444a55 }));
      rod.position.y = 0.25; g.add(rod);
      var tip = new THREE.Mesh(new THREE.SphereGeometry(0.1, 20, 20), antMat);
      tip.position.y = 0.52; g.add(tip);
      g.position.set(x, 0.5, 0); g.rotation.z = -x * 0.3; head.add(g);
      return g;
    }
    var antL = antenna(-0.35), antR = antenna(0.35);
    R._proc = { head: head, antL: antL, antR: antR, bodyMat: bodyMat, antMat: antMat, eyeMat: eyeMat, eyeL: eyeL, eyeR: eyeR };
  };

  R._startPoll = function () {
    if (R._polling) return;
    R._polling = true;
    setInterval(function () {
      var el = document.querySelector("#__STATE_ID__ textarea, #__STATE_ID__ input");
      if (el) {
        var v = (el.value || "").trim();
        if (v && v !== R.target) R.setState(v);
      }
    }, R.pollMs);
  };

  R.boot = function () {
    var haveThree = !!window.THREE;
    var needLoader = (R.loaderMode === "urdf");
    var haveLoader = !needLoader || !!(window.URDFLoader && window.URDFLoader.URDFLoader);
    if (haveThree && haveLoader && document.getElementById("__CANVAS_ID__")) {
      R._init();
    } else {
      setTimeout(R.boot, 120);
    }
  };
})(window.peekyReachy);
"""


def head_html(states: Optional[dict] = None,
              urdf_url: Optional[str] = None,
              loader_mode: Optional[str] = None) -> str:
    """``<head>`` payload: three.js + (optional) urdf-loader CDN + scene script.

    Parameters
    ----------
    states : dict, optional
        Animation presets. Defaults to the vendored JSON.
    urdf_url : str, optional
        Absolute or relative URL of the URDF. Defaults to a Gradio-served
        path. Tests can pass ``None`` to force the procedural loader.
    loader_mode : str, optional
        ``"urdf"`` (default if URDF present) or ``"procedural"``. Tests can
        force the procedural path to verify the fallback.
    """
    cfg = states if states is not None else load_states()
    mode = loader_mode if loader_mode is not None else LOADER_MODE
    url = urdf_url if urdf_url is not None else (
        "/gradio_api/file=assets/reachy_mini/reachy_mini.urdf" if mode == "urdf" else ""
    )
    scene = (
        _SCENE_JS
        .replace("__STATES_JSON__", json.dumps(cfg))
        .replace("__POSES_JSON__", json.dumps(STATE_POSES))
        .replace("__LOADER_MODE__", json.dumps(mode))
        .replace("__URDF_URL__", json.dumps(url))
        .replace("__JOINT_PITCH__", json.dumps(URDF_JOINT_PITCH))
        .replace("__JOINT_YAW__", json.dumps(URDF_JOINT_YAW))
        .replace("__JOINT_L_ANT__", json.dumps(URDF_JOINT_L_ANT))
        .replace("__JOINT_R_ANT__", json.dumps(URDF_JOINT_R_ANT))
        .replace("__CANVAS_ID__", CANVAS_ELEM_ID)
        .replace("__STATE_ID__", STATE_ELEM_ID)
        .replace("__STATUS_ID__", STATUS_ELEM_ID)
    )
    loader_tag = (
        f'<script src="{_URDF_LOADER_CDN}"></script>\n' if mode == "urdf" else ""
    )
    return (
        f'<script src="{_THREE_CDN}"></script>\n'
        f"{loader_tag}"
        f"<script>\n{scene}\nwindow.peekyReachy.boot();\n</script>"
    )


# Re-run boot after Gradio mounts the DOM (idempotent; _init guards on _ready).
BOOT_JS = (
    "() => { if (window.peekyReachy && window.peekyReachy.boot) "
    "window.peekyReachy.boot(); }"
)
