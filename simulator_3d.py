"""
simulator_3d.py
================
COSMIC VECTOR — Real-time 3D Spacecraft CG-Actuation Simulator
------------------------------------------------------------------
Old version rendered the satellite with pydeck (a *geospatial* WebGL map
library) — which is exactly why it always looked like "a map" and never felt
synced: pydeck is built for lat/lon basemaps, not free-flying spacecraft.

This version drops pydeck entirely and renders a real WebGL scene with
Three.js: a procedurally built Aditya-L1-style spacecraft (bus + two
articulated solar panels + payload boom + dish) floating in deep space,
with a starfield, a glowing sun, and a thin blue Earth limb for scale —
built to evoke the same warm sun-glow/space atmosphere as a Three.js
"sky/sun shader" scene, but as a mission-ops HUD instead of a landscape.

The AI's live CG-shift command (read from actuator_command.json) directly
drives the spacecraft's roll/translation in the scene, and the HUD overlay
shows the same numbers driving the geometry, so what you see and what the
AI decided are always the same number — that's the "sync" fix.
"""

import json
import os

import streamlit as st
import streamlit.components.v1 as components


def _read_actuator_state(path: str = "actuator_command.json"):
    shift_val = 0.0
    actuator_status = "STABLE"
    threat_class = 0
    ts = ""
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            shift_str = str(data.get("shift_mm", "0")).replace("mm", "").strip()
            shift_val = float(shift_str)
            actuator_status = data.get("command", "STABLE")
            threat_class = int(data.get("class", 0))
            ts = str(data.get("time", ""))
        except Exception:
            pass
    return shift_val, actuator_status, threat_class, ts


def show_3d_simulation(height: int = 640):
    shift_val, actuator_status, threat_class, ts = _read_actuator_state()

    danger = abs(shift_val) > 5
    caution = 0 < abs(shift_val) <= 5
    hud_color = "#FF1744" if danger else ("#FFD700" if caution else "#00FF7F")
    rotation_deg = max(-35.0, min(35.0, shift_val * 3.0))  # clamp for a believable roll
    translate_x = max(-1.2, min(1.2, shift_val * 0.08))

    html = f"""
<div id="cv-sim-root" style="position:relative; width:100%; height:{height}px; border-radius:16px;
     overflow:hidden; border:1px solid rgba(0,229,255,0.25); background:#010409;
     box-shadow: 0 8px 40px rgba(0,0,0,0.6);">

  <canvas id="cv-canvas" style="display:block; width:100%; height:100%;"></canvas>

  <div style="position:absolute; top:14px; left:14px; z-index:5; font-family:'Orbitron',sans-serif;
       background:rgba(3,7,18,0.55); backdrop-filter: blur(8px); border:1px solid rgba(255,255,255,0.12);
       border-radius:12px; padding:10px 16px; color:#F1F5F9; min-width:230px;">
    <div style="font-size:11px; letter-spacing:2px; color:#94A3B8;">ACTUATOR STATUS</div>
    <div style="font-size:16px; font-weight:900; color:{hud_color}; text-shadow:0 0 12px {hud_color};">{actuator_status}</div>
    <div style="height:1px; background:rgba(255,255,255,0.1); margin:8px 0;"></div>
    <div style="font-size:11px; color:#94A3B8;">CG OFFSET</div>
    <div style="font-size:20px; font-weight:700; color:#F1F5F9;">{shift_val:+.1f} mm</div>
    <div style="font-size:11px; color:#94A3B8; margin-top:6px;">CORRECTIVE ROLL</div>
    <div style="font-size:14px; color:#00E5FF;">{rotation_deg:+.1f}&deg;</div>
  </div>

  <div style="position:absolute; top:14px; right:14px; z-index:5; font-family:'Orbitron',sans-serif;
       background:rgba(3,7,18,0.55); backdrop-filter: blur(8px); border:1px solid rgba(255,255,255,0.12);
       border-radius:12px; padding:10px 16px; color:#F1F5F9; text-align:right; min-width:170px;">
    <div style="font-size:11px; letter-spacing:2px; color:#94A3B8;">THREAT CLASS</div>
    <div style="font-size:20px; font-weight:900; color:{hud_color};">{threat_class}</div>
    <div style="font-size:11px; color:#94A3B8; margin-top:6px;">SEVERITY</div>
    <div style="font-size:16px; color:#B388FF;">{"█" * (threat_class + 1)}{"░" * (3 - threat_class)}</div>
  </div>

  <div style="position:absolute; bottom:12px; left:14px; z-index:5; font-family:'Inter',sans-serif;
       font-size:11px; color:#64748B; letter-spacing:1px;">
    TICK: {ts if ts else '—'} &nbsp;•&nbsp; Drag to orbit &nbsp;•&nbsp; Scroll to zoom
  </div>

  <div id="cv-pulse" style="position:absolute; bottom:12px; right:16px; z-index:5; display:flex; align-items:center; gap:6px;
       font-family:'Orbitron',sans-serif; font-size:11px; color:{hud_color};">
    <span id="cv-dot" style="width:8px; height:8px; border-radius:50%; background:{hud_color}; box-shadow:0 0 8px {hud_color};"></span>
    LIVE
  </div>
</div>

<style>
  #cv-dot {{ animation: cvPulse 1.4s ease-in-out infinite; }}
  @keyframes cvPulse {{ 0%,100% {{ opacity:1; transform:scale(1); }} 50% {{ opacity:.35; transform:scale(1.4); }} }}
</style>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/0.160.0/three.min.js"></script>
<script>
(function() {{
  const root = document.getElementById('cv-sim-root');
  const canvas = document.getElementById('cv-canvas');
  const W = root.clientWidth, H = root.clientHeight;

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x010409, 0.012);

  const camera = new THREE.PerspectiveCamera(45, W / H, 0.1, 4000);
  camera.position.set(9, 4, 11);

  const renderer = new THREE.WebGLRenderer({{ canvas: canvas, antialias: true, alpha: true }});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(W, H);
  renderer.setClearColor(0x010409, 1);

  // ---------------- Starfield ----------------
  function buildStars(count, spread, size, color) {{
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {{
      const r = spread * (0.6 + Math.random() * 0.4);
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      positions[i*3]   = r * Math.sin(phi) * Math.cos(theta);
      positions[i*3+1] = r * Math.sin(phi) * Math.sin(theta);
      positions[i*3+2] = r * Math.cos(phi);
    }}
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.PointsMaterial({{ color: color, size: size, sizeAttenuation: true, transparent:true, opacity:0.9 }});
    return new THREE.Points(geo, mat);
  }}
  scene.add(buildStars(2200, 900, 1.1, 0xffffff));
  scene.add(buildStars(500, 500, 2.0, 0x99e6ff));

  // ---------------- Sun (glow sprite) ----------------
  const sunLight = new THREE.PointLight(0xfff2cc, 3.2, 0, 0);
  sunLight.position.set(60, 25, -40);
  scene.add(sunLight);
  scene.add(new THREE.AmbientLight(0x223344, 1.1));

  function glowSprite(colorHex, sizePx, opacity) {{
    const c = document.createElement('canvas'); c.width = c.height = 256;
    const ctx = c.getContext('2d');
    const g = ctx.createRadialGradient(128,128,0,128,128,128);
    g.addColorStop(0, colorHex + 'FF'); g.addColorStop(0.35, colorHex + 'AA'); g.addColorStop(1, colorHex + '00');
    ctx.fillStyle = g; ctx.fillRect(0,0,256,256);
    const tex = new THREE.CanvasTexture(c);
    const mat = new THREE.SpriteMaterial({{ map: tex, transparent:true, opacity: opacity, depthWrite:false }});
    const spr = new THREE.Sprite(mat); spr.scale.set(sizePx, sizePx, 1);
    return spr;
  }}
  const sunGlow = glowSprite('#FFD98A', 40, 0.9);
  sunGlow.position.copy(sunLight.position);
  scene.add(sunGlow);
  const sunCore = glowSprite('#FFFFFF', 8, 1.0);
  sunCore.position.copy(sunLight.position);
  scene.add(sunCore);

  // ---------------- Earth limb (scale reference) ----------------
  const earthGeo = new THREE.SphereGeometry(14, 48, 48);
  const earthMat = new THREE.MeshPhongMaterial({{ color: 0x14314f, emissive: 0x0a1a2e, shininess: 8 }});
  const earth = new THREE.Mesh(earthGeo, earthMat);
  earth.position.set(-22, -14, -30);
  scene.add(earth);
  const rim = glowSprite('#3fd3ff', 34, 0.28);
  rim.position.copy(earth.position);
  scene.add(rim);

  // ---------------- Spacecraft (procedural Aditya-L1-style bus) ----------------
  const craft = new THREE.Group();

  const busMat = new THREE.MeshStandardMaterial({{ color: 0xd8dee6, metalness: 0.6, roughness: 0.35 }});
  const goldMat = new THREE.MeshStandardMaterial({{ color: 0xc9962c, metalness: 0.7, roughness: 0.4 }});
  const panelMat = new THREE.MeshStandardMaterial({{ color: 0x0d3b66, metalness: 0.3, roughness: 0.55, emissive: 0x041022 }});
  const dishMat = new THREE.MeshStandardMaterial({{ color: 0xeef1f4, metalness: 0.5, roughness: 0.3, side: THREE.DoubleSide }});

  const bus = new THREE.Mesh(new THREE.BoxGeometry(1.6, 1.6, 2.0), busMat);
  craft.add(bus);
  const wrap = new THREE.Mesh(new THREE.BoxGeometry(1.64, 1.64, 0.6), goldMat);
  wrap.position.z = 0.55; craft.add(wrap);

  function solarWing(sign) {{
    const grp = new THREE.Group();
    const panel = new THREE.Mesh(new THREE.BoxGeometry(3.2, 0.06, 1.1), panelMat);
    panel.position.x = sign * (0.8 + 1.6);
    grp.add(panel);
    for (let i=-1; i<=1; i++) {{
      const line = new THREE.Mesh(new THREE.BoxGeometry(3.15, 0.01, 0.02), new THREE.MeshBasicMaterial({{color:0x1c5a86}}));
      line.position.set(sign*(0.8+1.6), 0.031, i*0.35);
      grp.add(line);
    }}
    const strut = new THREE.Mesh(new THREE.CylinderGeometry(0.04,0.04,1.6,8), goldMat);
    strut.rotation.z = Math.PI/2;
    strut.position.x = sign * 0.8;
    grp.add(strut);
    return grp;
  }}
  craft.add(solarWing(1));
  craft.add(solarWing(-1));

  const dish = new THREE.Mesh(new THREE.ConeGeometry(0.55, 0.35, 24, 1, true), dishMat);
  dish.rotation.x = Math.PI; dish.position.set(0, 1.05, -0.3);
  craft.add(dish);
  const boom = new THREE.Mesh(new THREE.CylinderGeometry(0.05,0.05,1.4,8), busMat);
  boom.rotation.z = Math.PI/2; boom.position.set(0,0,1.6);
  craft.add(boom);
  const payload = new THREE.Mesh(new THREE.CylinderGeometry(0.3,0.3,0.5,16), goldMat);
  payload.rotation.z = Math.PI/2; payload.position.set(0,0,2.3);
  craft.add(payload);

  // Baseline reference axis to visualise the commanded shift against
  const axisGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-4,0,0), new THREE.Vector3(4,0,0)
  ]);
  const axis = new THREE.Line(axisGeo, new THREE.LineDashedMaterial({{ color: 0x2c3e50, dashSize:0.15, gapSize:0.1 }}));
  axis.computeLineDistances();
  craft.add(axis);

  scene.add(craft);

  // ---------------- Live actuation pose ----------------
  const targetRotZ = THREE.MathUtils.degToRad({rotation_deg});
  const targetX = {translate_x};
  craft.rotation.z = 0; craft.position.x = 0;

  // ---------------- Orbit-style mouse controls (lightweight, no extra deps) ----------------
  let isDragging = false, lastX = 0, lastY = 0;
  let camTheta = 0.9, camPhi = 1.15, camRadius = 15;
  function updateCamera() {{
    camera.position.x = camRadius * Math.sin(camPhi) * Math.cos(camTheta);
    camera.position.z = camRadius * Math.sin(camPhi) * Math.sin(camTheta);
    camera.position.y = camRadius * Math.cos(camPhi);
    camera.lookAt(0,0,0);
  }}
  updateCamera();
  canvas.addEventListener('pointerdown', e => {{ isDragging = true; lastX = e.clientX; lastY = e.clientY; }});
  window.addEventListener('pointerup', () => isDragging = false);
  window.addEventListener('pointermove', e => {{
    if (!isDragging) return;
    camTheta -= (e.clientX - lastX) * 0.006;
    camPhi = Math.max(0.25, Math.min(2.7, camPhi - (e.clientY - lastY) * 0.006));
    lastX = e.clientX; lastY = e.clientY;
    updateCamera();
  }});
  canvas.addEventListener('wheel', e => {{
    camRadius = Math.max(6, Math.min(40, camRadius + e.deltaY * 0.01));
    updateCamera();
    e.preventDefault();
  }}, {{ passive: false }});

  // ---------------- Animation loop ----------------
  let t = 0;
  function animate() {{
    t += 0.016;
    craft.rotation.z += (targetRotZ - craft.rotation.z) * 0.05;
    craft.position.x += (targetX - craft.position.x) * 0.05;
    craft.rotation.y = Math.sin(t * 0.15) * 0.05;
    craft.position.y = Math.sin(t * 0.4) * 0.05;
    camTheta += 0.0009;
    updateCamera();
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  }}
  animate();

  window.addEventListener('resize', () => {{
    const w = root.clientWidth, h = root.clientHeight;
    camera.aspect = w/h; camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }});
}})();
</script>
"""
    components.html(html, height=height + 4, scrolling=False)

    st.caption(
        "🛰️ **What you're seeing:** a live WebGL twin of the Aditya-L1 bus. "
        "The AI's most recent CG-shift command is applied directly to the craft's roll "
        "(°) and lateral offset (mm→scene units) — the HUD numbers and the geometry are "
        "reading the same `actuator_command.json` tick, so the picture never drifts out of "
        "sync with the model. Drag to orbit, scroll to zoom."
    )
