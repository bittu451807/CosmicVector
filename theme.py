"""
theme.py
========
COSMIC VECTOR — Visual theme engine, v3 (final).

What this module gives app.py:
  1. build_base_css()        - fonts, transparent app shell, glass cards,
                                 JARVIS-style sidebar (arc-reactor header),
                                 toast + scroll-reveal CSS keyframes.
  2. inject_space_background() - a REAL WebGL 3D scene (Three.js) reparented
                                 behind the whole page: parallax starfield,
                                 drifting nebula glow, slowly tumbling
                                 asteroid meshes, a distant glowing sun --
                                 not a flat CSS gradient. This is the actual
                                 "3D live thing in the background".
  3. inject_scroll_reveal()  - one-time IntersectionObserver so cards /
                                 charts animate in as you scroll to them.
  4. section_toast(icon, title) - a small animated popup shown once per
                                 section switch (not every rerun).
  5. jarvis_header()         - the animated arc-reactor sidebar header.
"""

import streamlit as st
import streamlit.components.v1 as components

ACCENTS = {
    "briefing": "#38bdf8",
    "telemetry": "#00E5FF",
    "simulator": "#B388FF",
    "solar": "#FFD700",
    "validation": "#FF9933",
    "logs": "#7DF9C1",
}


def build_css() -> str:
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600;700&family=Roboto+Mono:wght@400;700&display=swap');

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
.main, .block-container {
    background: transparent !important;
}
.stApp {
    background: radial-gradient(ellipse 1200px 800px at 80% -10%, rgba(56,189,248,0.10), transparent 60%),
                linear-gradient(180deg, #020617 0%, #0b1330 45%, #1e1b4b 100%) !important;
    color: #F3F6FB !important;
    font-family: 'Inter', sans-serif !important;
    overflow-x: hidden;
}
[data-testid="stHeader"] { background: transparent !important; }
#cv-space-host { position: fixed; inset: 0; z-index: -1; pointer-events: none; }

/* ================= TYPOGRAPHY ================= */
h1, h2, h3, h4 { font-family: 'Orbitron', sans-serif !important; color: #F3F6FB !important;
    text-shadow: 0 2px 18px rgba(0,0,0,0.6); }
.cv-title-gradient {
    background: linear-gradient(90deg, #5eead4, #38bdf8 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent !important;
    background-size: 140% auto; text-shadow: none !important;
}
.cv-kicker { font-family:'Roboto Mono',monospace; font-size: 10.5px; letter-spacing: 3px; color: #5eead4;
    text-transform: uppercase; opacity: .85; }

/* ================= GLASS CARDS (engineered HUD chrome, not a soft AI gradient card) ================= */
.premium-card, .briefing-card {
    background: linear-gradient(150deg, rgba(6,10,24,0.74), rgba(10,14,32,0.56)) !important;
    backdrop-filter: blur(18px) saturate(140%);
    -webkit-backdrop-filter: blur(18px) saturate(140%);
    border: 1px solid rgba(148,197,255,0.14);
    border-radius: 8px; padding: 20px 22px; margin-bottom: 20px;
    box-shadow: 0 10px 34px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05);
    transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
    animation: cvFadeUp .55s ease both;
    position: relative;
}
.premium-card::before, .premium-card::after, .briefing-card::before, .briefing-card::after {
    content: ""; position: absolute; width: 12px; height: 12px; pointer-events: none; opacity: .55;
}
.premium-card::before, .briefing-card::before { top: -1px; left: -1px; border-top: 2px solid #5eead4; border-left: 2px solid #5eead4; }
.premium-card::after, .briefing-card::after { bottom: -1px; right: -1px; border-bottom: 2px solid #5eead4; border-right: 2px solid #5eead4; }
.premium-card:hover, .briefing-card:hover {
    transform: translateY(-3px);
    border-color: rgba(94,234,212,0.35);
    box-shadow: 0 16px 44px rgba(0,0,0,0.55), 0 0 22px rgba(94,234,212,0.1);
}
.premium-card:hover::before, .premium-card:hover::after,
.briefing-card:hover::before, .briefing-card:hover::after { opacity: 1; }
.premium-card small { color: #9FB2CE; letter-spacing: 1.5px; font-size: 11px; font-family:'Roboto Mono',monospace; }
.briefing-header { font-family:'Orbitron',sans-serif; color:#5eead4; margin-top:0; }
@keyframes cvFadeUp { from { opacity:0; transform: translateY(18px); filter: blur(3px); } to { opacity:1; transform: translateY(0); filter: blur(0); } }

[data-testid="stPlotlyChart"], [data-testid="stDataFrame"], iframe {
    border-radius: 10px !important;
    background: rgba(5,9,20,0.55) !important;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.10);
    padding: 6px;
    animation: cvFadeUp .5s ease both;
    transition: box-shadow .3s ease, transform .3s ease;
}
[data-testid="stPlotlyChart"]:hover { box-shadow: 0 0 28px rgba(0,229,255,0.16); }

.emergency-active {
    background: linear-gradient(135deg, rgba(255,23,68,.3), rgba(120,0,0,.4)) !important;
    backdrop-filter: blur(14px);
    border: 1px solid #FF1744 !important; border-radius: 20px; padding: 22px; text-align:center;
    box-shadow: 0 0 44px rgba(255,23,68,.55); margin-bottom: 22px;
    animation: cvAlarmFlash 1.6s ease-in-out infinite;
}
@keyframes cvAlarmFlash { 0%,100% { box-shadow: 0 0 30px rgba(255,23,68,.45); } 50% { box-shadow: 0 0 68px rgba(255,23,68,.9); } }

.eval-pass { border: 2px solid #00FF7F !important; box-shadow: 0 0 18px rgba(0,255,127,0.3) !important; }
.eval-fail { border: 2px solid #FF1744 !important; box-shadow: 0 0 18px rgba(255,23,68,0.3) !important; }

.cv-caption { color: #D8E1F0; font-size: 12.5px; line-height:1.55; margin: 6px 0 10px 0;
    border-left: 3px solid rgba(0,229,255,0.5); padding: 8px 12px;
    background: rgba(0,229,255,0.05); border-radius: 0 8px 8px 0; animation: cvFadeUp .5s ease both; }

.cv-live-chip { display:inline-flex; align-items:center; gap:7px; font-family:'Orbitron',sans-serif;
     font-size: 11px; letter-spacing: 2px; color:#00FF7F; padding: 6px 14px; border-radius: 20px;
     border: 1px solid rgba(0,255,127,0.4); background: rgba(0,20,10,0.45); backdrop-filter: blur(8px); }
.cv-live-dot { width:7px; height:7px; border-radius:50%; background:#00FF7F; box-shadow:0 0 8px #00FF7F;
     animation: cvPulseDot 1.4s ease-in-out infinite; }
@keyframes cvPulseDot { 0%,100% { opacity:1; transform:scale(1); } 50% { opacity:.3; transform:scale(1.5); } }

/* ================= BUTTONS ================= */
.stButton > button {
    background: linear-gradient(135deg, #00E5FF 0%, #7C4DFF 55%, #FFD700 100%) !important;
    color:#040814 !important; font-family:'Orbitron',sans-serif !important; font-weight:900 !important;
    border-radius: 12px !important; border: none !important; letter-spacing: 1px;
    transition: transform .15s ease, box-shadow .15s ease;
    box-shadow: 0 4px 14px rgba(0,0,0,0.4);
}
.stButton > button:hover { transform: translateY(-2px) scale(1.02); box-shadow: 0 8px 24px rgba(0,229,255,0.35); }

/* ================= PILL NAV ================= */
div[data-testid="stRadio"] > div {
    flex-direction: row !important; gap: 8px; flex-wrap: wrap;
    background: rgba(4,8,18,0.6); backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.14); border-radius: 16px; padding: 9px; margin-bottom: 4px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.35);
}
div[data-testid="stRadio"] label {
    background: transparent; border-radius: 11px; padding: 9px 18px !important;
    font-family:'Orbitron',sans-serif; font-size: 12.5px; transition: all .2s ease; border: 1px solid transparent;
}
div[data-testid="stRadio"] label:hover { background: rgba(0,229,255,0.1); border-color: rgba(0,229,255,0.3); }
div[data-testid="stRadio"] label[data-checked="true"] {
    background: linear-gradient(135deg, rgba(0,229,255,0.25), rgba(179,136,255,0.2)) !important;
    border-color: rgba(0,229,255,0.55) !important; box-shadow: 0 0 16px rgba(0,229,255,0.28);
}

.cv-section { animation: cvFadeUp .45s ease both; }

/* ================= HERO BANNERS ================= */
.cv-hero {
    display:flex; align-items:center; gap: 22px; padding: 22px 26px; margin-bottom: 22px;
    border-radius: 20px; background: linear-gradient(120deg, rgba(4,8,18,0.72), rgba(4,8,18,0.42));
    backdrop-filter: blur(16px); border: 1px solid rgba(255,255,255,0.14);
    box-shadow: 0 12px 36px rgba(0,0,0,0.45); animation: cvFadeUp .5s ease both;
    position: relative; overflow: hidden;
}
.cv-hero::after {
    content:""; position:absolute; top:-40%; right:-10%; width: 260px; height:260px; border-radius:50%;
    background: radial-gradient(circle, var(--cv-accent, #00E5FF) 0%, transparent 70%); opacity:.2; filter: blur(4px);
}
.cv-hero-icon { flex-shrink:0; width:64px; height:64px; }
.cv-hero-text h2 { margin:0 0 4px 0; font-size: 1.5rem; }
.cv-hero-text p { margin:0; color:#D8E1F0; font-size: 13.5px; max-width: 680px; line-height:1.5; }
.cv-hero-bar { height:3px; width:64px; border-radius:3px; margin-top:10px; background: linear-gradient(90deg, var(--cv-accent, #00E5FF), transparent); }

@keyframes cvOrbitSpin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
@keyframes cvFloatY { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }
@keyframes cvPulseGlow { 0%,100% { opacity:.7; } 50% { opacity:1; } }
.cv-spin { animation: cvOrbitSpin 12s linear infinite; transform-origin: 50% 50%; }
.cv-spin-slow { animation: cvOrbitSpin 22s linear infinite; transform-origin: 50% 50%; }
.cv-float { animation: cvFloatY 3s ease-in-out infinite; }
.cv-glow { animation: cvPulseGlow 2.2s ease-in-out infinite; }

/* ================= JARVIS SIDEBAR ================= */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(3,6,16,0.92), rgba(6,10,24,0.96)) !important;
    backdrop-filter: blur(18px);
    border-right: 1px solid rgba(0,229,255,0.18);
    box-shadow: inset -1px 0 0 rgba(0,229,255,0.06);
}
[data-testid="stSidebar"]::before {
    content:""; position:absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg, transparent, #00E5FF, transparent);
    animation: cvScan 3s linear infinite;
}
@keyframes cvScan { 0% { transform: translateY(0); opacity:.3; } 50% { opacity:1; } 100% { transform: translateY(600px); opacity:.3; } }

.cv-reactor-wrap { display:flex; flex-direction:column; align-items:center; padding: 6px 0 14px 0; }
.cv-reactor-ring1 { animation: cvOrbitSpin 6s linear infinite; transform-origin: 50% 50%; }
.cv-reactor-ring2 { animation: cvOrbitSpin 9s linear infinite reverse; transform-origin: 50% 50%; }
.cv-reactor-core { animation: cvPulseGlow 1.8s ease-in-out infinite; }
.cv-reactor-label { font-family:'Orbitron',sans-serif; letter-spacing:3px; font-size:13px; color:#00E5FF;
    margin-top: 6px; text-shadow: 0 0 10px rgba(0,229,255,0.6); }
.cv-reactor-sub { font-family:'Roboto Mono',monospace; font-size: 9.5px; color:#5EEAD4; letter-spacing:1.5px; opacity:.8; }

/* ================= TOAST (section-switch popup) ================= */
#cv-toast-host { position: fixed; top: 22px; left: 50%; transform: translateX(-50%) translateY(-30px);
    z-index: 999999; opacity: 0; pointer-events:none; transition: all .45s cubic-bezier(.2,.9,.25,1); }
#cv-toast-host.show { opacity: 1; transform: translateX(-50%) translateY(0); }
.cv-toast-card { display:flex; align-items:center; gap:12px; padding: 12px 22px; border-radius: 14px;
    background: linear-gradient(120deg, rgba(4,10,24,0.92), rgba(10,16,36,0.85));
    border: 1px solid rgba(0,229,255,0.4); backdrop-filter: blur(14px);
    box-shadow: 0 12px 34px rgba(0,0,0,0.5), 0 0 24px rgba(0,229,255,0.25);
    font-family:'Orbitron',sans-serif; color:#F3F6FB; font-size: 13px; letter-spacing: 1px; }

/* ================= SCROLL REVEAL ================= */
.cv-reveal-target { opacity: 0; transform: translateY(24px) scale(.985); transition: opacity .6s ease, transform .6s ease; }
.cv-reveal-target.cv-in-view { opacity: 1; transform: translateY(0) scale(1); }

::-webkit-scrollbar { width: 10px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
::-webkit-scrollbar-thumb { background: rgba(0,229,255,0.3); border-radius: 6px; }
</style>
"""


def inject_space_bg():
    """
    Real WebGL 3D background: parallax starfield across 3 depth layers, a slow
    tumbling debris/asteroid field, soft nebula glow sprites, and a distant
    glowing sun -- reparented out of the component iframe into the actual page
    so it's a true fixed full-viewport backdrop (not a flat CSS gradient).
    Idempotent: safe to call every rerun, it re-uses the same host + renderer
    state via a guard flag on the parent window instead of rebuilding it.
    """
    html = """
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script>
(function() {
  function boot() {
    try {
      const win = window.parent;
      const doc = win.document;
      if (win.__cvSpaceBooted) return;
      win.__cvSpaceBooted = true;

      let host = doc.getElementById('cv-space-host');
      if (!host) { host = doc.createElement('div'); host.id = 'cv-space-host'; doc.body.prepend(host); }
      const canvas = doc.createElement('canvas');
      canvas.style.width = '100%'; canvas.style.height = '100%'; canvas.style.display = 'block';
      host.appendChild(canvas);

      const THREE = win.__cvTHREE || window.THREE;
      const W = win.innerWidth, H = win.innerHeight;

      const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
      renderer.setPixelRatio(Math.min(win.devicePixelRatio || 1, 2));
      renderer.setSize(W, H);

      const scene = new THREE.Scene();
      scene.fog = new THREE.FogExp2(0x03060f, 0.00028);
      const camera = new THREE.PerspectiveCamera(60, W / H, 1, 6000);
      camera.position.set(0, 0, 260);

      function starLayer(count, spread, size, color, opacity) {
        const geo = new THREE.BufferGeometry();
        const pos = new Float32Array(count * 3);
        for (let i = 0; i < count; i++) {
          pos[i*3]   = (Math.random() - 0.5) * spread;
          pos[i*3+1] = (Math.random() - 0.5) * spread;
          pos[i*3+2] = (Math.random() - 0.5) * spread;
        }
        geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        const mat = new THREE.PointsMaterial({ color, size, sizeAttenuation: true, transparent: true, opacity });
        return new THREE.Points(geo, mat);
      }
      const layerFar = starLayer(1400, 3000, 1.4, 0xffffff, 0.75);
      const layerMid = starLayer(500, 2000, 2.2, 0x8fd6ff, 0.85);
      const layerNear = starLayer(140, 1200, 3.2, 0xffe9b3, 0.9);
      scene.add(layerFar, layerMid, layerNear);

      function glowSprite(hex, size, opacity) {
        const c = doc.createElement('canvas'); c.width = c.height = 256;
        const ctx = c.getContext('2d');
        const g = ctx.createRadialGradient(128,128,0,128,128,128);
        g.addColorStop(0, hex + 'FF'); g.addColorStop(0.4, hex + '66'); g.addColorStop(1, hex + '00');
        ctx.fillStyle = g; ctx.fillRect(0,0,256,256);
        const tex = new THREE.CanvasTexture(c);
        const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, opacity, depthWrite: false });
        const spr = new THREE.Sprite(mat); spr.scale.set(size, size, 1); return spr;
      }
      const nebula1 = glowSprite('#7C4DFF', 900, 0.10); nebula1.position.set(-500, 200, -900);
      const nebula2 = glowSprite('#00C2E0', 800, 0.09); nebula2.position.set(600, -250, -1100);
      scene.add(nebula1, nebula2);

      const sun = glowSprite('#FFE9A8', 260, 0.55); sun.position.set(420, 260, -800);
      const sunCore = glowSprite('#FFFFFF', 60, 0.9); sunCore.position.copy(sun.position);
      scene.add(sun, sunCore);
      scene.add(new THREE.PointLight(0xfff2cc, 1.2, 0, 0).translateX(420));
      scene.add(new THREE.AmbientLight(0x223355, 0.9));

      const asteroidMat = new THREE.MeshStandardMaterial({ color: 0x4b5563, roughness: 0.9, metalness: 0.2 });
      const asteroids = [];
      for (let i = 0; i < 10; i++) {
        const geo = new THREE.IcosahedronGeometry(4 + Math.random() * 7, 0);
        const pos = geo.attributes.position;
        for (let v = 0; v < pos.count; v++) {
          const jitter = 1 + (Math.random() - 0.5) * 0.35;
          pos.setXYZ(v, pos.getX(v)*jitter, pos.getY(v)*jitter, pos.getZ(v)*jitter);
        }
        geo.computeVertexNormals();
        const rock = new THREE.Mesh(geo, asteroidMat);
        rock.position.set((Math.random()-0.5)*500, (Math.random()-0.5)*300, -100 - Math.random()*400);
        rock.userData.spin = { x: (Math.random()-0.5)*0.006, y: (Math.random()-0.5)*0.006 };
        rock.userData.drift = (Math.random()-0.5)*0.04;
        scene.add(rock); asteroids.push(rock);
      }

      let t = 0;
      function animate() {
        t += 0.0035;
        layerFar.rotation.y = t * 0.15;
        layerMid.rotation.y = t * 0.3;
        layerNear.rotation.y = t * 0.5;
        camera.position.x = Math.sin(t * 0.4) * 22;
        camera.position.y = Math.cos(t * 0.3) * 12;
        camera.lookAt(0, 0, -300);
        asteroids.forEach(r => {
          r.rotation.x += r.userData.spin.x; r.rotation.y += r.userData.spin.y;
          r.position.x += r.userData.drift;
          if (r.position.x > 300) r.position.x = -300;
          if (r.position.x < -300) r.position.x = 300;
        });
        renderer.render(scene, camera);
        win.requestAnimationFrame(animate);
      }
      animate();

      win.addEventListener('resize', () => {
        const w = win.innerWidth, h = win.innerHeight;
        camera.aspect = w / h; camera.updateProjectionMatrix();
        renderer.setSize(w, h);
      });
    } catch (err) {
      console.warn('Space background unavailable, CSS gradient fallback stays active:', err);
    }
  }
  if (typeof THREE !== 'undefined') { boot(); }
  else { window.addEventListener('load', () => setTimeout(boot, 250)); }
})();
</script>
"""
    components.html(html, height=1, width=1)


def inject_scroll_reveal():
    """One-time IntersectionObserver so cards/charts gently animate in as the
    user scrolls to them, instead of the whole page just being static once
    rendered. Idempotent via a guard flag on the parent window."""
    html = """
<script>
(function() {
  try {
    const win = window.parent; const doc = win.document;
    function attach() {
      const targets = doc.querySelectorAll(
        '.premium-card, .briefing-card, [data-testid="stPlotlyChart"], [data-testid="stDataFrame"], .cv-hero'
      );
      targets.forEach(el => el.classList.add('cv-reveal-target'));
      if (!win.__cvObserver) {
        win.__cvObserver = new win.IntersectionObserver((entries) => {
          entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('cv-in-view'); });
        }, { threshold: 0.12 });
      }
      targets.forEach(el => { if (!el.classList.contains('cv-in-view')) win.__cvObserver.observe(el); });
    }
    attach();
    if (!win.__cvRevealInterval) {
      win.__cvRevealInterval = win.setInterval(attach, 900);
    }
  } catch (err) { /* no-op */ }
})();
</script>
"""
    components.html(html, height=1, width=1)


def section_toast(icon: str, title: str):
    """Shows a brief animated popup once per section change (caller is
    responsible for only invoking this when the section actually changed)."""
    html = f"""
<script>
(function() {{
  try {{
    const doc = window.parent.document;
    let host = doc.getElementById('cv-toast-host');
    if (!host) {{
      host = doc.createElement('div'); host.id = 'cv-toast-host';
      host.innerHTML = '<div class="cv-toast-card"><span style="font-size:20px;">{icon}</span><span>{title}</span></div>';
      doc.body.appendChild(host);
    }} else {{
      host.querySelector('.cv-toast-card').innerHTML = '<span style="font-size:20px;">{icon}</span><span>{title}</span>';
    }}
    requestAnimationFrame(() => host.classList.add('show'));
    clearTimeout(window.parent.__cvToastTimer);
    window.parent.__cvToastTimer = setTimeout(() => host.classList.remove('show'), 1700);
  }} catch (err) {{}}
}})();
</script>
"""
    components.html(html, height=1, width=1)


def jarvis_hud():
    return """
<div class="cv-reactor-wrap">
  <svg width="86" height="86" viewBox="0 0 86 86">
    <g class="cv-reactor-ring1"><circle cx="43" cy="43" r="38" stroke="#00E5FF" stroke-width="1.4" fill="none" stroke-dasharray="6 5" opacity=".8"/></g>
    <g class="cv-reactor-ring2"><circle cx="43" cy="43" r="30" stroke="#B388FF" stroke-width="1.2" fill="none" stroke-dasharray="3 6" opacity=".7"/></g>
    <circle cx="43" cy="43" r="19" fill="rgba(0,229,255,0.08)" stroke="#00E5FF" stroke-width="1"/>
    <circle class="cv-reactor-core" cx="43" cy="43" r="10" fill="#00E5FF"/>
    <circle cx="43" cy="43" r="4" fill="#FFFFFF"/>
  </svg>
  <div class="cv-reactor-label">J.A.R.V.I.S.</div>
  <div class="cv-reactor-sub">MISSION INTERFACE ONLINE</div>
</div>
"""


_ICONS = {
    "briefing": """<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="14" y="8" width="36" height="48" rx="3" stroke="#38bdf8" stroke-width="2" class="cv-float"/><line x1="20" y1="20" x2="44" y2="20" stroke="#38bdf8" stroke-width="2"/><line x1="20" y1="28" x2="44" y2="28" stroke="#38bdf8" stroke-width="1.5" opacity=".7"/><line x1="20" y1="36" x2="36" y2="36" stroke="#38bdf8" stroke-width="1.5" opacity=".7"/></svg>""",
    "telemetry": """<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="32" cy="32" r="6" fill="#00E5FF"/><g class="cv-spin"><ellipse cx="32" cy="32" rx="26" ry="10" stroke="#00E5FF" stroke-width="2" opacity=".7"/></g><g class="cv-spin-slow"><ellipse cx="32" cy="32" rx="26" ry="10" stroke="#FFD700" stroke-width="1.5" opacity=".5" transform="rotate(60 32 32)"/></g><circle cx="32" cy="32" r="26" stroke="#94A3B8" stroke-width="1" opacity=".25"/></svg>""",
    "simulator": """<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><g class="cv-float"><rect x="24" y="26" width="16" height="12" rx="2" fill="#B388FF"/><rect x="4" y="29" width="16" height="6" rx="1" fill="#00C2E0" opacity=".8"/><rect x="44" y="29" width="16" height="6" rx="1" fill="#00C2E0" opacity=".8"/><line x1="20" y1="32" x2="24" y2="32" stroke="#E2E8F0" stroke-width="2"/><line x1="40" y1="32" x2="44" y2="32" stroke="#E2E8F0" stroke-width="2"/><circle cx="32" cy="20" r="3" fill="#FFD700" class="cv-glow"/></g></svg>""",
    "solar": """<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="32" cy="32" r="12" fill="#FFD700" class="cv-glow"/><g class="cv-spin-slow" stroke="#FFD700" stroke-width="2" stroke-linecap="round"><line x1="32" y1="4" x2="32" y2="12"/><line x1="32" y1="52" x2="32" y2="60"/><line x1="4" y1="32" x2="12" y2="32"/><line x1="52" y1="32" x2="60" y2="32"/><line x1="12" y1="12" x2="18" y2="18"/><line x1="46" y1="46" x2="52" y2="52"/><line x1="12" y1="52" x2="18" y2="46"/><line x1="46" y1="18" x2="52" y2="12"/></g></svg>""",
    "validation": """<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="32" cy="32" r="24" stroke="#FF9933" stroke-width="1.5" opacity=".35"/><path d="M20 34 L27 26 L33 38 L40 22 L46 34" stroke="#FF9933" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round" class="cv-glow"/><circle cx="20" cy="34" r="2.4" fill="#00E5FF"/><circle cx="33" cy="38" r="2.4" fill="#00E5FF"/><circle cx="46" cy="34" r="2.4" fill="#00E5FF"/></svg>""",
    "logs": """<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><g class="cv-float"><ellipse cx="32" cy="18" rx="18" ry="7" fill="#7DF9C1" opacity=".85"/><path d="M14 18 V40 C14 44 50 44 50 40 V18" stroke="#7DF9C1" stroke-width="2" fill="none"/><ellipse cx="32" cy="29" rx="18" ry="7" stroke="#7DF9C1" stroke-width="1.5" fill="none" opacity=".6"/></g></svg>""",
}


def section_hero(kind: str, title: str, subtitle: str):
    accent = ACCENTS.get(kind, "#00E5FF")
    icon = _ICONS.get(kind, _ICONS["telemetry"])
    st.markdown(f"""
<div class="cv-hero" style="--cv-accent:{accent};">
    <div class="cv-hero-icon">{icon}</div>
    <div class="cv-hero-text">
        <h2 class="cv-title-gradient">{title}</h2>
        <p>{subtitle}</p>
        <div class="cv-hero-bar"></div>
    </div>
</div>
""", unsafe_allow_html=True)
