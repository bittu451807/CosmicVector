"""
theme.py — Cosmic Vector visual system.

Visual language adapted from a React/Tailwind landing-page spec into
Streamlit's CSS/JS injection model (Streamlit can't run React components,
so the *aesthetic* — fonts, liquid-glass cards, glow, scroll-progress
parallax, staggered reveal — is reproduced here instead of ported literally):

- Fonts: Dancing Script (brand wordmark), Instrument Serif italic (hero
  heading + quote-style captions), Inter (body / UI text).
- "Liquid-glass" card treatment: low-alpha blurred panels with a soft
  gradient hairline border, instead of flat bordered boxes.
- Text-glow on the hero title, button-glow on primary actions.
- A scroll-progress-driven parallax background (lerp-smoothed, translate3d,
  will-change: transform) layered under the existing animated starfield.
- Scroll-reveal now staggers siblings in, instead of all popping at once.
"""

import streamlit as st
import streamlit.components.v1 as components

# ----------------------------------------------------------------------
# Palette (single source of truth — change here, it changes everywhere)
# ----------------------------------------------------------------------
BG_BASE = "#05060f"
BG_PANEL = "rgba(13, 16, 30, 0.35)"   # lower alpha -> true "liquid glass", not a flat card
TEAL = "#5eead4"
CYAN = "#38bdf8"
VIOLET = "#a78bfa"
AMBER = "#fbbf24"
ROSE = "#fb7185"
TEXT = "#eef2f7"
MUTED = "#94a3b8"
GOOD = "#34d399"
WARN = "#f59e0b"
BAD = "#fb4d4d"


def build_css() -> str:
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Dancing+Script:wght@400;500;600;700&family=Instrument+Serif:ital@0;1&family=Inter:wght@300;400;500;600;700;800;900&family=Orbitron:wght@400;600;700;900&family=Roboto+Mono:wght@400;500;700&display=swap');

    html, body, .stApp {{
        background: {BG_BASE} !important;
        color: {TEXT} !important;
        font-family: 'Inter', sans-serif !important;
        overflow-x: hidden;
    }}

    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
        background: transparent !important;
    }}
    [data-testid="stHeader"] {{ background: rgba(5,6,15,0.35) !important; backdrop-filter: blur(8px); }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(9,11,24,0.86), rgba(5,6,15,0.92)) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(94,234,212,0.15);
    }}

    h1, h2, h3, h4 {{ font-family: 'Orbitron', sans-serif !important; color: {TEXT} !important; }}

    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-thumb {{ background: linear-gradient(180deg, {TEAL}, {VIOLET}); border-radius: 4px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}

    /* -------- Brand wordmark (Dancing Script) -------- */
    .cv-brand-script {{
        font-family: 'Dancing Script', cursive; font-weight: 700;
        color: {TEXT}; letter-spacing: 0.5px;
    }}

    /* -------- Hero title: Instrument Serif italic + text-glow -------- */
    .cv-title-gradient {{
        font-family: 'Instrument Serif', serif; font-style: italic;
        background: linear-gradient(90deg, {TEAL} 0%, {CYAN} 35%, {VIOLET} 68%, {AMBER} 100%);
        -webkit-background-clip: text; background-clip: text; color: transparent !important;
        font-weight: 400; letter-spacing: 0.5px;
    }}
    .cv-text-glow {{
        text-shadow: 0 0 40px rgba(94,234,212,0.35), 0 0 80px rgba(167,139,250,0.2), 0 0 120px rgba(255,255,255,0.08);
    }}

    /* -------- Quote-style captions (Instrument Serif italic) -------- */
    .cv-quote {{
        font-family: 'Instrument Serif', serif; font-style: italic;
        font-size: 1.15rem; color: {TEXT}; line-height: 1.6;
    }}

    .cv-live-chip {{
        display: inline-flex; align-items: center; gap: 8px; margin-top: 10px;
        background: rgba(94, 234, 212, 0.08); border: 1px solid rgba(94, 234, 212, 0.35);
        color: {TEAL}; font-family: 'Roboto Mono', monospace; font-size: 11px; letter-spacing: 1.5px;
        padding: 6px 14px; border-radius: 20px;
    }}
    .cv-live-dot {{ width: 8px; height: 8px; border-radius: 50%; background: {TEAL}; box-shadow: 0 0 8px {TEAL}; animation: cv-pulse 1.4s ease-in-out infinite; }}
    @keyframes cv-pulse {{ 0%,100% {{ opacity: 1; transform: scale(1); }} 50% {{ opacity: .4; transform: scale(1.4); }} }}

    /* -------- Liquid-glass panel treatment -------- */
    .premium-card, .briefing-card, .cv-section-hero {{
        background: {BG_PANEL} !important;
        background-blend-mode: luminosity;
        backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
        border: none !important;
        box-shadow: inset 0 1px 1px rgba(255,255,255,0.1), 0 14px 34px rgba(0,0,0,0.5);
        position: relative; overflow: hidden;
        border-radius: 16px;
    }}
    .premium-card::before, .briefing-card::before, .cv-section-hero::before {{
        content: ''; position: absolute; inset: 0; border-radius: inherit; padding: 1.4px;
        background: linear-gradient(180deg,
            rgba(94,234,212,0.5) 0%, rgba(167,139,250,0.18) 25%,
            rgba(255,255,255,0) 45%, rgba(255,255,255,0) 60%,
            rgba(251,191,36,0.18) 82%, rgba(94,234,212,0.5) 100%);
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor; mask-composite: exclude;
        pointer-events: none;
    }}
    .premium-card {{ padding: 18px 20px; margin-bottom: 18px; }}
    .premium-card small {{ color: {MUTED}; letter-spacing: 1px; font-size: 11px; text-transform: uppercase; font-family: 'Roboto Mono', monospace; }}
    .premium-card h3 {{ margin: 6px 0 0 0; color: {TEXT} !important; }}

    .briefing-card {{ padding: 24px 26px; margin-bottom: 20px; }}
    .briefing-header {{ color: {AMBER} !important; font-size: 1.3rem; margin-top: 0; font-family: 'Instrument Serif', serif; font-style: italic; font-weight: 400; }}

    .cv-caption {{
        font-size: 12.5px; color: {MUTED}; background: rgba(148,163,184,0.05);
        backdrop-filter: blur(6px);
        border-left: 2px solid {CYAN}; padding: 8px 12px; border-radius: 6px; margin: 6px 0 10px 0;
    }}

    .emergency-active {{
        background: linear-gradient(135deg, rgba(251,77,77,.20), rgba(120,0,0,.26)) !important;
        backdrop-filter: blur(10px);
        border: 1px solid {BAD} !important; border-radius: 18px; padding: 22px; text-align: center;
        box-shadow: 0 0 40px rgba(251,77,77,.5); margin-bottom: 22px; animation: alarm-flash 1.6s ease-in-out infinite;
    }}
    @keyframes alarm-flash {{ 0%,100% {{ box-shadow: 0 0 28px rgba(251,77,77,.4); }} 50% {{ box-shadow: 0 0 58px rgba(251,77,77,.8); }} }}

    .eval-pass {{ border: 2px solid {GOOD} !important; box-shadow: 0 0 15px rgba(52,211,153,0.25) !important; }}
    .eval-fail {{ border: 2px solid {BAD} !important; box-shadow: 0 0 15px rgba(251,77,77,0.25) !important; }}

    /* -------- Buttons: button-glow -------- */
    .stButton > button {{
        background: linear-gradient(135deg, {TEAL} 0%, {CYAN} 55%, {VIOLET} 100%) !important;
        color:#020617 !important; font-family:'Inter',sans-serif !important; font-weight:700 !important;
        letter-spacing: 0.3px;
        border: none !important; border-radius:999px !important;
        box-shadow: 0 0 20px rgba(94,234,212,0.3), 0 0 40px rgba(167,139,250,0.12);
        transition: transform .15s ease, box-shadow .15s ease;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 0 26px rgba(94,234,212,0.45), 0 0 52px rgba(167,139,250,0.2);
    }}

    [data-testid="stRadio"] label {{ font-family: 'Inter', sans-serif; }}

    /* -------- Section hero: liquid-glass + Instrument Serif heading -------- */
    .cv-section-hero {{
        padding: 22px 26px; margin: 6px 0 22px 0;
        background: linear-gradient(120deg, rgba(94,234,212,0.08), rgba(167,139,250,0.06) 55%, rgba(251,191,36,0.05));
    }}
    .cv-section-hero h2 {{ margin: 0 0 4px 0; font-size: 1.9rem; font-family: 'Instrument Serif', serif; font-style: italic; font-weight: 400; }}
    .cv-section-hero p {{ margin: 0; color: {MUTED}; font-size: 0.95rem; font-family: 'Inter', sans-serif; }}

    /* -------- Scroll-reveal: hidden -> visible, staggered by JS (see inject_scroll_reveal) -------- */
    .cv-reveal {{
        opacity: 0; transform: translate3d(0, 28px, 0); will-change: transform, opacity;
        transition: opacity .7s ease, transform .7s cubic-bezier(.2,.7,.3,1);
    }}
    .cv-reveal.cv-visible {{ opacity: 1; transform: translate3d(0,0,0); }}

    /* -------- Professional chart framing: every Plotly chart gets the same
       liquid-glass treatment as the cards, so graphs read as part of the
       dashboard instead of bare plots dropped on a black background -------- */
    [data-testid="stPlotlyChart"] {{
        background: {BG_PANEL} !important;
        backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
        border-radius: 16px; padding: 10px 6px 4px 6px;
        box-shadow: inset 0 1px 1px rgba(255,255,255,0.08), 0 10px 28px rgba(0,0,0,0.45);
        border: 1px solid rgba(148,163,184,0.10);
        margin-bottom: 14px;
    }}
    [data-testid="stDataFrame"] {{
        border-radius: 14px !important; overflow: hidden;
        border: 1px solid rgba(148,163,184,0.15) !important;
    }}
    </style>
    """


def jarvis_hud() -> str:
    """Sidebar HUD block. (Function name kept as jarvis_hud so app.py's
    existing `theme.jarvis_hud()` call doesn't need to change.)"""
    return f"""
    <div style="text-align:center; padding: 4px 0 2px 0;">
        <div class="cv-brand-script" style="font-size: 1.9rem; line-height:1;">Cosmic Interface</div>
        <p style="color:{MUTED}; font-size:10.5px; letter-spacing:1.5px; margin:6px 0 0 0; text-transform:uppercase;">
            Autonomous Spacecraft Defense Link
        </p>
    </div>
    """


def section_hero(section_id: str, title: str, subtitle: str) -> None:
    accents = {
        "briefing": (TEAL, "📖"), "telemetry": (CYAN, "📊"), "simulator": (VIOLET, "🪐"),
        "solar": (AMBER, "🌐"), "validation": (ROSE, "🧠"), "logs": (MUTED, "🗄️"),
    }
    color, icon = accents.get(section_id, (TEAL, "🛰️"))
    st.markdown(
        f"""
        <div class="cv-section-hero cv-reveal">
            <h2 style="color:{TEXT} !important;">{icon} {title}</h2>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_toast(icon: str, name: str) -> None:
    try:
        st.toast(f"{name}", icon=icon)
    except Exception:
        pass


# ----------------------------------------------------------------------
# Live animated space background + scroll-progress parallax.
#
# Two things layered on one <canvas>, pinned behind the Streamlit app:
#  1. Time-based drift (nebula blobs + twinkling stars + shooting star) —
#     unchanged from before, gives the "always alive" feel.
#  2. Scroll-progress parallax — a soft aurora band whose vertical offset
#     is driven by how far the page has scrolled (lerp-smoothed toward the
#     target each frame, like the reference spec's rainbow/cloud layers),
#     so the background visibly *responds* to scrolling instead of just
#     sitting there. Pure 2D canvas on purpose: it must never compete for
#     GPU/video-memory with the gesture simulator's WebGL context, which
#     was the actual cause of the earlier black-screen crash.
# ----------------------------------------------------------------------
def inject_space_bg() -> None:
    components.html(
        """
        <script>
        (function() {
            const doc = window.parent.document;
            if (doc.getElementById('cv-space-bg')) return; // already injected, don't duplicate on rerun

            const canvas = doc.createElement('canvas');
            canvas.id = 'cv-space-bg';
            canvas.style.position = 'fixed';
            canvas.style.top = 0; canvas.style.left = 0;
            canvas.style.width = '100vw'; canvas.style.height = '100vh';
            canvas.style.zIndex = -1;
            canvas.style.pointerEvents = 'none';
            doc.body.prepend(canvas);

            const ctx = canvas.getContext('2d');
            let w, h, dpr;
            function resize() {
                dpr = Math.min(window.parent.devicePixelRatio || 1, 1.5);
                w = canvas.width = window.parent.innerWidth * dpr;
                h = canvas.height = window.parent.innerHeight * dpr;
                canvas.style.width = window.parent.innerWidth + 'px';
                canvas.style.height = window.parent.innerHeight + 'px';
            }
            resize();
            window.parent.addEventListener('resize', resize);

            const STAR_COLORS = ['#eef2f7', '#5eead4', '#a78bfa', '#38bdf8'];
            // -- true 3D starfield: each star lives in (x,y,z) space centered on the
            // viewer and drifts slowly toward the camera; projected to screen with a
            // perspective divide (screen = focal * (world / z)). This is what gives
            // real depth -- near stars sweep faster and grow bigger, far stars barely
            // move -- rather than the flat 2D parallax trick used before. Kept on a
            // 2D canvas (not WebGL) deliberately, so it never competes for GPU/video
            // memory with the gesture simulator's WebGL context.
            const STAR_COUNT = 320;
            const FOCAL = 300;
            const STAR_SPEED = 0.35; // slow, ambient drift -- not a "warp speed" effect
            const stars3d = [];
            function spawnStar(randomZ) {
                return {
                    x: (Math.random() - 0.5) * 2000,
                    y: (Math.random() - 0.5) * 2000,
                    z: randomZ ? Math.random() * 1000 + 1 : 1000,
                    color: STAR_COLORS[Math.floor(Math.random() * STAR_COLORS.length)]
                };
            }
            for (let i = 0; i < STAR_COUNT; i++) stars3d.push(spawnStar(true));

            const nebulae = [
                { cx: 0.2, cy: 0.25, r: 0.55, color: 'rgba(94,234,212,0.10)' },
                { cx: 0.8, cy: 0.15, r: 0.5, color: 'rgba(167,139,250,0.09)' },
                { cx: 0.55, cy: 0.75, r: 0.6, color: 'rgba(251,191,36,0.06)' },
                { cx: 0.1, cy: 0.85, r: 0.4, color: 'rgba(56,189,248,0.08)' }
            ];

            // scroll-progress-driven aurora band (lerp-smoothed)
            let auroraY = 0, auroraYTarget = 0;
            function getScrollProgress() {
                const doc2 = window.parent.document;
                const scroller = doc2.querySelector('section.main') || doc2.scrollingElement || doc2.documentElement;
                const scrollTop = scroller.scrollTop || window.parent.pageYOffset || 0;
                const scrollHeight = Math.max(1, (scroller.scrollHeight || doc2.body.scrollHeight) - window.parent.innerHeight);
                return Math.max(0, Math.min(1, scrollTop / scrollHeight));
            }

            let shooting = null;
            let t = 0;

            function maybeSpawnShootingStar() {
                if (!shooting && Math.random() < 0.004) {
                    shooting = {
                        x: Math.random() * w, y: Math.random() * h * 0.4,
                        vx: (Math.random() * 6 + 8), vy: (Math.random() * 3 + 3),
                        life: 1.0
                    };
                }
            }

            function draw() {
                t += 0.002;
                ctx.clearRect(0, 0, w, h);

                const g = ctx.createLinearGradient(0, 0, w, h);
                g.addColorStop(0, '#070912');
                g.addColorStop(0.5, '#0b0e1c');
                g.addColorStop(1, '#050610');
                ctx.fillStyle = g;
                ctx.fillRect(0, 0, w, h);

                // -- scroll-progress aurora band: target moves from +120 -> -160 (in css px * dpr) as you scroll, lerp factor 0.06 --
                const progress = getScrollProgress();
                auroraYTarget = (120 - progress * 280) * dpr;
                auroraY += (auroraYTarget - auroraY) * 0.06;
                const auroraGrad = ctx.createLinearGradient(0, auroraY, 0, auroraY + h * 0.5);
                auroraGrad.addColorStop(0, 'rgba(94,234,212,0.12)');
                auroraGrad.addColorStop(0.5, 'rgba(167,139,250,0.07)');
                auroraGrad.addColorStop(1, 'rgba(251,191,36,0.0)');
                ctx.fillStyle = auroraGrad;
                ctx.fillRect(0, auroraY, w, h * 0.5);

                nebulae.forEach((n, i) => {
                    const cx = (n.cx + Math.sin(t + i) * 0.02) * w;
                    const cy = (n.cy + Math.cos(t * 0.8 + i) * 0.02) * h;
                    const rad = n.r * Math.max(w, h);
                    const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, rad);
                    grad.addColorStop(0, n.color);
                    grad.addColorStop(1, 'rgba(0,0,0,0)');
                    ctx.fillStyle = grad;
                    ctx.fillRect(0, 0, w, h);
                });

                const cx0 = w / 2, cy0 = h / 2;
                for (let i = 0; i < stars3d.length; i++) {
                    const s = stars3d[i];
                    s.z -= STAR_SPEED;
                    if (s.z <= 1) { stars3d[i] = spawnStar(false); continue; }

                    const scale = (FOCAL * dpr) / s.z;
                    const px2 = cx0 + s.x * scale * 0.5;
                    const py2 = cy0 + s.y * scale * 0.5;
                    if (px2 < 0 || px2 > w || py2 < 0 || py2 > h) continue;

                    const depthAlpha = Math.max(0, Math.min(1, 1 - s.z / 1000));
                    const r = Math.max(0.3, scale * 0.9);
                    ctx.globalAlpha = 0.25 + depthAlpha * 0.75;
                    ctx.fillStyle = s.color;
                    ctx.beginPath();
                    ctx.arc(px2, py2, r, 0, Math.PI * 2);
                    ctx.fill();
                }
                ctx.globalAlpha = 1;

                maybeSpawnShootingStar();
                if (shooting) {
                    ctx.strokeStyle = 'rgba(238,242,247,' + shooting.life + ')';
                    ctx.lineWidth = 2 * dpr;
                    ctx.beginPath();
                    ctx.moveTo(shooting.x, shooting.y);
                    ctx.lineTo(shooting.x - shooting.vx * 8, shooting.y - shooting.vy * 8);
                    ctx.stroke();
                    shooting.x += shooting.vx * dpr;
                    shooting.y += shooting.vy * dpr;
                    shooting.life -= 0.02;
                    if (shooting.life <= 0 || shooting.x > w || shooting.y > h) shooting = null;
                }

                requestAnimationFrame(draw);
            }
            draw();
        })();
        </script>
        """,
        height=0,
    )


def inject_scroll_reveal() -> None:
    """Reveal .cv-reveal elements as they scroll into view, staggering
    siblings the way the reference spec staggers its mobile-menu links
    (~75ms apart) instead of every card popping in at once."""
    components.html(
        """
        <script>
        (function() {
            const doc = window.parent.document;
            function reveal() {
                const vh = window.parent.innerHeight;
                let staggerIndex = 0;
                doc.querySelectorAll('.cv-reveal:not(.cv-visible)').forEach(el => {
                    const rect = el.getBoundingClientRect();
                    if (rect.top < vh * 0.92) {
                        const delay = Math.min(staggerIndex, 6) * 75;
                        el.style.transitionDelay = delay + 'ms';
                        el.classList.add('cv-visible');
                        staggerIndex++;
                    }
                });
            }
            reveal();
            const container = doc.querySelector('section.main') || doc.body;
            container.addEventListener('scroll', reveal, { passive: true });
            window.parent.addEventListener('scroll', reveal, { passive: true });
            const mo = new MutationObserver(reveal);
            mo.observe(doc.body, { childList: true, subtree: true });
            setInterval(reveal, 400); // Streamlit reruns swap DOM nodes; keep checking
        })();
        </script>
        """,
        height=0,
    )
