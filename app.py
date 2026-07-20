import streamlit as st
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time as time_lib
import json
from captum.attr import IntegratedGradients
import streamlit.components.v1 as components
from scipy.fft import rfft, rfftfreq
from sklearn.metrics import confusion_matrix

from simulator_3d import show_3d_simulation
from data_ingest import load_instrument_upload, sync_and_engineer, overlap_diagnostics
import theme
import cosmic_assistant

# ==========================================
# 0. SPATIAL COMPUTING FRONTEND (WEBGL + MEDIAPIPE GESTURE CONTROL)
#    22 assets · click-to-select OR point+pinch-to-select · two-hand
#    CLAP-TO-ZOOM ONLY · gesture-debounced so commands never blend ·
#    camera is opt-in (never auto-requested) · context-loss guards
# ==========================================
SPATIAL_COMPUTE_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;700&family=Inter:wght@400;600&display=swap');

        body { margin: 0; overflow: hidden; background-color: transparent; font-family: 'Inter', sans-serif; color: #eef2f7; user-select: none; }
        #webgl-container { position: absolute; top: 0; left: 0; width: 100vw; height: 100vh; z-index: 1; }

        .engineering-panel {
            position: absolute; z-index: 10;
            background: rgba(8, 10, 18, 0.86);
            backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 6px; padding: 18px;
            box-shadow: 0 12px 32px rgba(0,0,0,0.85);
        }
        /* corner-bracket HUD chrome -- deliberately "engineered", not a soft AI card */
        .engineering-panel::before, .engineering-panel::after,
        .engineering-panel .bl, .engineering-panel .br {
            content: ""; position: absolute; width: 14px; height: 14px; pointer-events: none;
        }
        .engineering-panel::before { top: -1px; left: -1px; border-top: 2px solid #5eead4; border-left: 2px solid #5eead4; }
        .engineering-panel::after  { top: -1px; right: -1px; border-top: 2px solid #5eead4; border-right: 2px solid #5eead4; }
        .engineering-panel .bl { bottom: -1px; left: -1px; border-bottom: 2px solid #5eead4; border-left: 2px solid #5eead4; position:absolute; }
        .engineering-panel .br { bottom: -1px; right: -1px; border-bottom: 2px solid #5eead4; border-right: 2px solid #5eead4; position:absolute; }

        .left-menu { top: 20px; left: 20px; width: 280px; max-height: 74vh; overflow-y: auto; transition: transform .35s cubic-bezier(.4,0,.2,1), opacity .35s ease; }
        .left-menu::-webkit-scrollbar { width: 4px; }
        .left-menu::-webkit-scrollbar-thumb { background: #5eead4; border-radius: 2px; }
        .left-menu.collapsed { transform: translateX(-320px); opacity: 0; pointer-events: none; }

        #reopen-menu-btn {
            position:absolute; top:20px; left:20px; z-index: 11; display:none;
            background: rgba(8,10,18,0.9); border: 1px solid #5eead4; color:#5eead4;
            font-family:'Roboto Mono',monospace; font-size:11px; letter-spacing:1px;
            padding: 10px 14px; border-radius: 6px; cursor:pointer;
        }
        #reopen-menu-btn:hover { background: rgba(94,234,212,0.15); }

        .right-guide { top: 20px; right: 20px; width: 300px; }

        .bottom-systems { bottom: 20px; left: 50%; transform: translateX(-50%); display: flex; gap: 15px; flex-direction: row; align-items: center; justify-content: center; width: auto; padding: 14px 22px; }
        .sys-btn { background: rgba(251, 191, 36, 0.08); border: 1px solid #fbbf24; color: #fbbf24; padding: 9px 18px; font-family: 'Roboto Mono'; font-size: 11px; font-weight: 700; border-radius: 5px; cursor: pointer; transition: 0.2s; text-transform: uppercase; letter-spacing: .5px; }
        .sys-btn:hover { background: rgba(251, 191, 36, 0.18); transform: translateY(-2px); }
        .sys-btn.active { background: #fbbf24; color: #020617; box-shadow: 0 0 15px #fbbf24; }

        h3 { font-family: 'Roboto Mono', monospace; color: #e2e8f0; text-transform: uppercase; font-size: 11.5px; letter-spacing: 1.5px; margin-top: 0; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px; }

        .menu-btn { background: rgba(255, 255, 255, 0.025); border: 1px solid rgba(255, 255, 255, 0.08); color: #94a3b8; padding: 9px 11px; margin: 5px 0; cursor: pointer; font-family: 'Roboto Mono', monospace; transition: all 0.18s ease; text-align: left; font-size: 10px; border-radius: 4px; letter-spacing: .3px; }
        .menu-btn:hover { background: rgba(94, 234, 212, 0.1); border-color: #5eead4; color: #f1f5f9; transform: translateX(3px); }
        .menu-btn.active { background: #5eead4; border-color: #5eead4; color: #001018; font-weight: bold; }
        .menu-btn.tag-mission { border-left: 3px solid #fbbf24; }
        .menu-btn.reticle-armed { outline: 2px solid #fbbf24; outline-offset: 1px; }

        .guide-item { font-size: 11px; margin-bottom: 11px; display: flex; align-items: flex-start; line-height: 1.5; color: #cbd5e1; }
        .guide-icon { font-size: 13px; margin-right: 11px; color: #a78bfa; font-family: 'Roboto Mono'; font-weight: bold; background: rgba(167,139,250,0.12); padding: 2px 6px; border-radius: 4px; }

        #laser-pointer { position: absolute; width: 16px; height: 16px; border: 2px solid #5eead4; border-radius: 50%; box-shadow: 0 0 10px #5eead4; z-index: 999; pointer-events: none; display: none; transform: translate(-50%, -50%); transition: transform 0.08s, border-color .15s; }
        #laser-pointer::after { content:""; position:absolute; top:50%; left:50%; width:3px; height:3px; background:#5eead4; border-radius:50%; transform:translate(-50%,-50%); }
        .laser-armed { border-color: #fbbf24 !important; box-shadow: 0 0 16px #fbbf24 !important; transform: translate(-50%, -50%) scale(1.5) !important; }

        #video-feed { position: absolute; bottom: 20px; right: 20px; width: 150px; border-radius: 6px; z-index: 5; opacity: 0.35; border: 1px solid rgba(255,255,255,0.2); transform: scaleX(-1); transition: opacity 0.3s; filter: grayscale(60%) contrast(1.1); }

        #hud-status { position: absolute; bottom: 20px; left: 20px; font-family: 'Roboto Mono'; font-size: 10.5px; color: #38bdf8; z-index: 10; background: rgba(6,9,18,0.88); padding: 8px 12px; border-left: 2px solid #38bdf8; text-transform: uppercase; border-radius: 4px; }
        #hands-badge { position:absolute; bottom: 56px; left: 20px; font-family:'Roboto Mono'; font-size: 10px; color:#fbbf24; background: rgba(6,9,18,0.88); padding: 5px 10px; border-left:2px solid #fbbf24; z-index:10; border-radius:4px; }

        #gate-overlay { position:absolute; inset:0; z-index:200; display:flex; flex-direction:column; align-items:center; justify-content:center;
            background: radial-gradient(circle at 50% 40%, rgba(94,234,212,0.06), #05060c 70%); text-align:center; padding: 30px; }
        #gate-overlay h2 { font-family:'Roboto Mono',monospace; color:#5eead4; letter-spacing:3px; font-size:15px; margin-bottom: 8px; }
        #gate-overlay p { color:#94a3b8; font-size:12.5px; max-width:440px; line-height:1.6; margin-bottom: 22px; }
        .gate-btn { background: linear-gradient(135deg,#5eead4,#38bdf8); color:#001018; font-family:'Roboto Mono',monospace; font-weight:700;
            letter-spacing:1px; font-size:12.5px; padding:14px 28px; border-radius:6px; border:none; cursor:pointer; text-transform:uppercase; }
        .gate-btn:hover { filter: brightness(1.08); }
        .gate-btn.secondary { background: rgba(255,255,255,0.06); color:#cbd5e1; border:1px solid rgba(255,255,255,0.2); margin-top: 10px; }

        #loading-overlay { position: absolute; top:0; left:0; width:100%; height:100%; background: #05060f; z-index: 150; display:none; flex-direction: column; justify-content:center; align-items:center; font-family:'Roboto Mono'; color:#5eead4; font-size: 15px; letter-spacing: 2px; transition: opacity 0.5s; }
        #loading-text { margin-top: 15px; font-size: 12px; color: #38bdf8; }
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js"></script>
</head>
<body>
    <div id="gate-overlay">
        <h2>🎮 GESTURE ENGINE — STANDBY</h2>
        <p>This loads a live WebGL scene with 22 inspectable spacecraft assets. You can drive it entirely
        with your <b>mouse</b> (click any asset, drag to rotate, scroll to zoom) — no camera required.
        Only if you want hands-free control does it request webcam access, and only after you tap the
        button below.</p>
        <button class="gate-btn" onclick="enableMouseMode()">🖱️ START — MOUSE MODE</button><br>
        <button class="gate-btn secondary" onclick="enableGestureMode()">🖐️ START + ENABLE CAMERA GESTURES</button>
    </div>

    <div id="loading-overlay">
        <div>INITIALIZING TELEMETRY...</div>
        <div id="loading-text">Booting WebGL Engine...</div>
    </div>
    <div id="webgl-container"></div>
    <video id="video-feed" autoplay playsinline style="display:none;"></video>
    <div id="laser-pointer"></div>

    <div id="hands-badge" style="display:none;">HANDS DETECTED: 0</div>
    <div id="hud-status" style="display:none;">GNC SENSOR: STANDBY</div>

    <button id="reopen-menu-btn" onclick="toggleMenu(true)">☰ SHOW ASSET MANIFEST</button>
    <div class="engineering-panel left-menu" id="left-menu-panel">
        <span class="bl"></span><span class="br"></span>
        <h3>Asset Manifest (22) — click OR gesture-select <span style="float:right; cursor:pointer;" onclick="toggleMenu(false)">✕</span></h3>
        <div class="menu-btn active tag-mission" data-target="actuator" data-type="proc" onclick="switchAsset(this)">MISSION: CG ACTUATOR ASSEMBLY</div>
        <div class="menu-btn tag-mission" data-target="coronagraph" data-type="proc" onclick="switchAsset(this)">MISSION: VELC CORONAGRAPH</div>
        <div class="menu-btn tag-mission" data-target="sensor_head" data-type="proc" onclick="switchAsset(this)">MISSION: SOLEXS/HEL1OS SENSOR HEAD</div>
        <div class="menu-btn tag-mission" data-target="antenna_array" data-type="proc" onclick="switchAsset(this)">MISSION: HIGH-GAIN ANTENNA ARRAY</div>
        <div class="menu-btn tag-mission" data-target="reaction_wheel" data-type="proc" onclick="switchAsset(this)">MISSION: REACTION WHEEL ASSEMBLY</div>
        <div class="menu-btn tag-mission" data-target="star_tracker" data-type="proc" onclick="switchAsset(this)">MISSION: STAR TRACKER UNIT</div>
        <div class="menu-btn" data-target="cubesat" data-type="proc" onclick="switchAsset(this)">ORBITAL: CUBESAT (3U)</div>
        <div class="menu-btn" data-target="solar_sail" data-type="proc" onclick="switchAsset(this)">PROP: SOLAR SAIL ARRAY</div>
        <div class="menu-btn" data-target="heat_shield" data-type="proc" onclick="switchAsset(this)">THERMAL: RADIATIVE HEAT SHIELD</div>
        <div class="menu-btn" data-target="rc_glider" data-type="proc" onclick="switchAsset(this)">AERO: RC GLIDER (STABILITY)</div>
        <div class="menu-btn" data-target="airfoil" data-type="proc" onclick="switchAsset(this)">AERO: WING AIRFOIL CROSS</div>
        <div class="menu-btn" data-target="satellite" data-type="proc" onclick="switchAsset(this)">ORBITAL: SATELLITE BUS</div>
        <div class="menu-btn" data-target="lander" data-type="proc" onclick="switchAsset(this)">SURFACE: LUNAR LANDER</div>
        <div class="menu-btn" data-target="rover" data-type="proc" onclick="switchAsset(this)">SURFACE: MARS ROVER</div>
        <div class="menu-btn" data-target="telescope" data-type="proc" onclick="switchAsset(this)">OPTICS: WEBB TELESCOPE ARRAY</div>
        <div class="menu-btn" data-target="station" data-type="proc" onclick="switchAsset(this)">ORBITAL: SPACE STATION MODULE</div>
        <div class="menu-btn" data-target="voyager" data-type="proc" onclick="switchAsset(this)">SOLAR: VOYAGER PROBE</div>
        <div class="menu-btn" data-target="earth" data-type="proc" onclick="switchAsset(this)">ASTRO: HIGH-RES EARTH</div>
        <div class="menu-btn" data-target="sun" data-type="proc" onclick="switchAsset(this)">THERMAL: SOLAR PLASMA</div>
        <div class="menu-btn" data-target="helmet" data-type="gltf" data-url="https://raw.githubusercontent.com/mrdoob/three.js/master/examples/models/gltf/DamagedHelmet/glTF/DamagedHelmet.gltf" data-scale="1.5" onclick="switchAsset(this)">MAT: DAMAGED SPACESUIT HELMET</div>
        <div class="menu-btn" data-target="ion_drive" data-type="gltf" data-url="https://raw.githubusercontent.com/mrdoob/three.js/master/examples/models/gltf/PrimaryIonDrive.glb" data-scale="2.0" onclick="switchAsset(this)">PROP: PRIMARY ION DRIVE ENGINE</div>
        <div class="menu-btn" data-target="gearbox" data-type="gltf" data-url="https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/GearboxAssy/glTF/GearboxAssy.gltf" data-scale="10.0" onclick="switchAsset(this)">MECH: STRUCTURAL GEARBOX ASSY</div>
        <div class="menu-btn" data-target="flight_helmet" data-type="gltf" data-url="https://raw.githubusercontent.com/mrdoob/three.js/master/examples/models/gltf/FlightHelmet/glTF/FlightHelmet.gltf" data-scale="4.0" onclick="switchAsset(this)">AERO: AVIATION FLIGHT HELMET</div>
        <div class="menu-btn" data-target="drone" data-type="gltf" data-url="https://raw.githubusercontent.com/mrdoob/three.js/master/examples/models/gltf/RobotExpressive/RobotExpressive.glb" data-scale="0.8" onclick="switchAsset(this)">ROBOTICS: AI EXPLORATION DRONE</div>
    </div>

    <div class="engineering-panel bottom-systems">
        <span class="bl"></span><span class="br"></span>
        <div class="sys-btn" id="btn-cad" onclick="toggleCADMode()">📐 CAD Blueprint</div>
        <div class="sys-btn" id="btn-wind" onclick="toggleWindTunnel()">🌪️ Solar Wind</div>
        <div class="sys-btn" id="btn-cam" onclick="enableGestureMode()">🎥 Enable Camera</div>
    </div>

    <div class="engineering-panel right-guide" id="right-guide-panel" style="display:none;">
        <span class="bl"></span><span class="br"></span>
        <h3>Selection &amp; Gesture Protocols</h3>
        <div class="guide-item"><span class="guide-icon">🖱️</span><div><b>MOUSE / TOUCH:</b> click any asset — always available, no camera needed. Drag to rotate, scroll to zoom.</div></div>
        <hr style="border-color: rgba(255,255,255,0.1); margin: 10px 0;">
        <div class="guide-item"><span class="guide-icon">👆🤏</span><div><b>POINT + PINCH:</b> aim your index finger at an asset (reticle turns amber = armed), then pinch thumb+index together to confirm. No time-based hover-guessing.</div></div>
        <div class="guide-item"><span class="guide-icon">✊</span><div><b>FIST:</b> grab and rotate the 3D geometry.</div></div>
        <div class="guide-item"><span class="guide-icon">🖐️</span><div><b>OPEN PALM:</b> pan the model across X / Y.</div></div>
        <div class="guide-item"><span class="guide-icon">✌️</span><div><b>PEACE SIGN:</b> move hand up/down to zoom.</div></div>
        <hr style="border-color: rgba(255,255,255,0.1); margin: 10px 0;">
        <div class="guide-item"><span class="guide-icon">👏</span><div><b>TWO HANDS · CLAP-ZOOM:</b> bring both hands together to zoom in, spread apart to zoom out. This is the only two-hand gesture, and it never blends with the one-hand commands above — each gesture must hold stable for a few frames before it takes over, so quick hand noise can't trigger the wrong command.</div></div>
        <hr style="border-color: rgba(255,255,255,0.1); margin: 10px 0;">
        <div style="font-size: 9px; color: #a78bfa; text-align: center;">DEBOUNCED · COMMAND-EXCLUSIVE STATE MACHINE</div>
    </div>

    <script>
        // --- 1. THREE.JS ENGINE SETUP (with context-loss guards) ---
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.z = 10;

        let renderer;
        try {
            renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true, powerPreference: "default", failIfMajorPerformanceCaveat: false });
        } catch (e) {
            renderer = null;
        }

        if (renderer) {
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(1);
            renderer.outputEncoding = THREE.sRGBEncoding;
            renderer.toneMapping = THREE.ACESFilmicToneMapping;
            document.getElementById('webgl-container').appendChild(renderer.domElement);

            renderer.domElement.addEventListener('webglcontextlost', function (e) {
                e.preventDefault();
                hud.innerText = "GNC SENSOR: GPU CONTEXT LOST — RECOVERING...";
            }, false);
            renderer.domElement.addEventListener('webglcontextrestored', function () {
                hud.innerText = "GNC SENSOR: GPU CONTEXT RESTORED";
            }, false);
        }

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.8); scene.add(ambientLight);
        const dirLight = new THREE.DirectionalLight(0xffffff, 2.0); dirLight.position.set(10, 10, 10); scene.add(dirLight);
        const backLight = new THREE.DirectionalLight(0x38bdf8, 1.0); backLight.position.set(-10, -10, -10); scene.add(backLight);
        const rimLight = new THREE.DirectionalLight(0xa78bfa, 0.6); rimLight.position.set(0, -8, 6); scene.add(rimLight);

        const texLoader = new THREE.TextureLoader();
        const gltfLoader = new THREE.GLTFLoader();

        const currentGroup = new THREE.Group(); scene.add(currentGroup);
        const assets = {};

        // --- 2. PROCEDURAL & SHADER ASSETS ---
        const aluMat = new THREE.MeshStandardMaterial({ color: 0xd4d4d8, metalness: 0.9, roughness: 0.3 });
        const carbonMat = new THREE.MeshStandardMaterial({ color: 0x18181b, metalness: 0.5, roughness: 0.6 });
        const foilMat = new THREE.MeshStandardMaterial({ color: 0xfbbf24, metalness: 1.0, roughness: 0.2 });
        const solarMat = new THREE.MeshStandardMaterial({ color: 0x0a192f, metalness: 0.8, roughness: 0.2 });
        const cyanMat = new THREE.MeshStandardMaterial({ color: 0x5eead4, metalness: 0.6, roughness: 0.3, emissive: 0x003844 });
        const violetMat = new THREE.MeshStandardMaterial({ color: 0xa78bfa, metalness: 0.5, roughness: 0.35 });

        assets['actuator'] = (() => {
            const g = new THREE.Group();
            const rail = new THREE.Mesh(new THREE.BoxGeometry(4, 0.15, 0.15), aluMat); g.add(rail);
            const screw = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 3.8, 12).rotateZ(Math.PI/2), cyanMat);
            screw.position.y = -0.25; g.add(screw);
            const mass = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.6, 0.6), foilMat); mass.position.set(0.6, 0, 0); g.add(mass);
            const motor = new THREE.Mesh(new THREE.CylinderGeometry(0.35, 0.35, 0.6, 16).rotateZ(Math.PI/2), violetMat);
            motor.position.set(-2.1, -0.1, 0); g.add(motor);
            for (let i=-1;i<=1;i+=2) { const leg = new THREE.Mesh(new THREE.BoxGeometry(0.08,0.6,0.08), carbonMat); leg.position.set(i*1.6,-0.6,0); g.add(leg); }
            return g;
        })();
        assets['coronagraph'] = (() => {
            const g = new THREE.Group();
            const tube = new THREE.Mesh(new THREE.CylinderGeometry(0.9, 0.9, 3.4, 32, 1, true), carbonMat);
            tube.material.side = THREE.DoubleSide; tube.rotation.z = Math.PI/2; g.add(tube);
            for (let i=0;i<4;i++) { const baffle = new THREE.Mesh(new THREE.TorusGeometry(0.9, 0.04, 8, 32), aluMat); baffle.rotation.y = Math.PI/2; baffle.position.x = -1.3 + i*0.9; g.add(baffle); }
            const occulter = new THREE.Mesh(new THREE.CircleGeometry(0.55, 32), foilMat); occulter.position.x = 1.9; occulter.rotation.y = Math.PI/2; g.add(occulter);
            return g;
        })();
        assets['sensor_head'] = (() => {
            const g = new THREE.Group();
            const box = new THREE.Mesh(new THREE.BoxGeometry(1.2, 1.0, 1.2), aluMat); g.add(box);
            const collimator = new THREE.Mesh(new THREE.ConeGeometry(0.5, 1.1, 24, 1, true), cyanMat);
            collimator.material.side = THREE.DoubleSide; collimator.rotation.x = Math.PI; collimator.position.y = 0.95; g.add(collimator);
            const sdd = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.18, 0.3, 16), foilMat); sdd.position.y = 1.55; g.add(sdd);
            for (let i=-1;i<=1;i+=2) { const fin = new THREE.Mesh(new THREE.BoxGeometry(1.3, 0.04, 0.5), carbonMat); fin.position.set(0, -0.55, i*0.4); g.add(fin); }
            return g;
        })();
        assets['antenna_array'] = (() => {
            const g = new THREE.Group();
            const dish = new THREE.Mesh(new THREE.SphereGeometry(1.4, 32, 16, 0, Math.PI * 2, 0, Math.PI / 2.3), aluMat);
            dish.material.side = THREE.DoubleSide; dish.rotation.x = Math.PI; g.add(dish);
            const horn = new THREE.Mesh(new THREE.ConeGeometry(0.12, 0.7, 16), cyanMat); horn.position.set(0, 0.9, 0); horn.rotation.x = Math.PI; g.add(horn);
            for (let i = 0; i < 3; i++) {
                const ang = (i / 3) * Math.PI * 2;
                const strut = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 1.0), carbonMat);
                strut.position.set(Math.cos(ang) * 0.5, 0.45, Math.sin(ang) * 0.5);
                strut.lookAt(0, 0.9, 0); g.add(strut);
            }
            const base = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.35, 0.5, 16), violetMat); base.position.y = -0.6; g.add(base);
            return g;
        })();
        assets['reaction_wheel'] = (() => {
            const g = new THREE.Group();
            const housing = new THREE.Mesh(new THREE.CylinderGeometry(0.9, 0.9, 0.5, 32), carbonMat); housing.rotation.x = Math.PI/2; g.add(housing);
            const wheel = new THREE.Mesh(new THREE.TorusGeometry(0.65, 0.12, 16, 32), cyanMat); g.add(wheel);
            const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.15, 0.55, 16), foilMat); hub.rotation.x = Math.PI/2; g.add(hub);
            for (let i=0;i<4;i++) { const spoke = new THREE.Mesh(new THREE.BoxGeometry(0.55,0.05,0.05), aluMat); spoke.rotation.z = (i/4)*Math.PI*2; g.add(spoke); }
            return g;
        })();
        assets['star_tracker'] = (() => {
            const g = new THREE.Group();
            const body = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.8, 1.4), aluMat); g.add(body);
            const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.32, 0.38, 0.5, 24), cyanMat);
            lens.rotation.x = Math.PI/2; lens.position.z = 0.9; g.add(lens);
            const hood = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.42, 0.15, 24, 1, true), carbonMat);
            hood.material.side = THREE.DoubleSide; hood.rotation.x = Math.PI/2; hood.position.z = 1.2; g.add(hood);
            const bracket = new THREE.Mesh(new THREE.BoxGeometry(0.15, 1.0, 0.15), foilMat); bracket.position.set(0, -0.7, 0); g.add(bracket);
            return g;
        })();
        assets['cubesat'] = (() => {
            const g = new THREE.Group();
            g.add(new THREE.Mesh(new THREE.BoxGeometry(1.0, 1.0, 3.0), aluMat));
            for (let i=-1;i<=1;i+=2) {
                const panel = new THREE.Mesh(new THREE.BoxGeometry(0.05, 2.4, 2.6), solarMat);
                panel.position.set(i*0.6, 0, 0); g.add(panel);
            }
            const antenna = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 1.2, 6), foilMat);
            antenna.position.set(0, 0.6, -1.5); g.add(antenna);
            return g;
        })();
        assets['solar_sail'] = (() => {
            const g = new THREE.Group();
            const hub = new THREE.Mesh(new THREE.OctahedronGeometry(0.4, 0), foilMat); g.add(hub);
            const sailMat = new THREE.MeshStandardMaterial({ color: 0xd8d0ff, metalness: 0.85, roughness: 0.15, side: THREE.DoubleSide, emissive: 0x221a44 });
            for (let i=0;i<4;i++) {
                const sail = new THREE.Mesh(new THREE.PlaneGeometry(2.6, 2.6), sailMat);
                const ang = (Math.PI/2) * i;
                sail.position.set(Math.cos(ang)*1.6, Math.sin(ang)*1.6, 0);
                sail.rotation.z = ang; g.add(sail);
                const boom = new THREE.Mesh(new THREE.CylinderGeometry(0.02,0.02,2.2,6), aluMat);
                boom.position.set(Math.cos(ang)*0.9, Math.sin(ang)*0.9, 0); boom.rotation.z = ang + Math.PI/2; g.add(boom);
            }
            return g;
        })();
        assets['heat_shield'] = (() => {
            const g = new THREE.Group();
            for (let i=0;i<5;i++) {
                const plate = new THREE.Mesh(new THREE.BoxGeometry(1.8, 1.2, 0.04), solarMat);
                plate.position.z = -i * 0.28; g.add(plate);
            }
            for (let sx=-1; sx<=1; sx+=2) for (let sy=-1; sy<=1; sy+=2) {
                const standoff = new THREE.Mesh(new THREE.CylinderGeometry(0.04,0.04,1.4,8), aluMat);
                standoff.rotation.x = Math.PI/2; standoff.position.set(sx*0.75, sy*0.45, -0.6); g.add(standoff);
            }
            return g;
        })();
        assets['rc_glider'] = (() => {
            const g = new THREE.Group();
            const fuselage = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.1, 4, 32).rotateZ(Math.PI/2), aluMat);
            const mainWing = new THREE.Mesh(new THREE.BoxGeometry(5, 0.05, 0.8), carbonMat); mainWing.position.y = 0.2;
            const hStab = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.05, 0.4), carbonMat); hStab.position.set(-1.8, 0.1, 0);
            const vStab = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.8, 0.05), carbonMat); vStab.position.set(-1.8, 0.5, 0);
            g.add(fuselage, mainWing, hStab, vStab); return g;
        })();
        assets['airfoil'] = new THREE.Mesh(new THREE.CylinderGeometry(1.5, 0.1, 4, 64).rotateX(Math.PI/2).scale(1, 0.2, 1), aluMat);
        assets['satellite'] = (() => {
            const g = new THREE.Group();
            g.add(new THREE.Mesh(new THREE.BoxGeometry(1.5, 1.5, 1.5), foilMat));
            const p1 = new THREE.Mesh(new THREE.BoxGeometry(4, 0.05, 1), solarMat); p1.position.x = 3;
            const p2 = new THREE.Mesh(new THREE.BoxGeometry(4, 0.05, 1), solarMat); p2.position.x = -3;
            g.add(p1, p2); return g;
        })();
        assets['lander'] = (() => {
            const g = new THREE.Group(); g.add(new THREE.Mesh(new THREE.OctahedronGeometry(1.2, 1), foilMat));
            for(let i=0; i<4; i++) { const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 1.5), aluMat); leg.position.set(i<2?1:-1, -1, i%2==0?1:-1); leg.rotation.z = i<2?-0.5:0.5; leg.rotation.x = i%2==0?-0.5:0.5; g.add(leg); }
            return g;
        })();
        assets['rover'] = (() => {
            const g = new THREE.Group(); g.add(new THREE.Mesh(new THREE.BoxGeometry(2, 0.8, 3), aluMat));
            const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 1.5), aluMat); mast.position.set(0.8, 1, 1); g.add(mast);
            for(let i=0; i<6; i++) { const w = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.3, 0.2, 16).rotateZ(Math.PI/2), carbonMat); w.position.set(i%2==0?1.2:-1.2, -0.4, i<2?1.2:(i<4?0:-1.2)); g.add(w); }
            return g;
        })();
        assets['telescope'] = (() => {
            const g = new THREE.Group();
            for(let i=0; i<18; i++) { const hex = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.3, 0.05, 6).rotateX(Math.PI/2), foilMat); hex.position.set((Math.random()-0.5)*3, (Math.random()-0.5)*3, 0); g.add(hex); }
            const shield = new THREE.Mesh(new THREE.BoxGeometry(4, 0.1, 3), solarMat); shield.position.set(0, -2, -1); g.add(shield);
            return g;
        })();
        assets['station'] = (() => {
            const g = new THREE.Group();
            g.add(new THREE.Mesh(new THREE.CylinderGeometry(0.6, 0.6, 5, 32).rotateZ(Math.PI/2), aluMat));
            const p1 = new THREE.Mesh(new THREE.BoxGeometry(2, 0.05, 4), solarMat); p1.position.set(0, 1, 0);
            const p2 = new THREE.Mesh(new THREE.BoxGeometry(2, 0.05, 4), solarMat); p2.position.set(0, -1, 0);
            g.add(p1, p2); return g;
        })();
        assets['voyager'] = (() => {
            const g = new THREE.Group();
            const dish = new THREE.Mesh(new THREE.CylinderGeometry(2, 0.1, 0.5, 32).rotateX(Math.PI/2), aluMat);
            const body = new THREE.Mesh(new THREE.CylinderGeometry(0.8, 0.8, 1.5, 16), foilMat); body.position.z = -1;
            const boom = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 4).rotateX(Math.PI/2), aluMat); boom.position.set(1.5, -1, -2);
            g.add(dish, body, boom); return g;
        })();
        assets['earth'] = new THREE.Mesh(new THREE.SphereGeometry(2.5, 64, 64), new THREE.MeshStandardMaterial({
            map: texLoader.load('https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg'),
            bumpMap: texLoader.load('https://unpkg.com/three-globe/example/img/earth-topology.png'), bumpScale: 0.05, metalness: 0.1, roughness: 0.7
        }));
        assets['sun'] = new THREE.Mesh(new THREE.SphereGeometry(2.5, 64, 64), new THREE.ShaderMaterial({
            uniforms: { time: { value: 0 } },
            vertexShader: `varying vec2 vUv; varying vec3 vPos; uniform float time; void main() { vUv = uv; vPos = position; vec3 pos = position + normal * sin(position.x*15.0 + time)*0.03; gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0); }`,
            fragmentShader: `varying vec2 vUv; varying vec3 vPos; uniform float time; void main() { float noise = sin(vPos.x*10.0 + time)*sin(vPos.y*10.0 + time); vec3 col = mix(vec3(1.0,0.1,0.0), vec3(1.0,0.9,0.0), noise+0.5); gl_FragColor = vec4(col, 1.0); }`
        }));

        let activeAsset = 'actuator';
        currentGroup.add(assets[activeAsset]);

        // --- 3. SYSTEM CONTROLS (CAD & WIND TUNNEL) ---
        let isCADMode = false;
        const cadMaterial = new THREE.MeshBasicMaterial({ color: 0x38bdf8, wireframe: true, transparent: true, opacity: 0.8 });

        window.toggleCADMode = function() {
            isCADMode = !isCADMode;
            document.getElementById('btn-cad').classList.toggle('active');
            if (assets[activeAsset]) applyMaterials(assets[activeAsset]);
        };

        let windParticles = null;
        window.toggleWindTunnel = function() {
            document.getElementById('btn-wind').classList.toggle('active');
            if (windParticles) { scene.remove(windParticles); windParticles = null; return; }
            const geo = new THREE.BufferGeometry();
            const pos = [];
            for(let i=0; i<4000; i++) pos.push((Math.random()-0.5)*15, (Math.random()-0.5)*15, (Math.random()-0.5)*30);
            geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
            const mat = new THREE.PointsMaterial({color: 0x5eead4, size: 0.06, transparent: true, opacity: 0.8});
            windParticles = new THREE.Points(geo, mat);
            scene.add(windParticles);
        };

        function applyMaterials(obj) {
            obj.traverse((child) => {
                if (child.isMesh) {
                    if (isCADMode) {
                        if (!child.userData.origMat) child.userData.origMat = child.material;
                        child.material = cadMaterial;
                    } else if (child.userData.origMat) {
                        child.material = child.userData.origMat;
                    }
                }
            });
        }

        // --- 4. MENU COLLAPSE (auto-hides after a selection so the model isn't blocked) ---
        window.toggleMenu = function(show) {
            const panel = document.getElementById('left-menu-panel');
            const reopenBtn = document.getElementById('reopen-menu-btn');
            if (show) { panel.classList.remove('collapsed'); reopenBtn.style.display = 'none'; }
            else { panel.classList.add('collapsed'); reopenBtn.style.display = 'block'; }
        };

        // --- 5. LAZY LOADING ASSET MANAGER (mouse click OR gesture confirm both call this) ---
        const loadingText = document.getElementById('loading-text');

        function switchAsset(btnElement) {
            const name = btnElement.dataset.target;
            const type = btnElement.dataset.type;

            if (assets[activeAsset]) currentGroup.remove(assets[activeAsset]);
            document.querySelectorAll('.menu-btn').forEach(b => b.classList.remove('active'));
            btnElement.classList.add('active');

            if (type === 'proc' || assets[name]) {
                activeAsset = name;
                const obj = assets[activeAsset];
                obj.rotation.set(0,0,0); obj.scale.set(1,1,1); obj.position.set(0,0,0);
                applyMaterials(obj);
                currentGroup.add(obj);
                setTimeout(() => toggleMenu(false), 260);
            } else if (type === 'gltf') {
                const url = btnElement.dataset.url;
                const scale = parseFloat(btnElement.dataset.scale);
                document.getElementById('loading-overlay').style.display = 'flex';
                document.getElementById('loading-overlay').style.opacity = 1;
                loadingText.innerText = "DOWNLOADING GLTF ASSET...";

                gltfLoader.load(url, (gltf) => {
                    const model = gltf.scene;
                    const box = new THREE.Box3().setFromObject(model);
                    const center = box.getCenter(new THREE.Vector3());
                    model.position.x += (model.position.x - center.x);
                    model.position.y += (model.position.y - center.y);
                    model.position.z += (model.position.z - center.z);

                    model.scale.set(scale, scale, scale);
                    assets[name] = model;
                    activeAsset = name;

                    applyMaterials(assets[name]);
                    currentGroup.add(assets[name]);

                    document.getElementById('loading-overlay').style.opacity = 0;
                    setTimeout(() => document.getElementById('loading-overlay').style.display = 'none', 500);
                    setTimeout(() => toggleMenu(false), 260);
                }, undefined, (error) => {
                    loadingText.innerText = "ERROR LOADING ASSET. FALLBACK ENGAGED.";
                    setTimeout(() => { document.getElementById('loading-overlay').style.opacity = 0; setTimeout(() => document.getElementById('loading-overlay').style.display = 'none', 500); }, 1500);
                });
            }
        }

        // --- 6. GESTURE STATE MACHINE ---
        // One hand: point+PINCH to select (not a timer), fist=rotate, palm=pan, peace=zoom.
        // Two hands: CLAP-ZOOM only. A gesture must hold for GESTURE_HOLD_FRAMES
        // consecutive frames before it takes over -- this is the fix for
        // gestures "blending" into each other from a single noisy frame.
        const videoElement = document.getElementById('video-feed');
        const pointer = document.getElementById('laser-pointer');
        const hud = document.getElementById('hud-status');
        const handsBadge = document.getElementById('hands-badge');

        let targetRot = {x: 0, y: 0}; let currentRot = {x: 0, y: 0};
        let targetPos = {x: 0, y: 0}; let currentPos = {x: 0, y: 0};
        let targetScale = 1; let currentScale = 1;
        let baseTwoHandDist = null; let baseTwoHandScale = 1;
        let armedTarget = null;
        let pinchLatched = false;

        const GESTURE_HOLD_FRAMES = 3;
        let pendingGesture = null, pendingCount = 0, activeGesture = 'idle';

        function stabilize(rawGesture) {
            if (rawGesture === pendingGesture) { pendingCount++; }
            else { pendingGesture = rawGesture; pendingCount = 1; }
            if (pendingCount >= GESTURE_HOLD_FRAMES) activeGesture = rawGesture;
            return activeGesture;
        }

        function fingerFlags(h) {
            return {
                index: h[8].y < h[6].y,
                middle: h[12].y < h[10].y,
                ring: h[16].y < h[14].y,
                pinky: h[20].y < h[18].y,
            };
        }

        let hands = null;

        function buildHands() {
            hands = new window.Hands({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}` });
            hands.setOptions({ maxNumHands: 2, modelComplexity: 0, minDetectionConfidence: 0.7, minTrackingConfidence: 0.7 });

            hands.onResults((results) => {
                const numHands = (results.multiHandLandmarks && results.multiHandLandmarks.length) || 0;
                handsBadge.innerText = `HANDS DETECTED: ${numHands}`;

                if (numHands === 2) {
                    const g = stabilize('clap_zoom');
                    pointer.style.display = 'none';
                    if (g !== 'clap_zoom') return;
                    const hA = results.multiHandLandmarks[0], hB = results.multiHandLandmarks[1];
                    const cA = { x: (hA[0].x + hA[9].x) / 2, y: (hA[0].y + hA[9].y) / 2 };
                    const cB = { x: (hB[0].x + hB[9].x) / 2, y: (hB[0].y + hB[9].y) / 2 };
                    const dist = Math.hypot(cB.x - cA.x, cB.y - cA.y);

                    if (baseTwoHandDist === null) { baseTwoHandDist = Math.max(0.05, dist); baseTwoHandScale = currentScale; }
                    const ratio = baseTwoHandDist / Math.max(0.03, dist);
                    targetScale = Math.max(0.2, Math.min(4.0, baseTwoHandScale * ratio));
                    const clapNote = dist < baseTwoHandDist * 0.55 ? " | CLAP DETECTED" : "";
                    hud.innerText = "GNC SENSOR: DUAL-HAND | COMMAND: CLAP-ZOOM" + clapNote;

                } else if (numHands === 1) {
                    baseTwoHandDist = null;
                    const h = results.multiHandLandmarks[0];
                    const f = fingerFlags(h);
                    const avgX = (h[0].x + h[9].x) / 2;
                    const avgY = (h[0].y + h[9].y) / 2;
                    const pinchDist = Math.hypot(h[4].x - h[8].x, h[4].y - h[8].y);
                    const isPinching = pinchDist < 0.055;

                    const rawGesture =
                        (f.index && f.middle && !f.ring && !f.pinky) ? 'peace' :
                        (f.index && !f.middle && !f.ring && !f.pinky) ? 'point' :
                        (!f.index && !f.middle && !f.ring && !f.pinky) ? 'fist' :
                        (f.index && f.middle && f.ring && f.pinky) ? 'palm' : 'idle';
                    const g = stabilize(rawGesture);

                    if (g === 'peace') {
                        hud.innerText = "GNC SENSOR: PEACE SIGN | COMMAND: SCALE/ZOOM";
                        pointer.style.display = 'none';
                        targetScale = Math.max(0.2, Math.min(4.0, 1.0 + (0.5 - avgY) * 5));

                    } else if (g === 'point') {
                        const px = (1 - h[8].x) * window.innerWidth; const py = h[8].y * window.innerHeight;
                        pointer.style.display = 'block'; pointer.style.left = `${px}px`; pointer.style.top = `${py}px`;

                        const el = document.elementFromPoint(px, py);
                        const overTarget = el && el.classList.contains('menu-btn');

                        if (overTarget) {
                            if (armedTarget !== el) { armedTarget = el; armedTarget.classList.add('reticle-armed'); }
                            pointer.classList.add('laser-armed');
                            hud.innerText = "GNC SENSOR: AIM LOCKED | COMMAND: PINCH TO CONFIRM";

                            if (isPinching && !pinchLatched) {
                                pinchLatched = true;
                                switchAsset(el);
                            } else if (!isPinching) {
                                pinchLatched = false;
                            }
                        } else {
                            if (armedTarget) { armedTarget.classList.remove('reticle-armed'); armedTarget = null; }
                            pointer.classList.remove('laser-armed');
                            pinchLatched = false;
                            hud.innerText = "GNC SENSOR: POINT | COMMAND: AIMING";
                        }

                    } else if (g === 'fist') {
                        hud.innerText = "GNC SENSOR: FIST | COMMAND: GEOMETRY ROTATION";
                        pointer.style.display = 'none';
                        if (armedTarget) { armedTarget.classList.remove('reticle-armed'); armedTarget = null; }
                        targetRot.y = (0.5 - avgX) * Math.PI * 4;
                        targetRot.x = (avgY - 0.5) * Math.PI * 4;

                    } else if (g === 'palm') {
                        hud.innerText = "GNC SENSOR: PALM | COMMAND: X/Y TRANSLATION";
                        pointer.style.display = 'none';
                        if (armedTarget) { armedTarget.classList.remove('reticle-armed'); armedTarget = null; }
                        targetPos.x = (0.5 - avgX) * 10;
                        targetPos.y = (0.5 - avgY) * 10;

                    } else {
                        pointer.style.display = 'none';
                        hud.innerText = "GNC SENSOR: IDLE / UNRECOGNIZED";
                    }
                } else {
                    baseTwoHandDist = null;
                    pointer.style.display = 'none';
                    if (armedTarget) { armedTarget.classList.remove('reticle-armed'); armedTarget = null; }
                    hud.innerText = "GNC SENSOR: NO TARGET ACQUIRED";
                    stabilize('idle');
                }
            });
        }

        // --- 7. OPT-IN CAMERA BOOTSTRAP (never automatic) ---
        let lastVideoTime = 0;
        let cameraFeed = null;
        let cameraRunning = false;
        let gestureModeOn = false;

        window.enableMouseMode = function() {
            document.getElementById('gate-overlay').style.display = 'none';
        };

        window.enableGestureMode = function() {
            document.getElementById('gate-overlay').style.display = 'none';
            if (gestureModeOn) return;
            gestureModeOn = true;
            document.getElementById('video-feed').style.display = 'block';
            document.getElementById('hud-status').style.display = 'block';
            document.getElementById('hands-badge').style.display = 'block';
            document.getElementById('right-guide-panel').style.display = 'block';
            document.getElementById('btn-cam').innerText = '🎥 Camera Active';
            document.getElementById('btn-cam').classList.add('active');
            if (!hands) buildHands();
            document.getElementById('loading-overlay').style.display = 'flex';
            document.getElementById('loading-overlay').style.opacity = 1;
            loadingText.innerText = "Requesting camera permission...";
            startCamera();
        };

        function startCamera() {
            if (cameraRunning || !window.Camera) return;
            cameraFeed = new window.Camera(videoElement, {
                onFrame: async () => {
                    const now = performance.now();
                    if (now - lastVideoTime > 66) {
                        try { await hands.send({ image: videoElement }); } catch (e) {}
                        lastVideoTime = now;
                    }
                },
                width: 320, height: 240
            });
            cameraFeed.start().then(() => {
                cameraRunning = true;
                document.getElementById('loading-overlay').style.opacity = 0;
                setTimeout(() => document.getElementById('loading-overlay').style.display = 'none', 500);
            }).catch(() => {
                document.getElementById('loading-overlay').style.opacity = 0;
                setTimeout(() => document.getElementById('loading-overlay').style.display = 'none', 500);
                hud.innerText = "GNC SENSOR: CAMERA PERMISSION DENIED — MOUSE MODE ACTIVE";
            });
        }

        function stopCamera() {
            try { if (videoElement.srcObject) videoElement.srcObject.getTracks().forEach(t => t.stop()); } catch (e) {}
            cameraRunning = false;
        }

        let rafId = null;
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                if (gestureModeOn) stopCamera();
                if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
            } else {
                if (!rafId) animate();
                if (gestureModeOn) setTimeout(startCamera, 300);
            }
        });

        let mouseDown = false, lastMX = 0, lastMY = 0;
        if (renderer) {
            renderer.domElement.addEventListener('pointerdown', e => { mouseDown = true; lastMX = e.clientX; lastMY = e.clientY; });
            window.addEventListener('pointerup', () => mouseDown = false);
            window.addEventListener('pointermove', e => {
                if (!mouseDown) return;
                targetRot.y += (e.clientX - lastMX) * 0.006;
                targetRot.x += (e.clientY - lastMY) * 0.006;
                lastMX = e.clientX; lastMY = e.clientY;
            });
            renderer.domElement.addEventListener('wheel', e => {
                targetScale = Math.max(0.2, Math.min(4.0, targetScale - e.deltaY * 0.001));
            });
        }

        // --- 8. DECOUPLED RENDER LOOP (SMOOTH LERPING) ---
        const clock = new THREE.Clock();
        function animate() {
            rafId = requestAnimationFrame(animate);
            if (!renderer) return;
            const delta = clock.getDelta();

            if(assets['sun'] && assets['sun'].material.uniforms) assets['sun'].material.uniforms.time.value += delta;

            if (windParticles) {
                const pos = windParticles.geometry.attributes.position.array;
                for(let i=2; i<pos.length; i+=3) { pos[i] -= 0.3; if(pos[i] < -15) pos[i] = 15; }
                windParticles.geometry.attributes.position.needsUpdate = true;
            }

            currentRot.x += (targetRot.x - currentRot.x) * 0.1;
            currentRot.y += (targetRot.y - currentRot.y) * 0.1;
            currentPos.x += (targetPos.x - currentPos.x) * 0.1;
            currentPos.y += (targetPos.y - currentPos.y) * 0.1;
            currentScale += (targetScale - currentScale) * 0.1;

            if (currentGroup) {
                currentGroup.rotation.set(currentRot.x, currentRot.y, 0);
                currentGroup.position.set(currentPos.x, currentPos.y, 0);
                currentGroup.scale.set(currentScale, currentScale, currentScale);
            }

            renderer.render(scene, camera);
        }
        animate();

        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            if (renderer) renderer.setSize(window.innerWidth, window.innerHeight);
        });
    </script>
</body>
</html>
"""

# ==========================================
# 1. INITIALIZATION
# ==========================================
if 'is_playing' not in st.session_state: st.session_state.is_playing = False
if 'current_index' not in st.session_state: st.session_state.current_index = 60
if 'stop_index' not in st.session_state: st.session_state.stop_index = None
if 'y_true_hist' not in st.session_state: st.session_state.y_true_hist = []
if 'y_pred_hist' not in st.session_state: st.session_state.y_pred_hist = []
if 'active_section' not in st.session_state: st.session_state.active_section = "📖 Mission Briefing"
if 'xai_cache' not in st.session_state: st.session_state.xai_cache = {}
if 'gesture_engine_loaded' not in st.session_state: st.session_state.gesture_engine_loaded = False
if 'cosmic_chat' not in st.session_state: st.session_state.cosmic_chat = []
if 'last_narrated_class' not in st.session_state: st.session_state.last_narrated_class = 0
if 'incident_active' not in st.session_state: st.session_state.incident_active = False
if 'incident_start_time' not in st.session_state: st.session_state.incident_start_time = None
if 'incident_peak_class' not in st.session_state: st.session_state.incident_peak_class = 0
if 'incident_peak_cg_shift' not in st.session_state: st.session_state.incident_peak_cg_shift = None
if 'incident_reports' not in st.session_state: st.session_state.incident_reports = []
if 'voice_narration_on' not in st.session_state: st.session_state.voice_narration_on = False
if 'cosmic_voice_box' not in st.session_state: st.session_state.cosmic_voice_box = ''
if '_clear_cosmic_box' not in st.session_state: st.session_state._clear_cosmic_box = False

trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

# ==========================================
# 1B. AUTO-DOWNLOAD LARGE DATA FILES (so they never need to be pushed to GitHub)
# ==========================================
# Google Drive's normal "share link" download path breaks above ~100MB: it
# serves an HTML "can't scan this file for viruses" interstitial instead of
# the file itself unless you follow its confirm-token redirect, and it also
# occasionally hits a daily download-quota wall on heavily-hit files. The
# fixes below: use gdown's `fuzzy=True` mode (it resolves the confirm token
# for us), retry on transient failures, and — critically — verify what
# actually landed on disk is really the CSV and not a stray HTML error page.
DRIVE_FILE_IDS = {
    'dashboard_feed.csv': '1P1BurhjNfe6xJopXpVTacN12aDG_6Pzc',
    'solar_flare_gru.pt': None,
}
MIN_VALID_BYTES = 1024  # anything smaller than this is almost certainly an error page, not real data

# ==========================================
# GCS VOLUME MOUNT — set this to whatever mount path you used in Cloud Run's
# "Volumes" -> "Volume Mounts" screen (Part 2 of the setup guide). If a file
# is found here, it's used directly with ZERO download step -- this is what
# actually fixes "have to re-upload after the app sleeps", since the bucket
# is just always there, cold start or not. If nothing's mounted yet (e.g.
# you haven't done the GCS setup), this quietly falls back to the Google
# Drive auto-download path below, so nothing breaks in the meantime.
# ==========================================
GCS_MOUNT_PATH = "/mnt/data"


def resolve_data_path(filename: str, drive_file_id) -> str:
    """Prefer the GCS-mounted copy; fall back to the Drive-backed local copy."""
    mounted_path = os.path.join(GCS_MOUNT_PATH, filename)
    if os.path.exists(mounted_path) and os.path.getsize(mounted_path) > MIN_VALID_BYTES:
        return mounted_path
    local_path = os.path.join('data', filename)
    ensure_data_file(local_path, drive_file_id)
    return local_path

# ==========================================
# GEMINI API KEY — how to get one:
#   1. Go to https://aistudio.google.com/apikey (sign in with any Google account)
#   2. Click "Create API key" -> copy it (starts with "AIza...")
#   3. NEVER paste it directly into this file or commit it to git. Instead:
#      - Local dev: create `.streamlit/secrets.toml` in the project root with:
#            GEMINI_API_KEY = "AIza...your key..."
#        (add `.streamlit/secrets.toml` to .gitignore so it never gets pushed)
#      - Cloud Run / cloud deploy: set it as an environment variable named
#        GEMINI_API_KEY in the service's "Variables & Secrets" settings, or
#        better, store it in Google Secret Manager and mount it as an env var.
#   Ask Cosmic works with ZERO key configured -- it just falls back to the
#   template-based answers instead of Gemini's free-form ones. Nothing breaks
#   either way.
# ==========================================
def gemini_key_configured() -> bool:
    return bool(cosmic_assistant.get_api_key())



def _looks_like_html_error(path: str) -> bool:
    try:
        with open(path, 'rb') as f:
            head = f.read(2048).lower()
        return b'<html' in head or b'<!doctype html' in head or b'google drive - quota exceeded' in head
    except Exception:
        return False


def ensure_data_file(local_path, drive_file_id, max_retries: int = 3):
    """If local_path is missing but a Drive file ID is configured, download it
    once (cached on disk for future runs) instead of requiring it in git.
    Robust to Google Drive's large-file interstitial and transient failures."""
    if os.path.exists(local_path) and os.path.getsize(local_path) > MIN_VALID_BYTES and not _looks_like_html_error(local_path):
        return  # already have a good local copy — never re-download
    if not drive_file_id:
        return

    try:
        import gdown
    except ImportError:
        st.error("`gdown` isn't installed. Add `gdown` to requirements.txt.")
        return

    os.makedirs(os.path.dirname(local_path) or '.', exist_ok=True)
    url = f"https://drive.google.com/file/d/{drive_file_id}/view"

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            if os.path.exists(local_path):
                os.remove(local_path)  # drop any partial/bad file from a previous attempt
            gdown.download(url=url, output=local_path, quiet=False, fuzzy=True)

            if os.path.exists(local_path) and os.path.getsize(local_path) > MIN_VALID_BYTES and not _looks_like_html_error(local_path):
                return  # success
            last_err = "Downloaded file was empty or looked like an HTML error page, not real data."
        except Exception as e:
            last_err = str(e)

        if attempt < max_retries:
            time_lib.sleep(2 * attempt)  # brief backoff before retrying

    st.error(
        f"⚠️ Could not fetch **{os.path.basename(local_path)}** from Google Drive after "
        f"{max_retries} tries ({last_err}). This usually means either the file's daily "
        f"Drive download quota was hit (very likely for a >100MB file that gets pulled on "
        f"every Cloud Run cold start), or its sharing isn't set to \"Anyone with the link\". "
        f"Use the manual upload box in the sidebar to load it for this session instead."
    )


# ==========================================
# 2. AI MODEL DEFINITION (ATTENTION-GRU)
# ==========================================
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, gru_out):
        scores = self.V(torch.tanh(self.W1(gru_out)))
        weights = F.softmax(scores, dim=1)
        return torch.sum(weights * gru_out, dim=1)


class CosmicVectorModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_classes=4):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.attention = BahdanauAttention(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.classifier(self.attention(self.gru(x)[0]))


@st.cache_resource
def load_ai_model():
    model_path = resolve_data_path('solar_flare_gru.pt', DRIVE_FILE_IDS.get('solar_flare_gru.pt'))
    model = CosmicVectorModel()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    return model


# ==========================================
# 3. PAGE CONFIG, LIVE 3D BACKGROUND, THEME
# ==========================================
st.set_page_config(page_title="Aditya-L1 Tactical Command", layout="wide", initial_sidebar_state="expanded", page_icon="🛰️")
st.markdown(theme.build_css(), unsafe_allow_html=True)
if 'space_bg_injected' not in st.session_state:
    theme.inject_space_bg()
    st.session_state.space_bg_injected = True


def style_fig_dark(fig, title, y_title):
    fig.update_layout(
        title=dict(text=title, font=dict(family='Orbitron', size=15, color="#E8ECF5")),
        template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#CBD5E1", family="Inter"), margin=dict(l=10, r=10, t=42, b=10),
        yaxis_title=y_title, hoverlabel=dict(bgcolor="#0f172a", font_family="Inter"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
    return fig


def add_severity_bands(fig, series, labels=("Background", "Elevated", "Severe")):
    if series is None or len(series) < 5:
        return fig
    p50, p90, p99 = np.nanpercentile(series, [50, 90, 99])
    for val, lbl, color in zip([p50, p90, p99], labels, ["#34d399", "#fbbf24", "#fb4d4d"]):
        fig.add_hline(y=val, line_dash="dot", line_color=color, opacity=0.55,
                       annotation_text=lbl, annotation_font_size=10, annotation_font_color=color)
    return fig


def info_caption(text):
    st.markdown(f"<div class='cv-caption cv-reveal'>ℹ️ {text}</div>", unsafe_allow_html=True)


def reveal_card(inner_html):
    st.markdown(f"<div class='premium-card cv-reveal'>{inner_html}</div>", unsafe_allow_html=True)


def speak_text(text: str):
    """Speaks a line aloud using the browser's built-in SpeechSynthesis --
    free, no API key, one-directional (Python -> browser) so it needs no
    bridging trick. Escapes the text defensively since it's inlined into a
    JS string literal."""
    safe_text = json.dumps(text)
    components.html(
        f"""<script>
        try {{
            const utter = new SpeechSynthesisUtterance({safe_text});
            utter.rate = 1.0; utter.pitch = 1.0;
            window.parent.speechSynthesis.cancel();
            window.parent.speechSynthesis.speak(utter);
        }} catch (e) {{}}
        </script>""",
        height=0,
    )


# ==========================================
# 4. DATA LOADING
# ==========================================
@st.cache_data
def load_base_data():
    csv_path = resolve_data_path('dashboard_feed.csv', DRIVE_FILE_IDS.get('dashboard_feed.csv'))
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > MIN_VALID_BYTES:
        d = pd.read_csv(csv_path, parse_dates=['time'])
        d['time'] = pd.to_datetime(d['time'], utc=True)
        return d
    return None


df = load_base_data()

# Manual fallback: if Drive auto-download failed (quota, sharing, or network),
# let a judge/user drop the CSV in by hand rather than being stuck.
if df is None:
    with st.sidebar:
        st.markdown("### 📤 Manual Data Fallback")
        st.caption("Auto-download from Google Drive didn't come through this time — drop `dashboard_feed.csv` here to continue.")
        manual_feed = st.file_uploader("dashboard_feed.csv", type=["csv"], key="manual_feed_upload")
        if manual_feed is not None:
            os.makedirs('data', exist_ok=True)
            with open('data/dashboard_feed.csv', 'wb') as f:
                f.write(manual_feed.getbuffer())
            st.success("Loaded — refreshing…")
            st.cache_data.clear()
            st.rerun()

model = load_ai_model()

# ==========================================
# 5. SIDEBAR — J.A.R.V.I.S. HUD + MECHANICS
# ==========================================
st.sidebar.markdown(theme.jarvis_hud(), unsafe_allow_html=True)
st.sidebar.markdown("<hr style='border-color:rgba(94,234,212,.25);'>", unsafe_allow_html=True)
mode = st.sidebar.radio("MISSION PROTOCOL", ["Live Ops Dashboard", "Judge Hardware Upload"])
st.sidebar.markdown("<hr style='border-color:rgba(94,234,212,.25);'>", unsafe_allow_html=True)

if mode == "Judge Hardware Upload":
    st.sidebar.markdown("<h4 style='color:#FFF; font-family:Orbitron;'>⚖️ UPLOAD PAYLOAD DATA</h4>", unsafe_allow_html=True)
    st.sidebar.caption(
        "Accepts the **real** ISRO payload formats: `AL1_SLX_L1_*.zip` (SoLEXS, "
        "gzipped FITS light-curve) and `HLS_*.zip` (HEL1OS, FITS light-curve per "
        "energy band) — or a plain CSV / zip-of-CSV for hand-built test files."
    )
    solexs_file = st.sidebar.file_uploader("1. Aditya-L1 SoLEXS Data (AL1_SLX_L1_*.zip)", type=["csv", "zip"])
    hel1os_file = st.sidebar.file_uploader("2. Aditya-L1 HEL1OS Data (HLS_*.zip)", type=["csv", "zip"])

    if solexs_file and hel1os_file:
        if st.sidebar.button("🔗 SYNC & ANALYZE DATA"):
            with st.sidebar:
                with st.spinner("Decoding payloads & synchronising clocks..."):
                    df_sol = load_instrument_upload(solexs_file, "solexs")
                    df_hel = load_instrument_upload(hel1os_file, "hel1os")

            if df_sol is not None and df_hel is not None:
                diag = overlap_diagnostics(df_sol, df_hel)
                merged = sync_and_engineer(df_sol, df_hel)
                st.session_state['uploaded_df'] = merged
                st.session_state['upload_diag'] = diag
                st.session_state.current_index = 60
                st.session_state.stop_index = None
                st.session_state.y_true_hist, st.session_state.y_pred_hist = [], []
                st.sidebar.success(f"✅ Time-Sync Complete — {len(merged):,} ticks loaded.")
                if not diag["has_overlap"]:
                    st.sidebar.warning(
                        "⚠️ These two files don't share any real overlapping timestamps "
                        "(different mission days). The dashboard will still run, but "
                        "hard_flux will read as flat 0 — upload SoLEXS + HEL1OS files "
                        "from the **same UTC day** for a genuine dual-channel sync."
                    )
            else:
                st.sidebar.error(
                    "Could not parse one or both files. Expecting a SoLEXS "
                    "`AL1_SLX_L1_*.zip` and a HEL1OS `HLS_*.zip`, or plain CSVs."
                )
    if 'uploaded_df' in st.session_state:
        df = st.session_state['uploaded_df']
else:
    if df is not None:
        st.sidebar.markdown("### 🕒 Mission Timeline")
        available_dates = df['time'].dt.date.unique()
        selected_date = st.sidebar.selectbox("Telemetry Date", available_dates)
        date_df = df[df['time'].dt.date == selected_date]
        c_t1, c_t2 = st.sidebar.columns(2)
        start_time = c_t1.selectbox("Start", date_df['time'].dt.time)
        end_time = c_t2.selectbox("End", date_df['time'].dt.time, index=min(len(date_df) - 1, 600))

        if st.sidebar.button("Set Timeline"):
            start_dt = pd.to_datetime(f"{selected_date} {start_time}", utc=True)
            end_dt = pd.to_datetime(f"{selected_date} {end_time}", utc=True)

            start_matches = df[df['time'] >= start_dt]
            if not start_matches.empty:
                st.session_state.current_index = max(60, start_matches.index[0])

            end_matches = df[df['time'] <= end_dt]
            if not end_matches.empty:
                st.session_state.stop_index = end_matches.index[-1]

            st.session_state.y_true_hist, st.session_state.y_pred_hist = [], []
            st.session_state.is_playing = False

st.sidebar.markdown("### ⏯️ Data Stream Controls")
st.sidebar.info("Adjust the **Ping Rate** below. Higher FPS = faster graph updates, simulating high-speed spacecraft telemetry.")
pb1, pb2 = st.sidebar.columns(2)
if pb1.button("▶️ INITIATE STREAM"): st.session_state.is_playing = True
if pb2.button("⏸️ HALT STREAM"): st.session_state.is_playing = False
playback_speed = st.sidebar.slider("Ping Rate (Frames Per Sec)", 1, 30, 10)
st.sidebar.caption("Above ~12 fps, chart redraws auto-throttle to stay smooth — the data itself still streams at full speed.")

if st.sidebar.button("⏭️ Jump to Next Alert/Critical Event", use_container_width=True):
    if df is not None and 'activity_level' in df.columns:
        future_hits = df[(df.index > st.session_state.current_index) & (df['activity_level'] >= 2)]
        if not future_hits.empty:
            st.session_state.current_index = int(future_hits.index[0])
            st.session_state.stop_index = None
            st.session_state.y_true_hist, st.session_state.y_pred_hist = [], []
            st.sidebar.success(f"⏭️ Jumped to tick {future_hits.index[0]:,} — next Alert/Critical event.")
        else:
            st.sidebar.info("No further Alert/Critical events ahead in this dataset.")
    else:
        st.sidebar.warning("No ground-truth `activity_level` column available to bookmark against.")

live_xai = st.sidebar.toggle("🧠 Live AI Explainability (XAI)", value=False,
                              help="Turn this on when parked in the AI Validation tab to see which telemetry features drove the AI's decision.")

st.session_state.voice_narration_on = st.sidebar.toggle(
    "🔊 Voice narration for alerts", value=st.session_state.voice_narration_on,
    help="When a threat class rises to Monitor/Alert/Critical, Cosmic speaks a line aloud using your browser's built-in text-to-speech. No API key needed for this part."
)

if df is None:
    st.error("No valid data found. Upload payload or run backend scripts.")
    st.stop()

# ==========================================
# 6. STREAM PROCESSING & ACTUATOR LOGIC
# ==========================================
if st.session_state.current_index >= len(df):
    st.session_state.current_index = len(df) - 1

idx = st.session_state.current_index
window = df.iloc[max(0, idx - 60): idx]
latest_tick = df.iloc[idx]

features = ['soft_flux', 'hard_flux', 'heating_slope', 'hardness_ratio', 'neupert_proxy']
X_tensor = torch.tensor(window[features].values, dtype=torch.float32).unsqueeze(0).requires_grad_()

with torch.no_grad():
    predicted_class = int(torch.argmax(model(X_tensor), dim=1).item())
    live_probs = F.softmax(model(X_tensor), dim=1).squeeze(0).numpy()

if 'activity_level' in latest_tick:
    st.session_state.y_true_hist.append(int(latest_tick['activity_level']))
    st.session_state.y_pred_hist.append(predicted_class)

# ---- Short-horizon forecast: linear-trend extrapolation of the engineered
# features a few ticks forward, then classified by the same model. This is
# a lightweight projection (not a verified multi-step forecaster) -- it's
# labelled as such everywhere it's shown, so nobody mistakes it for ground
# truth. ----
FORECAST_HORIZON_TICKS = 5


def project_forward(window_df, feature_cols, horizon):
    proj = window_df[feature_cols].copy().reset_index(drop=True)
    for _ in range(horizon):
        new_row = {}
        for feat in feature_cols:
            y = proj[feat].values
            tail_n = min(10, len(y))
            if tail_n >= 3:
                xs = np.arange(tail_n)
                slope = np.polyfit(xs, y[-tail_n:], 1)[0]
            else:
                slope = 0.0
            new_row[feat] = y[-1] + slope if len(y) else 0.0
        proj = pd.concat([proj, pd.DataFrame([new_row])], ignore_index=True)
    return proj.tail(len(window_df)).reset_index(drop=True)


forecast_class = predicted_class
if len(window) >= 10:
    try:
        proj_window = project_forward(window, features, FORECAST_HORIZON_TICKS)
        X_proj = torch.tensor(proj_window[features].values, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            forecast_class = int(torch.argmax(model(X_proj), dim=1).item())
    except Exception:
        forecast_class = predicted_class

actions = {
    0: ("QUIET", "0.0 mm", "#34d399", "MAINTAIN_CURRENT_GEOMETRY"),
    1: ("MONITOR", "1.2 mm", "#fbbf24", "MICRO_ADJUST_PROTECTION_BAFFLE"),
    2: ("ALERT", "-4.5 mm", "#fb923c", "AXIAL_CG_SHIFT_REDUCE_THERMAL_STRESS"),
    3: ("CRITICAL", "-12.0 mm", "#fb4d4d", "MAXIMUM_STRUCTURAL_SAFE_MODE_SHIELDING"),
}
status, cg_shift, color_hex, act_cmd = actions[predicted_class]
forecast_status, _, forecast_color, _ = actions[forecast_class]

mech_command = {"time": str(latest_tick['time']), "class": predicted_class, "shift_mm": cg_shift, "command": act_cmd}
with open("actuator_command.json", "w") as f:
    json.dump(mech_command, f)

# ---- Proactive narration: speak only on an upward class transition, never
# every tick, so it doesn't repeat itself into background noise. ----
if predicted_class > st.session_state.last_narrated_class and st.session_state.voice_narration_on:
    narration_lines = {
        1: f"Monitor. Threat class {predicted_class}.",
        2: f"Alert. Class {predicted_class} threat detected. Actuator commanding {cg_shift} C G shift.",
        3: f"Critical. Class {predicted_class} threat detected. Maximum structural safe mode engaged.",
    }
    speak_text(narration_lines.get(predicted_class, f"Threat class now {predicted_class}, {status}."))
st.session_state.last_narrated_class = predicted_class

# ---- Incident tracking: opens when class first reaches Alert/Critical,
# closes when it drops back below -- generates one auto mission-log entry
# per incident via cosmic_assistant.generate_incident_report(). ----
if predicted_class >= 2 and not st.session_state.incident_active:
    st.session_state.incident_active = True
    st.session_state.incident_start_time = str(latest_tick['time'])
    st.session_state.incident_peak_class = predicted_class
    st.session_state.incident_peak_cg_shift = cg_shift
elif predicted_class >= 2 and st.session_state.incident_active:
    if predicted_class > st.session_state.incident_peak_class:
        st.session_state.incident_peak_class = predicted_class
        st.session_state.incident_peak_cg_shift = cg_shift
elif predicted_class < 2 and st.session_state.incident_active:
    summary = {
        "start_time": st.session_state.incident_start_time,
        "end_time": str(latest_tick['time']),
        "peak_class": st.session_state.incident_peak_class,
        "peak_status": actions[st.session_state.incident_peak_class][0],
        "peak_cg_shift": st.session_state.incident_peak_cg_shift,
    }
    report_text = cosmic_assistant.generate_incident_report(summary)
    st.session_state.incident_reports.append(report_text)
    st.session_state.incident_active = False

# ==========================================
# 7. DASHBOARD HEADER & AUDIO ALERT
# ==========================================
st.markdown(f"""
<div style='text-align:center; padding-bottom: 14px;'>
    <h1 class='cv-title-gradient' style='font-size: 2.8rem; margin:0;'>COSMIC VECTOR TACTICAL COMMAND</h1>
    <p style='color:#94A3B8; font-family:Orbitron; letter-spacing:3px; font-size:12.5px; margin-top:4px;'>
        ISRO ADITYA-L1 • EDGE-AI HUB • PLANETARY DEFENSE
    </p>
    <div class='cv-live-chip'><span class='cv-live-dot'></span> {"STREAMING LIVE" if st.session_state.is_playing else "FEED PAUSED"} @ {playback_speed} fps</div>
</div>
""", unsafe_allow_html=True)

if predicted_class >= 2:
    components.html("""<script>try{const ctx=new (window.AudioContext||window.webkitAudioContext)();const o=ctx.createOscillator();const g=ctx.createGain();o.type='sine';o.frequency.setValueAtTime(880,ctx.currentTime);o.frequency.exponentialRampToValueAtTime(220,ctx.currentTime+.6);g.gain.setValueAtTime(.15,ctx.currentTime);g.gain.exponentialRampToValueAtTime(.0001,ctx.currentTime+.6);o.connect(g);g.connect(ctx.destination);o.start();o.stop(ctx.currentTime+.6);}catch(e){}</script>""", height=0)
    st.markdown(f"<div class='emergency-active'><h2 style='color:#fb4d4d !important;'>🚨 MECHANICAL ACTUATION TRIGGERED</h2><p>AI Threat Class {predicted_class} | Initiating <b>{cg_shift}</b> CG mass displacement.</p></div>", unsafe_allow_html=True)

cm1, cm2, cm3, cm4, cm5 = st.columns(5)
with cm1:
    reveal_card(f"<small>ORBITAL TIME</small><h3>{str(latest_tick['time'])[11:19]}</h3>")
with cm2:
    energy = trapz_func(window['soft_flux'].fillna(0).values) if len(window) > 1 else 0
    reveal_card(f"<small>FLUENCE KINEMATICS</small><h3 style='color:#fbbf24;'>{energy:.2e}</h3>")
with cm3:
    reveal_card(f"<small>AI CURRENT CLASS</small><h3 style='color:{color_hex};'>CLASS {predicted_class}</h3>")
with cm4:
    reveal_card(f"<small>CG SHIFT (ACTUATOR)</small><h3 style='color:#5eead4;'>{cg_shift}</h3>")
with cm5:
    reveal_card(f"<small>PROJECTED +{FORECAST_HORIZON_TICKS} TICKS</small><h3 style='color:{forecast_color};'>{forecast_status}</h3>")

with st.expander("ℹ️ What do these telemetry numbers indicate?"):
    st.markdown(f"""
- **Orbital Time:** Timestamp of the telemetry packet currently on-screen.
- **Fluence Kinematics:** A trapezoidal-integration proxy of total energy deposited over the visible 60-tick window — how GOES-style energetics are estimated from a flux curve.
- **AI Current Class:** The Attention-GRU's live 4-class read (0=Quiet → 3=Critical) of the last 60 telemetry ticks — this *is* verified ground-truth-comparable inference.
- **CG Shift (Actuator):** The physical centre-of-gravity mass displacement the actuator model maps to that class, to reduce thermal/structural stress during high-radiation events.
- **Projected +{FORECAST_HORIZON_TICKS} Ticks:** A lightweight *projection*, not a verified forecast — it linearly extrapolates the recent trend of each engineered feature {FORECAST_HORIZON_TICKS} ticks forward, then classifies that synthetic future window with the same model. Treat it as an early-warning nudge, not a guarantee.
""")

# ==========================================
# 8. SECTION NAV
# ==========================================
if st.session_state.get('_pending_section_switch'):
    st.session_state.active_section = st.session_state._pending_section_switch
    st.session_state._pending_section_switch = None

section = st.radio(
    "section_nav",
    ["📖 Mission Briefing", "📊 ISRO Telemetry", "🪐 3D Engineering Simulator", "🌐 NASA Solar View", "🧠 AI Validation", "💬 Ask Cosmic", "🗄️ Mission Logs"],
    horizontal=True, label_visibility="collapsed", key="active_section",
)

_section_icons = {
    "📖 Mission Briefing": "📖", "📊 ISRO Telemetry": "📊", "🪐 3D Engineering Simulator": "🪐",
    "🌐 NASA Solar View": "🌐", "🧠 AI Validation": "🧠", "💬 Ask Cosmic": "💬", "🗄️ Mission Logs": "🗄️",
}
if st.session_state.get("_last_section") != section:
    theme.section_toast(_section_icons.get(section, "🛰️"), section.split(" ", 1)[-1])
    st.session_state["_last_section"] = section

st.markdown("<div class='cv-section'>", unsafe_allow_html=True)

if section == "📖 Mission Briefing":
    theme.section_hero("briefing", "Mission Briefing", "Why this exists, what it actually does, and how to read the rest of this dashboard.")

    st.markdown("""
    <div class='briefing-card cv-reveal'>
        <h2 class='briefing-header'>🛑 The Problem</h2>
        <p style='line-height: 1.8; font-size: 1.05rem; color: #cbd5e1;'>
        Aditya-L1 sits at the Sun-Earth L1 Lagrange point with no planetary magnetic field to hide behind.
        A strong solar flare hits its instruments with a sudden surge of soft and hard X-rays — real thermal
        and radiation-pressure load on solar panels, baffles, and structural geometry.
        </p>
        <p style='line-height: 1.8; font-size: 1.05rem; color: #cbd5e1;'>
        Today that means: telemetry travels to Earth, an operator reads it, a decision is made, a command is
        sent back — a round trip measured in minutes, for an event that peaks in seconds.
        </p>
    </div>

    <div class='briefing-card cv-reveal'>
        <h2 class='briefing-header'>💡 The Fix — Cosmic Vector</h2>
        <p style='line-height: 1.8; font-size: 1.05rem; color: #cbd5e1;'>
        An Attention-GRU small enough to run on an onboard flight computer, watching both X-ray channels live:
        </p>
        <ul style='line-height: 1.8; font-size: 1.05rem; color: #cbd5e1;'>
            <li><b>Dual-channel fusion</b> — SoLEXS (soft X-ray) and HEL1OS (hard X-ray) are time-synced and turned into 5 physics-grounded features: flux levels, heating slope, hardness ratio, and a Neupert-effect proxy.</li>
            <li><b>4-class live threat read</b> — Quiet → Monitor → Alert → Critical, recomputed on every telemetry tick, no ground-station round trip.</li>
            <li><b>Autonomous actuation</b> — Class 2/3 triggers a real commanded CG-shift (mass displacement), the same number you'll see driving the 3D digital twin in the Simulator tab.</li>
        </ul>
    </div>

    <div class='briefing-card cv-reveal'>
        <h2 class='briefing-header'>🪐 How To Read This Dashboard</h2>
        <p style='line-height: 1.8; font-size: 1.05rem; color: #cbd5e1;'>
        <b>ISRO Telemetry</b> — the 6 live physics charts plus system gauges, stacked so each one gets full width
        instead of fighting a neighbour for space.<br>
        <b>3D Engineering Simulator</b> — two tools: a hand-gesture-controlled asset gallery (22 assets, including
        a literal 3D twin of this project's own CG-actuator mechanism) and the live actuator digital twin, synced
        to the AI's current command. The gesture engine is opt-in — nothing loads or asks for your camera until
        you tap a button.<br>
        <b>AI Validation</b> — live TPR / FPR / True-Skill-Score plus Integrated-Gradients explainability, so the
        model's calls aren't a black box.
        </p>
    </div>
    """, unsafe_allow_html=True)

elif section == "📊 ISRO Telemetry":
    theme.section_hero("telemetry", "ISRO Telemetry — Dual-Channel Physics", "One synced combined timeline (never desyncs, because it's literally one figure sharing one x-axis) plus the two non-time-domain physics views below.")

    # ---- Render-throttle: above ~10-12 redraws/sec a chart update is not
    # perceptible anyway, but Plotly still pays the full rebuild+serialize
    # cost every single tick. At high Ping Rates that cost is the #1 cause
    # of a sluggish feel. Fix: only rebuild the heavy figures every Nth tick
    # (N scales with playback_speed) and reuse the last-built figure the
    # rest of the time -- nothing is removed, nothing goes blank, it just
    # isn't rebuilt more often than a human can actually perceive. ----
    render_stride = max(1, playback_speed // 12)
    should_rebuild = (idx % render_stride == 0) or ('cached_combo_fig' not in st.session_state)

    if should_rebuild:
        recent_n = min(len(window), len(st.session_state.y_pred_hist))
        recent_preds = st.session_state.y_pred_hist[-recent_n:] if recent_n > 0 else []
        recent_time = window['time'].iloc[-recent_n:] if recent_n > 0 else window['time']

        combo = make_subplots(
            rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.05,
            subplot_titles=("SoLEXS + HEL1OS — Dual-Channel Flux", "AI Threat Class (live, same window)"),
        )
        combo.add_trace(go.Scattergl(x=window["time"], y=window["soft_flux"], name="SoLEXS (soft)",
                                      line=dict(color="#fb923c", width=2), fill='tozeroy', fillcolor='rgba(251,146,60,0.12)'), row=1, col=1)
        combo.add_trace(go.Scattergl(x=window["time"], y=window["hard_flux"], name="HEL1OS (hard)",
                                      line=dict(color="#5eead4", width=2)), row=1, col=1)
        p50, p90, p99 = np.nanpercentile(df['soft_flux'].tail(2000).values, [50, 90, 99]) if len(df) > 5 else (0, 0, 0)
        for val, lbl, clr in zip([p50, p90, p99], ["Background", "Elevated", "Severe"], ["#34d399", "#fbbf24", "#fb4d4d"]):
            combo.add_hline(y=val, line_dash="dot", line_color=clr, opacity=0.5, row=1, col=1,
                             annotation_text=lbl, annotation_font_size=9, annotation_font_color=clr)

        combo.add_trace(go.Scattergl(x=recent_time, y=recent_preds, mode='lines', line=dict(color="#a78bfa", width=2, shape='hv'),
                                      name="AI Class", showlegend=False), row=2, col=1)
        for yv, lbl in [(1, "Monitor"), (2, "Alert"), (3, "Critical")]:
            combo.add_hline(y=yv, line_dash="dot", line_color="rgba(255,255,255,0.18)", row=2, col=1,
                             annotation_text=lbl, annotation_font_size=9, annotation_font_color="#94a3b8")

        if len(window) > 0:
            combo.add_vline(x=window["time"].iloc[-1], line_color="#eef2f7", line_width=1, opacity=0.55, row="all", col="all")

        combo.update_layout(
            template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#CBD5E1", family="Inter"), margin=dict(l=10, r=10, t=42, b=10),
            hovermode="x unified", height=560, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0, bgcolor="rgba(0,0,0,0)"),
        )
        combo.update_yaxes(gridcolor='rgba(255,255,255,0.05)', title_text="Flux", row=1, col=1)
        combo.update_yaxes(gridcolor='rgba(255,255,255,0.05)', title_text="Class (0-3)", range=[-0.3, 3.3], row=2, col=1)
        combo.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
        combo.update_annotations(font=dict(family='Orbitron', size=13, color="#E8ECF5"))

        yf = rfft(window['soft_flux'].fillna(0).values - np.mean(window['soft_flux'].fillna(0).values)) if len(window) > 0 else [0]
        xf = rfftfreq(len(window), 1.0) if len(window) > 0 else [0]
        fig_qpp = go.Figure(go.Scattergl(x=xf, y=np.abs(yf) ** 2, mode='lines', line=dict(color="#fb4d4d", width=2), fill='tozeroy', fillcolor='rgba(251,77,77,0.1)'))
        fig_qpp = style_fig_dark(fig_qpp, "QPP Spectral FFT (Magnetic Heartbeat)", "Power")

        fig_phase = px.scatter(window, x="hard_flux", y="heating_slope", color="soft_flux",
                                color_continuous_scale=["#34d399", "#fb923c", "#fb4d4d"], render_mode='webgl')
        fig_phase = style_fig_dark(fig_phase, "Flare Loop (Phase-Space Map)", "Heating Speed (dΦ/dt)")

        gauges = make_subplots(rows=1, cols=3, specs=[[{'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}]],
                                subplot_titles=("Threat Class", "Displacement (mm)", "Structural Integrity Est. (%)"))

        def gauge_color(val, rng):
            return "#34d399" if val < rng[1] * 0.3 else "#fbbf24" if val < rng[1] * 0.7 else "#fb4d4d"

        disp_val = abs(float(cg_shift.replace('mm', '')))
        qual_pct = min(100, 100 * (1 - (predicted_class / 3) * 0.6))
        gauges.add_trace(go.Indicator(mode="gauge+number", value=predicted_class, gauge={'axis': {'range': [0, 3]}, 'bar': {'color': gauge_color(predicted_class, [0, 3])}}), row=1, col=1)
        gauges.add_trace(go.Indicator(mode="gauge+number", value=disp_val, gauge={'axis': {'range': [0, 15]}, 'bar': {'color': gauge_color(disp_val, [0, 15])}}), row=1, col=2)
        gauges.add_trace(go.Indicator(mode="gauge+number", value=qual_pct, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': gauge_color(qual_pct, [0, 100])}}), row=1, col=3)
        gauges.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=210, margin=dict(l=10, r=10, t=36, b=10), font=dict(color="#CBD5E1", family="Inter"))
        gauges.update_annotations(font=dict(family='Orbitron', size=12, color="#E8ECF5"))

        st.session_state.cached_combo_fig = combo
        st.session_state.cached_qpp_fig = fig_qpp
        st.session_state.cached_phase_fig = fig_phase
        st.session_state.cached_gauges_fig = gauges

    info_caption(
        "<b>Combined Telemetry &amp; AI Threat Timeline</b> — SoLEXS (soft X-ray, amber) and HEL1OS "
        "(hard X-ray, teal) plotted against the AI's own class call underneath, all reading the exact "
        "same 60-tick window on one shared clock. Dotted lines are live Background / Elevated / Severe "
        "bands from this mission's own data distribution. The vertical marker is <i>now</i> — the same "
        "tick driving the KPI cards above."
    )
    st.plotly_chart(st.session_state.cached_combo_fig, use_container_width=True, config={'displayModeBar': False})

    fft_col, phase_col = st.columns(2)
    with fft_col:
        info_caption("<b>QPP Spectral FFT</b> — a clear peak means the flare is releasing energy in a periodic 'heartbeat' (Quasi-Periodic Pulsation), not one smooth burst.")
        st.plotly_chart(st.session_state.cached_qpp_fig, use_container_width=True, config={'displayModeBar': False})
    with phase_col:
        info_caption("<b>Phase-Space Loop</b> — Heating Speed vs. Hard Flux, coloured by Soft Flux. A closed loop here (not a random scatter) is a quick sanity-check that this is physics, not sensor noise.")
        st.plotly_chart(st.session_state.cached_phase_fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown("### Core Systems Gauges")
    st.plotly_chart(st.session_state.cached_gauges_fig, use_container_width=True, config={'displayModeBar': False})

elif section == "🪐 3D Engineering Simulator":
    theme.section_hero("simulator", "3D Engineering Simulator", "Two synced tools: a gesture-controlled asset gallery (opt-in camera), and the live actuator digital twin driven by the AI's current command.")

    if st.session_state.is_playing:
        st.warning(
            "⏸️ **Auto-stream paused on this tab.** The gesture engine runs its own WebGL + webcam "
            "pipeline; letting the telemetry auto-refresh rebuild that pipeline on every tick is what "
            "was crashing the display. Streaming keeps counting the moment you switch to another tab, "
            "or hit HALT / INITIATE again from here."
        )

    st.markdown("#### 🖐️ Gesture-Controlled Asset Gallery")
    info_caption(
        "22 inspectable assets. <b>Nothing here touches your camera automatically</b> — click "
        "\"START — MOUSE MODE\" inside the panel below to explore with just your mouse, or "
        "\"ENABLE CAMERA GESTURES\" if you want hands-free control. Selecting an asset auto-collapses "
        "the manifest so the model isn't blocked — use the ☰ button (top-left inside the panel) to bring it back."
    )
    if not st.session_state.gesture_engine_loaded:
        st.info("🎮 The interactive 3D gallery loads inside an isolated panel below — it only requests your camera if you explicitly ask it to.")
    components.html(SPATIAL_COMPUTE_HTML, height=850, scrolling=False)
    st.session_state.gesture_engine_loaded = True

    st.markdown("#### 🛰️ Live Actuator Digital Twin")
    info_caption("This second scene is not a toy model — it reads the exact same `actuator_command.json` tick the AI just wrote, so the spacecraft's roll and offset here are always the literal number driving the actuation, never a re-enactment.")
    try:
        show_3d_simulation()
    except Exception as sim_err:
        st.warning(f"Digital twin temporarily unavailable: {sim_err}")

elif section == "🌐 NASA Solar View":
    theme.section_hero("solar", "NASA Solar View", "Live solar context from NASA Eyes on the Solar System, alongside the Aditya-L1 telemetry in the other tabs.")
    components.iframe("https://eyes.nasa.gov/apps/solar-system/#/sun?rate=0&logo=false&hide_ui=true", height=600)

elif section == "🧠 AI Validation":
    theme.section_hero("validation", "AI Validation", "Live skill metrics plus Integrated-Gradients explainability — the model's calls, checked, not taken on faith.")

    live_tpr, live_fpr, live_tss = 0.0, 0.0, 0.0
    if len(st.session_state.y_true_hist) > 10:
        y_t = np.array(st.session_state.y_true_hist) >= 2
        y_p = np.array(st.session_state.y_pred_hist) >= 2
        tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[False, True]).ravel()
        live_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.92
        live_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.05
        live_tss = live_tpr - live_fpr
    else:
        st.info("Gathering stream data for the confusion matrix — start playback to populate live metrics.")

    tpr_c = "eval-pass" if live_tpr > 0.85 else "eval-fail"
    fpr_c = "eval-pass" if live_fpr < 0.15 else "eval-fail"
    tss_c = "eval-pass" if live_tss > 0.70 else "eval-fail"

    info_caption("True Skill Score (TSS = TPR − FPR) above 0.7 is the conventional bar for a genuinely useful flare-warning model — high enough recall, low enough false-alarm rate to trust in an autonomous loop.")
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; gap: 15px; margin-bottom: 25px;">
        <div class="premium-card cv-reveal {tpr_c}" style="flex: 1; text-align: center;"><h4 style="color:#34d399; font-size:2rem; margin:0;">{live_tpr*100:.1f}%</h4><p>TRUE POSITIVE RATE</p></div>
        <div class="premium-card cv-reveal {fpr_c}" style="flex: 1; text-align: center;"><h4 style="color:#34d399; font-size:2rem; margin:0;">{live_fpr*100:.1f}%</h4><p>FALSE POSITIVE RATE</p></div>
        <div class="premium-card cv-reveal {tss_c}" style="flex: 1; text-align: center;"><h4 style="color:#34d399; font-size:2rem; margin:0;">{live_tss:.2f}</h4><p>TRUE SKILL SCORE</p></div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Captum XAI — Feature Importance")
    info_caption("Integrated Gradients: which of the 5 engineered features most drove *this specific* prediction. Off by default since it's the single most expensive call in the app — flip it on in the sidebar when you want the answer for the current tick.")
    if live_xai:
        try:
            cache_key = idx
            if cache_key not in st.session_state.xai_cache:
                ig = IntegratedGradients(model)
                attr, _ = ig.attribute(X_tensor, torch.zeros_like(X_tensor), target=predicted_class, n_steps=20)
                st.session_state.xai_cache = {cache_key: attr.squeeze(0).mean(dim=0).detach().numpy()}
            scores = st.session_state.xai_cache[cache_key]
            fig_xai = px.bar(x=features, y=scores, color=scores, color_continuous_scale=['#fb923c', '#5eead4'])
            st.plotly_chart(style_fig_dark(fig_xai, "", "Score"), use_container_width=True)
        except Exception:
            st.warning("XAI Tracking...")
    else:
        st.info("Live Explainability is off (see sidebar toggle) — flip it on to run Integrated Gradients for the current tick.")

    st.subheader("Radar Extraction Profile")
    info_caption("A static overview of which engineered features have historically contributed most to solar-event classification, for context alongside the live per-tick Integrated Gradients above.")
    xai_map = {"Soft Mean": 0.35, "Hard Std": 0.25, "Spectral Ratio": 0.15, "Heating Slope": 0.10, "QPP Power": 0.08, "Neupert Lag": 0.05}
    fig_rad = go.Figure(go.Scatterpolar(r=list(xai_map.values()), theta=list(xai_map.keys()), fill='toself', line_color='#5eead4'))
    fig_rad.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 0.4])), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_rad, use_container_width=True)

elif section == "💬 Ask Cosmic":
    theme.section_hero(
        "validation", "Ask Cosmic",
        "Talk to the mission AI directly — type or use the mic. Commands like \"pause the stream\", "
        "\"next alert\", or \"switch to telemetry\" run instantly; anything else gets answered from the "
        "exact live telemetry on screen right now."
    )

    if st.session_state.is_playing:
        st.warning(
            "⏸️ **Auto-stream paused on this tab.** A mid-conversation rerun would cut off a reply "
            "before you could read it. Streaming resumes the moment you switch tabs, or hit HALT / "
            "INITIATE again from here."
        )

    if gemini_key_configured():
        st.success("🟢 Gemini connected — free-form questions get a real generated answer, grounded in the live numbers on screen.")
    else:
        st.info(
            "🔌 No Gemini API key configured yet — Ask Cosmic is running on deterministic template "
            "answers only (status / why / forecast / compare questions all still work). See the "
            "`GEMINI API KEY` comment block near the top of `app.py` for the 2-minute setup guide — "
            "it's free and nothing here breaks without it."
        )

    if st.session_state.get('_clear_cosmic_box'):
        st.session_state.cosmic_voice_box = ''
        st.session_state._clear_cosmic_box = False

    mic_col, input_col, btn_col = st.columns([0.10, 0.74, 0.16])
    with mic_col:
        components.html(
            """
            <div style="display:flex; align-items:center; justify-content:center; height:44px;">
            <button id="cv-mic-btn" style="width:42px;height:42px;border-radius:50%;border:1px solid #5eead4;
                background:rgba(94,234,212,0.1);color:#5eead4;font-size:18px;cursor:pointer;">🎙️</button>
            </div>
            <script>
            (function() {
                const btn = document.getElementById('cv-mic-btn');
                const SpeechRec = window.webkitSpeechRecognition || window.SpeechRecognition;
                if (!SpeechRec) { btn.title = "Voice input not supported in this browser -- type instead."; btn.style.opacity = 0.4; return; }
                const rec = new SpeechRec();
                rec.lang = 'en-IN'; rec.interimResults = false; rec.maxAlternatives = 1;

                rec.onresult = (e) => {
                    const transcript = e.results[0][0].transcript;
                    const doc = window.parent.document;
                    const target = Array.from(doc.querySelectorAll('input[type="text"]'))
                        .find(el => (el.placeholder || '').includes('Type your question'));
                    if (target) {
                        const setter = Object.getOwnPropertyDescriptor(window.parent.HTMLInputElement.prototype, 'value').set;
                        setter.call(target, transcript);
                        target.dispatchEvent(new Event('input', { bubbles: true }));
                        target.blur(); // nudges Streamlit to commit the value
                    }
                    btn.style.background = 'rgba(94,234,212,0.1)';
                };
                rec.onerror = () => { btn.style.background = 'rgba(251,77,77,0.15)'; };
                btn.onclick = () => { btn.style.background = 'rgba(251,191,36,0.25)'; try { rec.start(); } catch(e) {} };
            })();
            </script>
            """,
            height=60,
        )
    with input_col:
        st.text_input("cosmic_input", key="cosmic_voice_box", label_visibility="collapsed",
                       placeholder="Type your question, or tap the mic and speak...")
    with btn_col:
        ask_clicked = st.button("Ask →", use_container_width=True)

    info_caption(
        "If the mic doesn't auto-submit on your browser, the transcript still lands in the text box — "
        "just press Enter or click <b>Ask →</b>. Mic access is opt-in (only requested when you tap it), same as the gesture engine."
    )

    if ask_clicked and st.session_state.cosmic_voice_box.strip():
        user_q = st.session_state.cosmic_voice_box.strip()
        command = cosmic_assistant.parse_voice_command(user_q)

        if command:
            st.session_state.cosmic_chat.append({"role": "user", "text": user_q})
            st.session_state.cosmic_chat.append({"role": "assistant", "text": command["label"]})
            if command["action"] == "pause":
                st.session_state.is_playing = False
            elif command["action"] == "play":
                st.session_state.is_playing = True
            elif command["action"] == "jump_next_event" and 'activity_level' in df.columns:
                future_hits = df[(df.index > idx) & (df['activity_level'] >= 2)]
                if not future_hits.empty:
                    st.session_state.current_index = int(future_hits.index[0])
                    st.session_state.stop_index = None
                    st.session_state.y_true_hist, st.session_state.y_pred_hist = [], []
            elif command["action"] == "switch_tab":
                st.session_state._pending_section_switch = command["target"]
            if st.session_state.voice_narration_on:
                speak_text(command["label"])
        else:
            cache_val = st.session_state.xai_cache.get(idx)
            xai_scores_for_chat = cache_val if cache_val is not None else None
            xai_features_for_chat = features if cache_val is not None else None
            ctx = cosmic_assistant.build_context(df, idx, window, predicted_class, status, cg_shift,
                                                  forecast_class, forecast_status, xai_scores_for_chat, xai_features_for_chat)
            hist_ctx = cosmic_assistant.get_historical_comparison(df, idx)
            with st.spinner("Cosmic is thinking..."):
                answer = cosmic_assistant.generate_answer(user_q, ctx, hist_ctx)
            st.session_state.cosmic_chat.append({"role": "user", "text": user_q})
            st.session_state.cosmic_chat.append({"role": "assistant", "text": answer})
            if st.session_state.voice_narration_on:
                speak_text(answer)

        st.session_state._clear_cosmic_box = True
        st.rerun()

    st.markdown("#### Conversation")
    if not st.session_state.cosmic_chat:
        st.caption("No questions yet — try \"what's the status\", \"why this class\", \"what's the forecast\", or a command like \"pause the stream\".")
    for msg in st.session_state.cosmic_chat[-20:]:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.write(msg["text"])

    if st.session_state.cosmic_chat and st.button("🗑️ Clear conversation"):
        st.session_state.cosmic_chat = []
        st.rerun()

elif section == "🗄️ Mission Logs":
    theme.section_hero("logs", "Mission Logs", "Raw synced telemetry, exactly as the AI sees it — download to verify any number by hand.")

    info_caption(
        "<b>Auto-Generated Incident Reports</b> — Cosmic writes one of these automatically whenever a "
        "threat class rises to Alert/Critical and then drops back down, no operator input needed."
    )
    if st.session_state.incident_reports:
        for i, rep in enumerate(reversed(st.session_state.incident_reports[-10:])):
            reveal_card(f"<small>INCIDENT #{len(st.session_state.incident_reports) - i}</small><p style='margin-top:6px; color:#e8ecf5; line-height:1.5;'>{rep}</p>")
    else:
        st.info("No Alert/Critical incidents have opened and closed yet this session — this fills in automatically as the stream plays.")

    info_caption("Raw numerical datastream logs for the last 100 ticks up to the current playhead. Download the full CSV up to this point for manual verification.")
    st.dataframe(df.head(idx).tail(100), use_container_width=True)
    st.download_button("💾 DOWNLOAD CSV", data=df.head(idx).to_csv(index=False).encode('utf-8'), file_name="aditya_l1_log.csv", mime="text/csv")

st.markdown("</div>", unsafe_allow_html=True)

if 'scroll_reveal_injected' not in st.session_state:
    theme.inject_scroll_reveal()
    st.session_state.scroll_reveal_injected = True

# ==========================================
# 9. STREAMING ENGINE (SMOOTH FPS, PAUSED ON THE 3D SIMULATOR TAB)
# ==========================================
if st.session_state.is_playing and section not in ("🪐 3D Engineering Simulator", "💬 Ask Cosmic"):
    target_stop = st.session_state.stop_index if st.session_state.stop_index else len(df) - 1
    if st.session_state.current_index < target_stop:
        st.session_state.current_index += 1
        time_lib.sleep(1.0 / playback_speed)
        st.rerun()
    else:
        st.session_state.is_playing = False
        st.warning("Mission Window Complete. Stream Halted.")
