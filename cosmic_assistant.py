"""
cosmic_assistant.py — the "brain" behind Ask Cosmic.

Three jobs:
1. Answer free-form questions about the live mission state (status, "why did
   the AI call this", forecast, compare-to-history) using Gemini when an API
   key is configured, falling back to a deterministic template answer when
   it isn't (or the call fails) so the demo never just breaks.
2. Parse voice/text input for dashboard *commands* (pause, resume, jump to
   next alert, switch tab) so those short-circuit before ever hitting the
   LLM — instant and free.
3. Write a short incident-report narrative when an Alert/Critical event
   ends, for the Mission Logs tab.

Model name note: `MODEL_NAME` below is current as of this writing, but
Google renames/retires Gemini model IDs over time. If calls start failing
with a "model not found" error, open https://aistudio.google.com, check
the model picker there for the current flash-tier model name, and update
MODEL_NAME.
"""

import os
import numpy as np
import pandas as pd

MODEL_NAME = "gemini-2.0-flash"


def get_api_key():
    """Looks for the Gemini key in Streamlit secrets first (recommended for
    Cloud Run / cloud deploys), then falls back to an environment variable
    (handy for local `streamlit run`)."""
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")


# ==========================================
# CONTEXT PACKET — the live numbers handed to the LLM (or the template
# engine) so answers are grounded in this exact tick, not guessed.
# ==========================================
def build_context(df, idx, window, predicted_class, status, cg_shift,
                   forecast_class, forecast_status, xai_scores=None, xai_features=None):
    latest = df.iloc[idx]
    ctx = {
        "time": str(latest.get("time", "")),
        "predicted_class": predicted_class,
        "status": status,
        "cg_shift_mm": cg_shift,
        "forecast_class": forecast_class,
        "forecast_status": forecast_status,
        "soft_flux_latest": float(latest.get("soft_flux", float("nan"))),
        "hard_flux_latest": float(latest.get("hard_flux", float("nan"))),
        "hardness_ratio_latest": float(latest.get("hardness_ratio", float("nan"))),
        "soft_flux_window_mean": float(window["soft_flux"].mean()) if len(window) else None,
        "hard_flux_window_mean": float(window["hard_flux"].mean()) if len(window) else None,
    }
    if xai_scores is not None and xai_features is not None and len(xai_scores) == len(xai_features):
        pairs = sorted(zip(xai_features, [float(s) for s in xai_scores]), key=lambda p: abs(p[1]), reverse=True)
        ctx["top_xai_drivers"] = pairs[:3]
    return ctx


def get_historical_comparison(df, idx, window_size=60):
    """Grabs an earlier comparable window (a different date if the dataset
    spans more than one, otherwise the start of the same run) so
    "compare to history"-style questions have something concrete to point at."""
    try:
        if "time" not in df.columns:
            return None
        dates = df["time"].dt.date.unique()
        if len(dates) > 1:
            earlier_date = dates[0] if df["time"].iloc[idx].date() != dates[0] else dates[-1]
            slice_df = df[df["time"].dt.date == earlier_date].head(window_size)
        else:
            slice_df = df.head(window_size)
        if slice_df.empty:
            return None
        return {
            "period": str(slice_df["time"].iloc[0].date()),
            "soft_flux_mean": float(slice_df["soft_flux"].mean()),
            "hard_flux_mean": float(slice_df["hard_flux"].mean()),
            "max_activity_level": int(slice_df["activity_level"].max()) if "activity_level" in slice_df.columns else None,
        }
    except Exception:
        return None


# ==========================================
# VOICE COMMAND PARSING — deterministic, checked before the LLM
# ==========================================
def parse_voice_command(text: str):
    """Returns {'action': ..., 'label': ...} for recognized dashboard
    commands, or None if this looks like a real question instead."""
    t = text.lower().strip()

    if any(p in t for p in ["pause", "halt", "stop the stream", "stop stream"]):
        return {"action": "pause", "label": "Halting the telemetry stream."}
    if any(p in t for p in ["resume", "play", "start the stream", "start stream", "initiate stream"]):
        return {"action": "play", "label": "Resuming the telemetry stream."}
    if any(p in t for p in ["next alert", "jump to next", "next critical", "next event"]):
        return {"action": "jump_next_event", "label": "Jumping to the next Alert or Critical event."}

    tab_map = {
        "briefing": "📖 Mission Briefing",
        "telemetry": "📊 ISRO Telemetry",
        "simulator": "🪐 3D Engineering Simulator",
        "3d": "🪐 3D Engineering Simulator",
        "solar view": "🌐 NASA Solar View",
        "nasa": "🌐 NASA Solar View",
        "validation": "🧠 AI Validation",
        "logs": "🗄️ Mission Logs",
    }
    for keyword, section_name in tab_map.items():
        if f"switch to {keyword}" in t or f"open {keyword}" in t or f"go to {keyword}" in t:
            return {"action": "switch_tab", "target": section_name, "label": f"Switching to {section_name}."}

    return None


# ==========================================
# TEMPLATE ANSWERS — used when no API key is set, or Gemini errors out
# ==========================================
def template_answer(question: str, ctx: dict, history_ctx=None) -> str:
    q = question.lower()

    if any(w in q for w in ["why", "reason", "caused", "driving"]):
        if ctx.get("top_xai_drivers"):
            top = ctx["top_xai_drivers"][0]
            return (f"The current class {ctx['predicted_class']} call is driven mainly by {top[0]}, "
                    f"with a signed contribution of {top[1]:.3f} in the Integrated-Gradients breakdown. "
                    f"Soft flux is averaging {ctx['soft_flux_window_mean']:.3g} and hard flux "
                    f"{ctx['hard_flux_window_mean']:.3g} over the current window.")
        return (f"Class {ctx['predicted_class']} ({ctx['status']}) reflects soft flux around "
                f"{ctx['soft_flux_window_mean']:.3g} and hard flux around {ctx['hard_flux_window_mean']:.3g} "
                "over the last 60 ticks. Turn on Live AI Explainability in the sidebar for the exact "
                "per-feature breakdown.")

    if any(w in q for w in ["forecast", "next", "predict", "future", "going to"]):
        return (f"Current class is {ctx['predicted_class']} ({ctx['status']}). The short-horizon "
                f"projection points to class {ctx['forecast_class']} ({ctx['forecast_status']}) — "
                "that's a trend extrapolation, not a verified forecast, so treat it as an early hint.")

    if any(w in q for w in ["compare", "history", "yesterday", "earlier", "before"]):
        if history_ctx:
            return (f"Compared to the {history_ctx['period']} reference window: soft flux averaged "
                    f"{history_ctx['soft_flux_mean']:.3g} then versus {ctx['soft_flux_window_mean']:.3g} now, "
                    f"and hard flux {history_ctx['hard_flux_mean']:.3g} then versus "
                    f"{ctx['hard_flux_window_mean']:.3g} now.")
        return "I don't have a distinct earlier period to compare against in the currently loaded dataset."

    return (f"Right now: class {ctx['predicted_class']} ({ctx['status']}), commanding a "
            f"{ctx['cg_shift_mm']} CG shift. Ask me 'why', 'what's the forecast', or "
            "'compare to earlier' for more detail.")


# ==========================================
# GEMINI CALL (with graceful fallback)
# ==========================================
def generate_answer(question: str, ctx: dict, history_ctx=None) -> str:
    api_key = get_api_key()
    if not api_key:
        return template_answer(question, ctx, history_ctx)

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(MODEL_NAME)

        prompt = f"""You are Cosmic, the onboard AI copilot narrating ISRO's Aditya-L1 Cosmic Vector
dashboard. Answer the operator's question in 2-4 short sentences, grounded ONLY in the
telemetry context below. Be direct, confident, mission-control in tone. Do not invent numbers
not present in the context.

LIVE CONTEXT: {ctx}
HISTORICAL COMPARISON (if relevant): {history_ctx}

OPERATOR QUESTION: {question}
"""
        response = model.generate_content(prompt)
        text = (response.text or "").strip()
        return text if text else template_answer(question, ctx, history_ctx)
    except Exception:
        return template_answer(question, ctx, history_ctx)


def generate_incident_report(event_summary: dict) -> str:
    """Short narrative for the Mission Logs tab when an Alert/Critical
    event ends. Uses Gemini if available, otherwise a plain template."""
    api_key = get_api_key()
    base = (f"[{event_summary.get('start_time')} → {event_summary.get('end_time')}] "
            f"Peak class {event_summary.get('peak_class')} ({event_summary.get('peak_status')}). "
            f"Actuator commanded {event_summary.get('peak_cg_shift')} CG shift at peak.")
    if not api_key:
        return base

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(MODEL_NAME)
        prompt = (
            "Write a 2-sentence, professional mission-log incident summary from this data, "
            f"in the tone of a spacecraft ops log entry: {event_summary}"
        )
        response = model.generate_content(prompt)
        text = (response.text or "").strip()
        return text if text else base
    except Exception:
        return base
