# app.py — OCULAIRE Neon Lab v5 with Glaucoma Chatbot (bubble + debug fallback)
# Replace your current app.py with this file (overwrite).
# Notes:
# - If bubble doesn't open chat in your environment, click "Open Chat (debug)" (visible button).
# - Add GEMINI_API_KEY to Streamlit secrets or environment to enable generative AI responses.
# - This file preserves your RNFLT/B-scan functionality (Grad-CAM, PDF downloads).

import os
import io
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import cv2
from matplotlib.backends.backend_pdf import PdfPages
import streamlit as st

# Try to import official SDK; fallback to requests if not installed
try:
    import google.generativeai as genai
    USE_SDK = True
except Exception:
    import requests
    USE_SDK = False

# Model name used for SDK/REST
MODEL_NAME = "models/gemini-2.5-pro"

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="OCULAIRE: Neon Glaucoma Detection Dashboard",
                   layout="wide", page_icon="👁️")

# -----------------------
# Session state defaults
# -----------------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chat_open' not in st.session_state:
    st.session_state.chat_open = False
if 'chat_input' not in st.session_state:
    st.session_state.chat_input = ""

# -----------------------
# Helper: get API key
# -----------------------
def get_api_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
        if "gemini" in st.secrets and "api_key" in st.secrets["gemini"]:
            return st.secrets["gemini"]["api_key"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GENAI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or None

API_KEY = get_api_key()

# -----------------------
# Neon matplotlib styling
# -----------------------
plt.style.use('dark_background')
plt.rcParams.update({
    "figure.facecolor": "#050612",
    "axes.facecolor": "#050612",
    "axes.edgecolor": "#00f5ff",
    "axes.labelcolor": "#e6faff",
    "xtick.color": "#00f5ff",
    "ytick.color": "#ff40c4",
    "text.color": "#e6faff",
    "font.size": 12,
    "axes.titleweight": "bold",
})

# -----------------------
# CSS (neon + bubble + debug button)
# -----------------------
st.markdown("""
<style>
:root { --neonA:#00f5ff; --neonB:#ff40c4; --muted:#a4b1c9; --panel:#0a0f25; }
.stApp { background: radial-gradient(circle at 20% 20%, #091133, #020208 90%); color: #e6faff; font-family: Inter, system-ui; }
.header { text-align:center; margin-top:8px; margin-bottom:6px; }
.header h1 { font-size:40px; font-weight:900; background: linear-gradient(90deg,var(--neonA),var(--neonB)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); padding:14px; border-radius:12px; border:1px solid rgba(255,255,255,0.03); }

/* floating bubble (purely visual) */
#oculaire_chat_bubble { position: fixed; bottom: 28px; right: 28px; width:64px; height:64px; border-radius:50%; background: linear-gradient(135deg,var(--neonA),var(--neonB)); display:flex; align-items:center; justify-content:center; font-size:28px; cursor:pointer; z-index:9999; box-shadow:0 14px 40px rgba(0,0,0,0.6); color:#021617; font-weight:700; }

/* floating pill text */
#oculaire_chat_pill { position: fixed; bottom: 38px; right: 108px; padding:10px 18px; border-radius:28px; background: rgba(0,0,0,0.25); color: #e6faff; z-index:9998; font-weight:800; cursor:pointer; }

/* Visible debug button (fallback) */
#oculaire_debug_button_container { position: fixed; bottom: 26px; right: 190px; z-index:9997; }
#oculaire_debug_button { background: transparent; border: none; color: #e6faff; font-weight:800; padding:8px 12px; border-radius:8px; cursor:pointer; box-shadow: 0 8px 30px rgba(0,0,0,0.45); }

/* sidebar chat header */
.chat-header { text-align:center; font-weight:800; color:#e6faff; }

/* small chat message styles inside sidebar */
.user-msg { background: rgba(0,245,255,0.06); padding:8px; border-radius:8px; margin:6px 0; }
.assistant-msg { background: rgba(255,64,196,0.04); padding:8px; border-radius:8px; margin:6px 0; }

footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Header
# -----------------------
st.markdown("""
<div class="header">
  <h1>👁️ OCULAIRE</h1>
  <div style="color:var(--muted)">AI-Powered Glaucoma Detection Dashboard — Neon Lab v5</div>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# -----------------------
# Load models & artifacts (cached)
# -----------------------
@st.cache_resource
def load_models_and_artifacts():
    try:
        b_model = tf.keras.models.load_model("bscan_cnn.h5", compile=False)
    except Exception:
        b_model = None
    try:
        scaler = joblib.load("rnflt_scaler.joblib")
        kmeans = joblib.load("rnflt_kmeans.joblib")
        avg_healthy = np.load("avg_map_healthy.npy")
        avg_glaucoma = np.load("avg_map_glaucoma.npy")
        thin_cluster = 0 if np.nanmean(avg_healthy) > np.nanmean(avg_glaucoma) else 1
    except Exception:
        scaler = kmeans = avg_healthy = avg_glaucoma = thin_cluster = None
    return b_model, scaler, kmeans, avg_healthy, avg_glaucoma, thin_cluster

b_model, scaler, kmeans, avg_healthy, avg_glaucoma, thin_cluster = load_models_and_artifacts()

# -----------------------
# Helper functions (RNFLT/B-scan)
# -----------------------
def process_npz_file(f):
    try:
        buf = io.BytesIO(f.getvalue())
        arrs = np.load(buf, allow_pickle=True)
        key = "volume" if "volume" in arrs else arrs.files[0]
        arr = arrs[key]
        if arr.ndim == 3:
            arr = arr[0, :, :]
        vals = arr.flatten().astype(float)
        metrics = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals)),
                   "min": float(np.nanmin(vals)), "max": float(np.nanmax(vals))}
        return arr, metrics
    except Exception as e:
        st.error(f"Could not read .npz: {e}")
        return None, None

def compute_risk_map_local(rnflt_map, avg_map, threshold=-10):
    if rnflt_map.shape != avg_map.shape:
        avg_map = cv2.resize(avg_map, (rnflt_map.shape[1], rnflt_map.shape[0]))
    diff = rnflt_map - avg_map
    risk = np.where(diff < threshold, diff, np.nan)
    total = np.isfinite(diff).sum()
    risky = np.isfinite(risk).sum()
    severity = (risky / total) * 100 if total > 0 else 0.0
    return diff, risk, severity

def preprocess_bscan_image(image_pil, size=(224,224)):
    arr = np.array(image_pil.convert('L'))
    arr = np.clip(arr, 0, np.percentile(arr, 99))
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    arr_res = cv2.resize(arr, size, interpolation=cv2.INTER_NEAREST)
    arr_rgb = np.repeat(arr_res[..., None], 3, axis=-1)
    batch = np.expand_dims(arr_rgb, axis=0).astype(np.float32)
    return batch, arr_res

def gradcam_local(batch, model):
    try:
        last_conv = None
        for layer in reversed(model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                last_conv = layer.name
                break
        if last_conv is None:
            return None
        grad_model = tf.keras.models.Model(model.inputs, [model.get_layer(last_conv).output, model.output])
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(batch)
            loss = preds[:, 0]
        grads = tape.gradient(loss, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0,1,2))
        conv_out = conv_out[0]
        heat = conv_out @ pooled[..., tf.newaxis]
        heat = tf.squeeze(heat)
        heat = tf.maximum(heat, 0) / (tf.reduce_max(heat) + 1e-6)
        return heat.numpy()
    except Exception:
        return None

def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.getvalue()

def create_pdf_bytes(figs):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        for f in figs:
            pdf.savefig(f, bbox_inches='tight', facecolor=f.get_facecolor())
    buf.seek(0)
    return buf.getvalue()

def render_severity_html(pct):
    pct = float(max(0.0, min(100.0, pct)))
    html = f"""
    <div class='sev-wrap'>
      <div class='sev-outer'><div id='sev_inner' class='sev-inner' style='width:0%'></div></div>
      <div style='text-align:center'><div class='sev-chip'>{pct:.1f}%</div></div>
    </div>
    <script>
      setTimeout(function(){{
        var el = document.getElementById('sev_inner');
        if(el) el.style.width = '{pct:.1f}%';
      }}, 120);
    </script>
    """
    return html

# -----------------------
# Assistant backend
# -----------------------
def ask_glaucoma_assistant(question, history, api_key):
    if not api_key:
        return "⚠️ No Gemini API key configured. Add GEMINI_API_KEY to secrets or environment."
    system_instruction = ("You are a specialized assistant for glaucoma/OCT/RNFLT. Answer concisely and include a short 'educational only' disclaimer.")
    recent = history[-6:] if history else []
    if USE_SDK:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(MODEL_NAME)
            chat_history = []
            for msg in recent:
                role = "user" if msg["role"] == "user" else "model"
                chat_history.append({"role": role, "parts": [msg["content"]]})
            chat = model.start_chat(history=chat_history)
            response = chat.send_message(f"{system_instruction}\n\nUser question: {question}")
            return getattr(response, "text", str(response))
        except Exception as e:
            return f"⚠️ SDK error: {e}"
    else:
        try:
            conversation_context = ""
            for msg in recent:
                role = "User" if msg["role"]=="user" else "Assistant"
                conversation_context += f"{role}: {msg['content']}\n\n"
            full_prompt = f"{system_instruction}\n\n{conversation_context}User: {question}\n\nAssistant:"
            url = f"https://generativelanguage.googleapis.com/v1beta/{MODEL_NAME}:generateContent?key={api_key}"
            payload = {"contents":[{"parts":[{"text":full_prompt}]}],"generationConfig":{"temperature":0.2,"maxOutputTokens":400}}
            resp = requests.post(url, headers={"Content-Type":"application/json"}, json=payload, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            elif resp.status_code == 403:
                return "🔑 API key invalid or restricted (403)."
            else:
                return f"⚠️ REST error ({resp.status_code}): {resp.text[:200]}"
        except Exception as e:
            return f"⚠️ REST call error: {e}"

# -----------------------
# Sidebar: API info
# -----------------------
with st.sidebar:
    st.markdown("<div class='chat-header'>🔑 API Status</div>", unsafe_allow_html=True)
    if API_KEY:
        st.success("✅ Gemini API Key configured")
    else:
        st.error("❌ No API Key found")
    st.markdown("---")
    st.markdown("<div style='font-size:12px;color:var(--muted)'>Add GEMINI_API_KEY to Streamlit secrets or env var.</div>", unsafe_allow_html=True)

# -----------------------
# Visual bubble/pill HTML (bubble is purely visual & JS will try to click hidden button)
# -----------------------
st.markdown('<div id="oculaire_chat_bubble">🤖</div><div id="oculaire_chat_pill">Ask Assistant</div>', unsafe_allow_html=True)

# -----------------------
# Hidden toggle (Streamlit button off-screen) used by JS to toggle chat_open
# -----------------------
_TOGGLE_LABEL = "__OCULAIRE_TOGGLE__"
st.markdown('<div style="position:absolute; left:-9999px; top:-9999px; opacity:0;">', unsafe_allow_html=True)
toggle_pressed = st.button(_TOGGLE_LABEL, key="__oculaire_hidden_toggle__")
st.markdown('</div>', unsafe_allow_html=True)

if toggle_pressed:
    st.session_state.chat_open = not st.session_state.chat_open
    st.experimental_rerun()

# -----------------------
# Visible debug fallback button (guaranteed to work)
# -----------------------
# This is a visible Streamlit button that always toggles chat — click this if the bubble doesn't work.
# It is intentionally visible so you can test the toggle behavior immediately.
debug_clicked = st.button("Open Chat (debug)", key="__oculaire_debug_open__")
if debug_clicked:
    st.session_state.chat_open = True
    st.experimental_rerun()

# -----------------------
# JS: when bubble or pill clicked, try to press the hidden Streamlit button
# -----------------------
st.markdown(f"""
<script>
(function(){{
  const targetLabel = "{_TOGGLE_LABEL}";
  function clickHidden() {{
    // search current doc buttons (Streamlit renders buttons as <button>)
    let btns = Array.from(document.querySelectorAll('button'));
    for (let b of btns) {{
      if ((b.innerText || "").trim() === targetLabel) {{ b.click(); return true; }}
    }}
    // try parent (iframe) — may throw cross-origin error in some hosts
    try {{
      if (window.parent && window.parent.document) {{
        let pbtns = Array.from(window.parent.document.querySelectorAll('button'));
        for (let b of pbtns) {{
          if ((b.innerText || "").trim() === targetLabel) {{ b.click(); return true; }}
        }}
      }}
    }} catch(e){{ /* ignore cross-origin */ }}
    console.warn("OCULAIRE: hidden toggle button not found:", targetLabel);
    return false;
  }}

  const bubble = document.getElementById('oculaire_chat_bubble');
  const pill = document.getElementById('oculaire_chat_pill');
  [bubble, pill].forEach(el => {{
    if (!el) return;
    el.style.cursor = 'pointer';
    el.addEventListener('click', function(e) {{
      e.preventDefault();
      el.style.transform = 'scale(0.97)';
      setTimeout(()=> el.style.transform = '', 120);
      clickHidden();
    }});
  }});
}})();
</script>
""", unsafe_allow_html=True)

# -----------------------
# Layout: RNFLT / B-scan upload UI & analysis
# -----------------------
left_col, right_col = st.columns([3, 1])
with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🩺 RNFLT Map Analysis (.npz)")
    rnflt_file = st.file_uploader("Upload RNFLT .npz", type=["npz"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card" style="margin-top:12px">', unsafe_allow_html=True)
    st.subheader("👁️ B-scan Slice Analysis (image)")
    bscan_file = st.file_uploader("Upload B-scan image (jpg/png)", type=["jpg","png","jpeg"])
    st.markdown('</div>', unsafe_allow_html=True)

    threshold = st.slider("Thin-zone threshold (µm)", min_value=5, max_value=50, value=10)

with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<div style='color:var(--muted)'>Overview</div>", unsafe_allow_html=True)
    if scaler is None:
        st.markdown("<div style='color:#ff8a8a'>RNFLT artifacts: missing</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='color:#8affd6'>RNFLT artifacts: loaded</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# analysis (same logic as before)
figs_for_report = []
severity_overall = 0.0

if rnflt_file is not None:
    rnflt_map, metrics = process_npz_file(rnflt_file)
    if rnflt_map is not None and avg_healthy is not None and scaler is not None:
        X = np.array([[metrics["mean"], metrics["std"], metrics["min"], metrics["max"]]])
        Xs = scaler.transform(X)
        cluster = int(kmeans.predict(Xs)[0]) if kmeans is not None else -1
        label_r = "Glaucoma-like" if cluster == thin_cluster else "Healthy-like"
        diff, risk, sev = compute_risk_map_local(rnflt_map, avg_healthy, threshold=-threshold)
        severity_overall = max(severity_overall, sev)

        c1, c2, c3, c4 = st.columns([2,2,2,1])
        c1.markdown(f"<div style='color:var(--muted)'>Status</div><div style='font-weight:800;font-size:20px'>{'🚨' if 'Glaucoma' in label_r else '✅'} {label_r}</div>", unsafe_allow_html=True)
        c2.markdown(f"<div style='color:var(--muted)'>Mean RNFLT (µm)</div><div style='font-weight:800;font-size:20px'>{metrics['mean']:.2f}</div>", unsafe_allow_html=True)
        c3.markdown(f"<div style='color:var(--muted)'>Std Dev</div><div style='font-weight:800;font-size:20px'>{metrics['std']:.2f}</div>", unsafe_allow_html=True)
        c4.markdown(f"<div style='color:var(--muted)'>Cluster</div><div style='font-weight:800;font-size:20px'>{cluster}</div>", unsafe_allow_html=True)

        st.markdown(render_severity_html(sev), unsafe_allow_html=True)
        fig, axes = plt.subplots(1,3,figsize=(15,5), constrained_layout=True)
        axes[0].imshow(rnflt_map, cmap='turbo'); axes[0].axis('off'); axes[0].set_title("Uploaded RNFLT")
        axes[1].imshow(diff, cmap='bwr', vmin=-30, vmax=30); axes[1].axis('off'); axes[1].set_title("Difference (vs Healthy)")
        axes[2].imshow(risk, cmap='hot'); axes[2].axis('off'); axes[2].set_title("Risk Map (thinner zones)")
        fig.patch.set_facecolor("#050612")
        for ax in axes: ax.set_facecolor("#050612")
        st.pyplot(fig)
        figs_for_report.append(fig)

if bscan_file is not None and b_model is not None:
    image_pil = Image.open(bscan_file).convert("L")
    batch, proc = preprocess_bscan_image(image_pil)
    try:
        pred_raw = float(b_model.predict(batch, verbose=0)[0][0])
    except Exception:
        pred_raw = 0.0
    label_b = "Glaucoma-like" if pred_raw > 0.5 else "Healthy-like"
    conf = pred_raw*100 if label_b == "Glaucoma-like" else (1 - pred_raw)*100
    severity_overall = max(severity_overall, conf)

    st.markdown("<hr>", unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    m1.markdown(f"<div style='color:var(--muted)'>CNN Prediction</div><div style='font-weight:800;font-size:20px'>{'🚨' if 'Glaucoma' in label_b else '✅'} {label_b}</div>", unsafe_allow_html=True)
    m2.markdown(f"<div style='color:var(--muted)'>Confidence</div><div style='font-weight:800;font-size:20px'>{conf:.2f}%</div>", unsafe_allow_html=True)
    st.markdown(render_severity_html(conf), unsafe_allow_html=True)

    heat = gradcam_local(batch, b_model)
    if heat is not None:
        heat_r = cv2.resize(heat, (224,224))
        hm = (heat_r * 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        overlay = (np.stack([proc]*3, axis=-1) * 255).astype(np.uint8)
        overlay = cv2.addWeighted(overlay, 0.6, hm_color, 0.4, 0)
        st.image([image_pil, overlay], caption=["Original B-scan", "Grad-CAM Overlay"], use_column_width=True)
        fig2, ax2 = plt.subplots(1,2,figsize=(8,4)); ax2[0].imshow(image_pil, cmap='gray'); ax2[0].axis('off'); ax2[0].set_title("Original")
        ax2[1].imshow(overlay); ax2[1].axis('off'); ax2[1].set_title("Grad-CAM Overlay")
        fig2.patch.set_facecolor("#050612")
        figs_for_report.append(fig2)

# Combined severity + downloads
if (rnflt_file is not None) or (bscan_file is not None):
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center'>Overall Severity Index</h4>", unsafe_allow_html=True)
    st.markdown(render_severity_html(severity_overall), unsafe_allow_html=True)
    if figs_for_report:
        png_bytes = fig_to_png_bytes(figs_for_report[0])
        pdf_bytes = create_pdf_bytes(figs_for_report)
        st.download_button("📸 Download First Figure (PNG)", data=png_bytes, file_name="oculaire_fig.png", mime="image/png")
        st.download_button("📄 Download Full Report (PDF)", data=pdf_bytes, file_name="oculaire_report.pdf", mime="application/pdf")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:var(--muted);padding:6px;'>OCULAIRE Neon Lab v5 — For research/demo use only</div>", unsafe_allow_html=True)

# -----------------------
# Sidebar Chat UI (renders only when chat_open True)
# -----------------------
if st.session_state.chat_open:
    with st.sidebar:
        st.markdown("---")
        st.markdown("<div class='chat-header'>🤖 Glaucoma Assistant</div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center;color:var(--muted); margin-bottom:8px;'>Ask about glaucoma, RNFLT, or B-scans</div>", unsafe_allow_html=True)
        st.markdown("<div style='max-height:50vh; overflow:auto; padding:6px;'>", unsafe_allow_html=True)
        for msg in st.session_state.chat_history[-80:]:
            if msg["role"] == "user":
                st.markdown(f"<div class='user-msg'><strong>You:</strong> {msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='assistant-msg'><strong>Assistant:</strong> {msg['content']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # bound to session_state.chat_input
        q = st.text_input("Your question:", key="chat_input", placeholder="e.g., What is RNFLT?")

        col1, col2, col3 = st.columns([3,1,1])
        with col1:
            if st.button("📤 Send", use_container_width=True):
                qq = st.session_state.chat_input.strip()
                if not qq:
                    st.warning("Type a question first.")
                else:
                    st.session_state.chat_history.append({"role":"user","content":qq})
                    if not API_KEY:
                        st.session_state.chat_history.append({"role":"assistant","content":"⚠️ No Gemini API key configured. Add GEMINI_API_KEY to secrets or env var."})
                        st.session_state.chat_input = ""
                        st.experimental_rerun()
                    with st.spinner("🔍 Thinking..."):
                        reply = ask_glaucoma_assistant(qq, st.session_state.chat_history, API_KEY)
                    st.session_state.chat_history.append({"role":"assistant","content":reply})
                    st.session_state.chat_input = ""
                    st.experimental_rerun()
        with col2:
            if st.button("🗑️", use_container_width=True):
                st.session_state.chat_history = []
                st.experimental_rerun()
        with col3:
            if st.button("✖️", use_container_width=True):
                st.session_state.chat_open = False
                st.experimental_rerun()
