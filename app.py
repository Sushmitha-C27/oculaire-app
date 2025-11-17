# app.py — OCULAIRE Neon Lab v5 with beating floating pill + overlay chat (full file)
# Run: streamlit run app.py

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import io
from matplotlib.backends.backend_pdf import PdfPages
import os
import requests
import json
import time

# Try to import google.generativeai, fallback to requests
try:
    import google.generativeai as genai
    USE_SDK = True
except Exception:
    USE_SDK = False

# -----------------------
# Page Config
# -----------------------
st.set_page_config(page_title="OCULAIRE: Neon Glaucoma Detection Dashboard",
                   layout="wide",
                   page_icon="👁️")

# -----------------------
# Initialize Session State for Chat
# -----------------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Keep last processed question to avoid duplicate processing across reruns
if 'last_processed_q' not in st.session_state:
    st.session_state['last_processed_q'] = None

# -----------------------
# Get API key from Streamlit secrets or environment variable
# -----------------------
def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key:
        return env_key
    return None

API_KEY = get_api_key()

# -----------------------
# Neon Matplotlib Config
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
# CSS — Neon Theme + Animations
# -----------------------
st.markdown("""
<style>
:root {
  --bg:#020208;
  --panel:#0a0f25;
  --neonA:#00f5ff;
  --neonB:#ff40c4;
  --muted:#a4b1c9;
}
.stApp {
  background: radial-gradient(circle at 20% 20%, #091133, #020208 90%);
  color: #e6faff;
  font-family: 'Plus Jakarta Sans', Inter, system-ui;
}
.header { text-align:center; margin-top:10px; margin-bottom:10px; }
.header h1 {
  font-size:42px; font-weight:900; letter-spacing:3px;
  background: linear-gradient(90deg, var(--neonA), var(--neonB));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  text-shadow: 0 0 20px rgba(0,245,255,0.8), 0 0 35px rgba(255,64,196,0.5);
}
.header h3 { color:var(--muted); font-weight:400; font-size:15px; text-shadow: 0 0 12px rgba(255,255,255,0.2); }
.card {
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
  border:1px solid rgba(255,255,255,0.05);
  box-shadow: 0 0 25px rgba(0,245,255,0.05), 0 0 35px rgba(255,64,196,0.05);
  border-radius:12px; padding:16px;
}
.metric-label { color:var(--muted); font-size:12px; }
.large-metric { font-weight:800; font-size:22px; color:#fff; text-shadow:0 0 15px rgba(0,245,255,0.5); }

/* Severity Bar */
.sev-wrap { margin-top:16px; }
.sev-outer { height:18px; width:100%; background: rgba(255,255,255,0.05); border-radius:14px; overflow:hidden; }
.sev-inner {
  height:100%; width:0%;
  background: linear-gradient(90deg,var(--neonA),var(--neonB));
  border-radius:14px;
  box-shadow: 0 0 25px rgba(0,245,255,0.6), 0 0 25px rgba(255,64,196,0.5);
  transition: width 1s ease-in-out;
}
.sev-chip {
  margin-top:6px; display:inline-block;
  padding:6px 12px; border-radius:12px;
  font-weight:800; font-size:14px; color:#021617;
  background: linear-gradient(90deg, rgba(0,245,255,0.9), rgba(255,64,196,0.9));
  box-shadow: 0 0 20px rgba(0,245,255,0.4), 0 0 20px rgba(255,64,196,0.3);
  animation: pulse 1.8s infinite;
}
@keyframes pulse { 0%{transform:scale(1);} 50%{transform:scale(1.06);} 100%{transform:scale(1);} }
.download-btns { margin-top:14px; display:flex; gap:10px; justify-content:center; }

/* Chat message styling */
.user-msg {
  background: linear-gradient(135deg, rgba(0,245,255,0.15), rgba(0,245,255,0.05));
  border-left: 3px solid var(--neonA);
  padding: 12px;
  border-radius: 8px;
  margin: 8px 0;
}
.assistant-msg {
  background: linear-gradient(135deg, rgba(255,64,196,0.15), rgba(255,64,196,0.05));
  border-left: 3px solid var(--neonB);
  padding: 12px;
  border-radius: 8px;
  margin: 8px 0;
}
.chat-header {
  text-align: center;
  background: linear-gradient(90deg, var(--neonA), var(--neonB));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-weight: 800;
  font-size: 24px;
  margin-bottom: 20px;
  text-shadow: 0 0 20px rgba(0,245,255,0.3);
}

/* Ensure footer hidden */
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Chatbot Function
# -----------------------
def ask_glaucoma_assistant(question, history, api_key):
    """Call Google Gemini API with glaucoma-specific context"""
    if not api_key or not api_key.strip():
        return "⚠️ Please configure your Google Gemini API key (see sidebar)."
    system_instruction = """You are a specialized medical AI assistant focused exclusively on glaucoma. 

Your role:
- Answer ONLY questions related to glaucoma, eye health, OCT imaging, RNFLT measurements, optic nerve health, intraocular pressure, and glaucoma diagnosis/treatment
- Provide accurate, evidence-based information about glaucoma
- Explain medical terminology clearly
- If asked about non-glaucoma topics, politely redirect to glaucoma-related questions
- Keep responses concise and under 200 words
- Always include a brief disclaimer that you're providing educational information, not medical advice

Important: Always remind users to consult healthcare professionals for medical decisions."""
    try:
        if USE_SDK:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            chat_history = []
            for msg in history[-6:]:
                role = "user" if msg["role"] == "user" else "model"
                chat_history.append({"role": role, "parts": [msg["content"]]})
            chat = model.start_chat(history=chat_history)
            response = chat.send_message(f"{system_instruction}\n\nUser question: {question}")
            return response.text
        else:
            conversation_context = ""
            for msg in history[-6:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                conversation_context += f"{role}: {msg['content']}\n\n"
            full_prompt = f"{system_instruction}\n\n{conversation_context}User: {question}\n\nAssistant:"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": full_prompt}]}],
                    "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1000}
                },
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            elif response.status_code == 403:
                return "🔑 API key invalid. Get a new key at https://aistudio.google.com/apikey"
            elif response.status_code == 404:
                return "❌ API not accessible. Your key might be restricted. Try creating a new unrestricted key."
            else:
                return f"❌ Error ({response.status_code}): {response.text[:200]}"
    except Exception as e:
        return f"❌ Error: {str(e)}\n\nTip: Make sure your API key from https://aistudio.google.com/apikey is unrestricted."

# -----------------------
# Header
# -----------------------
st.markdown("""
<div class="header">
  <h1>👁️ OCULAIRE</h1>
  <h3>AI-Powered Glaucoma Detection Dashboard — Neon Lab v5</h3>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# -----------------------
# Load Models
# -----------------------
@st.cache_resource
def load_models():
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

b_model, scaler, kmeans, avg_healthy, avg_glaucoma, thin_cluster = load_models()

# -----------------------
# Helpers
# -----------------------
def process_npz(f):
    try:
        buf = io.BytesIO(f.getvalue())
        data = np.load(buf, allow_pickle=True)
        arr = data["volume"] if "volume" in data else data[data.files[0]]
        if arr.ndim == 3:
            arr = arr[0, :, :]
        vals = arr.flatten().astype(float)
        m = {"mean": np.nanmean(vals), "std": np.nanstd(vals), "min": np.nanmin(vals), "max": np.nanmax(vals)}
        return arr, m
    except Exception as e:
        st.error(f"Error reading NPZ: {e}")
        return None, None

def compute_risk_map(rnflt, healthy, threshold=-10):
    if rnflt.shape != healthy.shape:
        healthy = cv2.resize(healthy, (rnflt.shape[1], rnflt.shape[0]))
    diff = rnflt - healthy
    risk = np.where(diff < threshold, diff, np.nan)
    total = np.isfinite(diff).sum()
    risky = np.isfinite(risk).sum()
    severity = (risky / total) * 100 if total else 0
    return diff, risk, severity

def preprocess_bscan(image_pil, size=(224,224)):
    arr = np.array(image_pil.convert('L'))
    arr = np.clip(arr, 0, np.percentile(arr, 99))
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    arr_res = cv2.resize(arr, size, interpolation=cv2.INTER_NEAREST)
    arr_rgb = np.repeat(arr_res[..., None], 3, axis=-1)
    batch = np.expand_dims(arr_rgb, axis=0).astype(np.float32)
    return batch, arr_res

def gradcam(batch, model):
    try:
        last_conv = None
        for layer in reversed(model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                last_conv = layer.name
                break
        if not last_conv:
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

def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.getvalue()

def create_pdf(figs):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        for f in figs:
            pdf.savefig(f, bbox_inches="tight", facecolor=f.get_facecolor())
    buf.seek(0)
    return buf.getvalue()

def render_severity(pct):
    pct = max(0.0, min(100.0, float(pct)))
    html = f"""
    <div class='sev-wrap'>
      <div class='sev-outer'><div id='sev_inner' class='sev-inner'></div></div>
      <div style='text-align:center'><div class='sev-chip'>{pct:.1f}%</div></div>
    </div>
    <script>
      setTimeout(function(){{
        var el=document.getElementById('sev_inner');
        if(el) el.style.width='{pct:.1f}%';
      }},150);
    </script>
    """
    return html

# -----------------------
# SIDEBAR - API Key Status
# -----------------------
with st.sidebar:
    st.markdown("<div class='chat-header'>🔑 API Status</div>", unsafe_allow_html=True)
    if API_KEY:
        st.success("✅ Gemini API Key configured")
        st.info("Using API key from secrets/environment")
    else:
        st.error("❌ No API Key found")
        st.warning("Chatbot will not work without an API key")
    st.markdown("---")
    st.markdown("""
    <div style='font-size:12px; color:var(--muted);'>
    <strong>How to configure Gemini API key:</strong><br><br>
    <strong>For Streamlit Cloud:</strong><br>
    1. Go to your app settings<br>
    2. Add to Secrets:<br>
    <code>GEMINI_API_KEY = "your-key-here"</code><br><br>
    <strong>For Local Development:</strong><br>
    1. Create <code>.streamlit/secrets.toml</code><br>
    2. Add: <code>GEMINI_API_KEY = "your-key-here"</code><br>
    3. Or set environment variable:<br>
    <code>export GEMINI_API_KEY="your-key-here"</code><br><br>
    <strong>Get FREE API key:</strong><br>
    1. Visit <a href='https://aistudio.google.com/apikey' target='_blank'>Google AI Studio</a><br>
    2. Click "Get API Key"<br>
    3. Copy your key<br><br>
    <strong>✨ Gemini is FREE with generous limits!</strong>
    </div>
    """, unsafe_allow_html=True)

# -----------------------
# LAYOUT (RNFLT / B-Scan upload UI)
# -----------------------
colA, colB = st.columns(2)

with colA:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🩺 RNFLT Map Analysis (.npz)")
    rnflt_file = st.file_uploader("Upload RNFLT file", type=["npz"])
    st.markdown("</div>", unsafe_allow_html=True)

with colB:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("👁️ B-Scan Slice Analysis (Image)")
    bscan_file = st.file_uploader("Upload B-Scan Image", type=["jpg","png","jpeg"])
    st.markdown("</div>", unsafe_allow_html=True)

threshold = st.slider("Thin-zone threshold (µm)", 5, 50, 10)

# -----------------------
# ANALYSIS (unchanged logic)
# -----------------------
if rnflt_file or bscan_file:
    figs = []
    severity_overall = 0
    st.markdown("<hr>", unsafe_allow_html=True)

    # RNFLT Processing
    if rnflt_file and scaler is not None:
        rnflt, metrics = process_npz(rnflt_file)
        if rnflt is not None:
            X = np.array([[metrics["mean"], metrics["std"], metrics["min"], metrics["max"]]])
            Xs = scaler.transform(X)
            cluster = int(kmeans.predict(Xs)[0])
            label_r = "Glaucoma-like" if cluster == thin_cluster else "Healthy-like"
            diff, risk, sev = compute_risk_map(rnflt, avg_healthy, -threshold)
            severity_overall = max(severity_overall, sev)
            m1, m2, m3, m4 = st.columns([2,2,2,2])
            m1.markdown(f"<div class='metric-label'>Status</div><div class='large-metric'>{'🚨' if 'Glaucoma' in label_r else '✅'} {label_r}</div>", unsafe_allow_html=True)
            m2.markdown(f"<div class='metric-label'>Mean RNFLT</div><div class='large-metric'>{metrics['mean']:.2f}</div>", unsafe_allow_html=True)
            m3.markdown(f"<div class='metric-label'>Std Dev</div><div class='large-metric'>{metrics['std']:.2f}</div>", unsafe_allow_html=True)
            m4.markdown(f"<div class='metric-label'>Cluster</div><div class='large-metric'>{cluster}</div>", unsafe_allow_html=True)

            st.markdown(render_severity(sev), unsafe_allow_html=True)
            fig, axes = plt.subplots(1,3,figsize=(18,6),constrained_layout=True)
            im0=axes[0].imshow(rnflt,cmap='turbo');axes[0].axis('off');axes[0].set_title("Uploaded RNFLT")
            plt.colorbar(im0,ax=axes[0],shrink=0.85)
            im1=axes[1].imshow(diff,cmap='bwr',vmin=-30,vmax=30);axes[1].axis('off');axes[1].set_title("Difference (vs Healthy)")
            plt.colorbar(im1,ax=axes[1],shrink=0.85)
            im2=axes[2].imshow(risk,cmap='hot');axes[2].axis('off');axes[2].set_title("Risk Map")
            plt.colorbar(im2,ax=axes[2],shrink=0.85)
            fig.patch.set_facecolor("#050612")
            st.pyplot(fig)
            figs.append(fig)

    # B-Scan Processing
    if bscan_file and b_model is not None:
        image_pil = Image.open(bscan_file).convert("L")
        batch, proc = preprocess_bscan(image_pil)
        pred_raw = b_model.predict(batch, verbose=0)[0][0]
        label_b = "Glaucoma-like" if pred_raw > 0.5 else "Healthy-like"
        conf = pred_raw*100 if label_b=="Glaucoma-like" else (1-pred_raw)*100
        severity_overall = max(severity_overall, conf)

        st.markdown("<hr>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.markdown(f"<div class='metric-label'>CNN Prediction</div><div class='large-metric'>{'🚨' if 'Glaucoma' in label_b else '✅'} {label_b}</div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='metric-label'>Confidence</div><div class='large-metric'>{conf:.2f}%</div>", unsafe_allow_html=True)
        st.markdown(render_severity(conf), unsafe_allow_html=True)

        heat = gradcam(batch, b_model)
        if heat is not None:
            heat_r = cv2.resize(heat, (224,224))
            hm = (heat_r * 255).astype(np.uint8)
            hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
            overlay = (np.stack([proc]*3, axis=-1)*255).astype(np.uint8)
            overlay = cv2.addWeighted(overlay, 0.6, hm_color, 0.4, 0)
            st.image([image_pil, overlay], caption=["Original B-Scan", "Grad-CAM Overlay"], use_column_width=True)

    # Combined Severity Summary
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align:center'>Overall Severity Index</h4>", unsafe_allow_html=True)
    st.markdown(render_severity(severity_overall), unsafe_allow_html=True)

    # Download buttons
    if figs:
        png_bytes = fig_to_png(figs[0])
        pdf_bytes = create_pdf(figs)
        st.markdown("<div class='download-btns'>", unsafe_allow_html=True)
        st.download_button("📸 Download RNFLT PNG", data=png_bytes, file_name="oculaire_rnflt.png", mime="image/png")
        st.download_button("📄 Download Full Report (PDF)", data=pdf_bytes, file_name="oculaire_report.pdf", mime="application/pdf")
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:var(--muted);padding:6px;'>OCULAIRE Neon Lab v5 — For research use only</div>", unsafe_allow_html=True)

# --------------------------
# BEATING PILL + OVERLAY (client open/close, server-side processing of question param)
# --------------------------

# Read query params so server can process a submitted question once
query_params = st.experimental_get_query_params()
open_chat = query_params.get("open_chat", ["0"])[0] in ("1", "true", "True")
question_param = query_params.get("question", [None])[0]

# If question_param present and not processed before, process it and store history
if question_param:
    last_q = st.session_state.get("last_processed_q", None)
    if question_param != last_q:
        st.session_state["last_processed_q"] = question_param
        st.session_state.chat_history.append({"role": "user", "content": question_param})
        if API_KEY:
            reply = ask_glaucoma_assistant(question_param, st.session_state.chat_history, API_KEY)
        else:
            reply = "❌ No API key configured. Add GEMINI_API_KEY to secrets or environment."
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        # Keep overlay open but remove raw question param to avoid reprocessing on refresh
        st.experimental_set_query_params(open_chat="1")
        st.experimental_rerun()

# ---------- Updated floating_html with larger input + robust navigation ----------
floating_html = r'''
<div id="oculaire-float-root">
  <style>
    /* pill container */
    #oculaire-pill {
      position: fixed;
      bottom: 24px;
      right: 24px;
      z-index: 200000;
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 12px 20px;
      border-radius: 999px;
      background: linear-gradient(90deg,#00f5ff,#ff40c4);
      box-shadow: 0 12px 40px rgba(0,0,0,0.6), 0 0 40px rgba(0,245,255,0.12);
      cursor: pointer;
      transform-origin: center;
      animation: beat 1.6s infinite ease-in-out;
      -webkit-user-select: none;
      user-select: none;
    }

    @keyframes beat {
      0% { transform: scale(1); filter: drop-shadow(0 0 12px rgba(0,245,255,0.06)); }
      25% { transform: scale(1.02); filter: drop-shadow(0 0 20px rgba(0,245,255,0.12)); }
      50% { transform: scale(1.04); filter: drop-shadow(0 0 30px rgba(255,64,196,0.15)); }
      75% { transform: scale(1.02); filter: drop-shadow(0 0 20px rgba(0,245,255,0.12)); }
      100% { transform: scale(1); filter: drop-shadow(0 0 12px rgba(0,245,255,0.06)); }
    }

    /* icon circle */
    #oculaire-pill .icon {
      width: 38px; height: 38px; border-radius:50%;
      background: white; display:flex; align-items:center; justify-content:center;
      font-size: 18px; box-shadow: 0 4px 18px rgba(0,0,0,0.45);
    }

    /* pill label forced white */
    #oculaire-pill .label {
      font-weight: 800; font-size: 15px; color: white !important;
      text-shadow: 0 0 6px rgba(0,0,0,0.6);
      padding-right: 6px;
    }

    /* overlay - initially hidden */
    #oculaire-overlay {
      position: fixed;
      bottom: 100px;
      right: 24px;
      z-index: 200001;
      width: 420px;
      max-width: 92vw;
      border-radius: 14px;
      overflow: hidden;
      background: linear-gradient(180deg, rgba(10,15,37,0.98), rgba(2,2,8,0.98));
      border: 1px solid rgba(255,255,255,0.08);
      box-shadow: 0 20px 60px rgba(0,0,0,0.6);
      display: none;
    }

    #oculaire-overlay .hdr {
      padding: 12px 14px; display:flex; justify-content:space-between; align-items:center;
      background: rgba(255,255,255,0.02);
    }
    #oculaire-overlay .hdr .title { color: white; font-weight:800; font-size:16px; }
    #oculaire-overlay .hdr .close-btn {
      background:none; border:none; font-size:18px; color:white; cursor:pointer;
    }

    #oculaire-overlay .body {
      padding: 16px 16px; max-height: 300px; overflow-y: auto; color: white;
      background: transparent;
    }

    /* == BIGGER input == */
    #oculaire-overlay .footer {
      padding: 12px 12px; display:flex; gap:8px; align-items:center; background: rgba(255,255,255,0.01);
    }

    #oculaire-overlay input[type="text"] {
      width: 100%;
      padding:14px 14px;
      border-radius:10px;
      border:1px solid rgba(255,255,255,0.06);
      background: rgba(255,255,255,0.02);
      color:white;
      outline:none;
      font-size:15px;           /* larger font */
      height:44px;              /* larger clickable area */
      box-sizing: border-box;
    }

    .oculaire-btn {
      padding: 10px 14px; border-radius:10px; border:none; cursor:pointer; font-weight:800; font-size:14px;
    }
    .oculaire-btn.send { background: linear-gradient(90deg,#00f5ff,#ff40c4); color:#021617; }
    .oculaire-btn.clear { background:transparent; color:white; border:1px solid rgba(255,255,255,0.06); }

    @media (max-width: 480px) {
      #oculaire-overlay { right: 8px; left: 8px; width: auto; bottom: 90px; }
      #oculaire-pill { right: 12px; bottom: 12px; padding:10px 14px; }
    }

  </style>

  <!-- PILL -->
  <div id="oculaire-pill" title="Ask OCULAIRE" role="button" tabindex="0">
    <div class="icon">💬</div>
    <div class="label">Ask OCULAIRE</div>
  </div>

  <!-- OVERLAY -->
  <div id="oculaire-overlay" role="dialog" aria-label="OCULAIRE chat">
    <div class="hdr">
      <div class="title">🤖 OCULAIRE Assistant</div>
      <button class="close-btn" aria-label="Close chat">✖</button>
    </div>

    <div class="body" id="oculaire-body">
      <div id="oculaire-messages">
        <div style="opacity:0.9; font-size:14px; margin-bottom:10px;">Ask me about glaucoma, OCT, RNFLT...</div>
      </div>
    </div>

    <div class="footer">
      <input id="oculaire-input" type="text" placeholder="Type your question here..." aria-label="Chat input">
      <button class="oculaire-btn send" id="oculaire-send">Send</button>
      <button class="oculaire-btn clear" id="oculaire-clear">Clear</button>
    </div>
  </div>

  <script>
    (function(){
      const pill = document.getElementById('oculaire-pill');
      const overlay = document.getElementById('oculaire-overlay');
      const closeBtn = document.querySelector('#oculaire-overlay .close-btn');
      const sendBtn = document.getElementById('oculaire-send');
      const clearBtn = document.getElementById('oculaire-clear');
      const input = document.getElementById('oculaire-input');

      // enable keyboard activation for accessibility
      pill.addEventListener('keydown', function(e){
        if(e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pill.click(); }
      });

      // Show overlay if open_chat or question present on load
      try {
        const params = new URLSearchParams(window.location.search);
        if (params.get('open_chat') === '1') {
          overlay.style.display = 'block';
        }
        if (params.get('question')) {
          overlay.style.display = 'block';
          input.value = params.get('question');
        }
      } catch(e){ console.warn('oculaire: URL params error', e); }

      // open overlay immediately (no reload)
      pill.addEventListener('click', function(e){
        overlay.style.display = 'block';
        setTimeout(()=> input.focus(), 120);
      });

      // close overlay (no reload)
      closeBtn.addEventListener('click', function(e){
        overlay.style.display = 'none';
      });

      // clear just hides and clears input
      clearBtn.addEventListener('click', function(e){
        input.value = '';
        overlay.style.display = 'none';
      });

      // Send: robust navigation with fallbacks + console logs for debugging
      sendBtn.addEventListener('click', function(e){
        const q = (input.value || '').trim();
        if (!q) { input.focus(); return; }
        try {
          const url = new URL(window.location.href);
          url.searchParams.set('open_chat','1');
          url.searchParams.set('question', q);
          console.log('oculaire: navigating to', url.toString());
          // Primary approach
          window.location.href = url.toString();
          // Fallbacks (in case some hosts block direct assignment)
          setTimeout(()=> {
            try { window.location.assign(url.toString()); } catch(e) { console.warn(e); }
          }, 150);
          setTimeout(()=> {
            try { window.open(url.toString(), '_self'); } catch(e) { console.warn(e); }
          }, 300);
        } catch (err) {
          console.warn('oculaire send fallback', err);
          window.location.search = '?open_chat=1&question=' + encodeURIComponent(q);
        }
      });

      // Enter to send
      input.addEventListener('keydown', function(e){
        if (e.key === 'Enter') {
          e.preventDefault();
          sendBtn.click();
        }
      });
    })();
  </script>
</div>
'''

# render with enough height to avoid clipping
components.html(floating_html, height=260)

# End of file
