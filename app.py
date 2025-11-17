import streamlit as st
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
import sqlite3
from datetime import datetime

# Try to import google.generativeai, fallback to requests
try:
    import google.generativeai as genai
    USE_SDK = True
except Exception:
    USE_SDK = False

# -----------------------
# Page Config
# -----------------------
st.markdown("""
<div class="header" style="text-align:center; margin-top:20px; margin-bottom:20px; animation: fadeIn 1.2s ease-out;">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/Human_eye_icon.svg/1024px-Human_eye_icon.svg.png" width="90" style="filter: drop-shadow(0 0 12px rgba(0,245,255,0.6)); margin-bottom:10px;" />
  <h1 style="font-size:56px; font-weight:900; letter-spacing:4px; background: linear-gradient(90deg, #00f5ff, #ff40c4); -webkit-background-clip:text; -webkit-text-fill-color:transparent; text-shadow: 0 0 35px rgba(0,245,255,0.9), 0 0 45px rgba(255,64,196,0.8); animation: glowPulse 2.5s infinite ease-in-out;">
    OCULAIRE
  </h1>
  <h2 style="color:#a4b1c9; font-weight:500; margin-top:-10px; font-size:20px;">Illuminating Vision. Detecting Glaucoma.</h2>
  <h3 style="color:#7fa6ff; font-weight:400; font-size:16px;">AI-Powered Glaucoma Detection Dashboard — Neon Lab v5</h3>
</div>
<hr>
<style>
@keyframes glowPulse {
  0% { text-shadow: 0 0 25px rgba(0,245,255,0.5), 0 0 35px rgba(255,64,196,0.4); transform: scale(1); }
  50% { text-shadow: 0 0 45px rgba(0,245,255,1), 0 0 60px rgba(255,64,196,0.8); transform: scale(1.03); }
  100% { text-shadow: 0 0 25px rgba(0,245,255,0.5), 0 0 35px rgba(255,64,196,0.4); transform: scale(1); }
}
@keyframes fadeIn {
  from { opacity:0; transform: translateY(-20px); }
  to { opacity:1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# -----------------------
st.set_page_config(page_title="OCULAIRE: Neon Glaucoma Detection Dashboard",
                   layout="wide",
                   page_icon="👁️")

# -----------------------
# Session State
# -----------------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_processed_q' not in st.session_state:
    st.session_state.last_processed_q = None

# -----------------------
# API Key helper
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
# Matplotlib / Theme
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
# CSS — full-width patch + neon severity bar + other styles
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

/* ------------------------------------------------------------------
   CRITICAL: override Streamlit's block-container sizing so our
   severity bar can span the page. This lets .sev-outer be full width.
   ------------------------------------------------------------------ */
.block-container {
  max-width: 98% !important;
  padding-left: 1rem !important;
  padding-right: 1rem !important;
}

/* general app */
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
.header h3 { color:var(--muted); font-weight:400; font-size:15px; }

/* card */
.card {
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
  border:1px solid rgba(255,255,255,0.05);
  box-shadow: 0 0 25px rgba(0,245,255,0.05), 0 0 35px rgba(255,64,196,0.05);
  border-radius:12px; padding:16px;
}
.metric-label { color:var(--muted); font-size:12px; }
.large-metric { font-weight:800; font-size:22px; color:#fff; text-shadow:0 0 15px rgba(0,245,255,0.5); }

/* ===========================================
   SEVERITY BAR (FULL WIDTH + NEON BEAT)
   =========================================== */
.sev-wrap {
  margin-top: 18px;
  width: 100%;
  display:flex;
  align-items:center;
  justify-content:center;
  flex-direction:column;
}
.sev-outer {
  height: 26px;
  width: 100% !important;         /* allow full-page width via .block-container override */
  background: rgba(255,255,255,0.03);
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,0.05);
  overflow: hidden;
  min-width: 280px;
  max-width: 1400px;
}
.sev-inner {
  height: 100%;
  width: 0%;
  background: linear-gradient(90deg, var(--neonA), var(--neonB));
  border-radius: 18px;
  transition: width 0.9s cubic-bezier(.22,.9,.04,1);
  animation: sev-beat 0.9s infinite ease-in-out;
  box-shadow: 0 0 35px rgba(0,245,255,0.7), 0 0 45px rgba(255,64,196,0.6);
  transform-origin: left center;
}
@keyframes sev-beat {
  0% { transform: scaleX(1) }
  50% { transform: scaleX(1.02) }
  100% { transform: scaleX(1) }
}

/* Chip (percentage) — floats below center by default */
.sev-chip {
  margin-top: 10px;
  padding: 8px 16px;
  font-size: 15px;
  font-weight: 900;
  border-radius: 20px;
  background: linear-gradient(90deg, var(--neonA), var(--neonB));
  color: #021617;
  box-shadow: 0 6px 30px rgba(0,245,255,0.25), 0 6px 30px rgba(255,64,196,0.18);
  animation: chip-beat 0.9s infinite ease-in-out;
}
@keyframes chip-beat {
  0% { transform: translateY(0) scale(1) }
  50% { transform: translateY(-4px) scale(1.02) }
  100% { transform: translateY(0) scale(1) }
}

/* small screens adjustments */
@media (max-width: 700px) {
  .sev-outer { width: 92% !important; max-width: 720px; }
  .sev-chip { font-size: 13px; padding: 7px 12px; }
}

/* chat & expander styles (kept neon-friendly) */
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

/* floating expander neon visuals */
.floating-expander details {
  background: linear-gradient(180deg, rgba(10,15,37,0.98), rgba(2,2,8,0.98)) !important;
  border: 2px solid rgba(0,245,255,0.35) !important;
  border-radius: 16px !important;
  box-shadow: 0 20px 60px rgba(0,0,0,0.6);
}
.floating-expander details summary {
  background: linear-gradient(135deg, rgba(0,245,255,0.2), rgba(255,64,196,0.2)) !important;
  padding: 12px !important;
  border-radius: 12px !important;
  font-weight: 800 !important;
  font-size: 16px !important;
  color: #e6faff !important;
  display:flex !important;
  align-items:center !important;
  gap:8px !important;
}
.floating-expander details summary::before {
  content: "💬";
  font-size: 22px;
  margin-right:8px;
}

/* hide footer */
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Model selectors / name
# -----------------------
MODEL_NAME = "models/gemini-2.5-flash"

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
            model = genai.GenerativeModel(MODEL_NAME)
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
            url = f"https://generativelanguage.googleapis.com/v1beta/{MODEL_NAME}:generateContent?key={api_key}"
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
        return f"❌ Error: {str(e)}"

# -----------------------
# Load Models (cache)
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
# Helpers (processing)
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

def create_pdf(figs, metadata=None):
    # Create a multipage PDF in-memory using matplotlib PdfPages.
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # cover page with metadata
        fig = plt.figure(figsize=(8.5,11))
        fig.patch.set_facecolor('#050612')
        title = metadata.get('title','OCULAIRE Report') if metadata else 'OCULAIRE Report'
        subtitle = metadata.get('subtitle','Glaucoma analysis') if metadata else 'Glaucoma analysis'
        patient = metadata.get('patient','-') if metadata else '-'
        pid = metadata.get('patient_id','-') if metadata else '-'
        ts = metadata.get('timestamp', datetime.utcnow().isoformat()) if metadata else datetime.utcnow().isoformat()
        txt = f"{title}\n\n{subtitle}\n\nPatient: {patient} (ID: {pid})\nTimestamp: {ts}\n\nMetrics:\n"
        metrics = metadata.get('metrics',{}) if metadata else {}
        for k,v in metrics.items():
            txt += f"{k}: {v}\n"
        txt += f"\nOverall severity: {metadata.get('severity',0):.2f}%\n\nFor research use only. This is not clinical advice."
        fig.text(0.05, 0.95, txt, va='top', wrap=True, fontsize=10, color='white')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # add provided figures
        for f in figs:
            pdf.savefig(f, bbox_inches='tight', facecolor=f.get_facecolor())
            plt.close(f)
    buf.seek(0)
    return buf.getvalue()

# -----------------------
# Database (SQLite) helpers for persistent history
# -----------------------
DB_PATH = os.path.join('.', 'oculaire_runs.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient TEXT,
            patient_id TEXT,
            timestamp TEXT,
            metrics TEXT,
            severity REAL,
            pdf BLOB
        )
    ''')
    conn.commit()
    conn.close()

def save_run(patient, patient_id, metrics, severity, pdf_bytes):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO runs (patient, patient_id, timestamp, metrics, severity, pdf) VALUES (?,?,?,?,?,?)',
              (patient, patient_id, datetime.utcnow().isoformat(), json.dumps(metrics), float(severity), sqlite3.Binary(pdf_bytes)))
    conn.commit()
    conn.close()

def list_runs(limit=20):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, patient, patient_id, timestamp, metrics, severity FROM runs ORDER BY id DESC LIMIT ?', (limit,))
    rows = c.fetchall()
    conn.close()
    return rows

def load_run_pdf(run_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT pdf FROM runs WHERE id=?', (run_id,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

# init DB on start
init_db()

# -----------------------
# Severity renderer (HTML + JS sets width)
# -----------------------
def render_severity(pct):
    pct = max(0.0, min(100.0, float(pct)))
    html = f"""
    <div class='sev-wrap'>
      <div class='sev-outer'><div id='sev_inner' class='sev-inner'></div></div>
      <div class='sev-chip'>{pct:.1f}%</div>
    </div>
    <script>
      // small delay to ensure layout ready, then set width
      setTimeout(function(){{
        var el = document.getElementById('sev_inner');
        if (el) el.style.width = '{pct:.1f}%';
      }}, 60);
    </script>
    """
    return html

# -----------------------
# Sidebar (API status + RNFLT/B-scan input mode & converters + patient quick fields)
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
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Patient (optional)")
    patient_name = st.text_input("Patient name", key="patient_name")
    patient_id = st.text_input("Patient ID", key="patient_id")

    st.markdown("---")
    st.subheader("RNFLT Input & Tools")
    rnflt_input_mode = st.radio("RNFLT input type", ["NPZ (recommended)", "Image (single RNFLT image)"])
    st.markdown("Image → NPZ converter (RNFLT slices)")
    rnflt_conv_files = st.file_uploader("Upload RNFLT slices (PNG/JPG) for NPZ", accept_multiple_files=True, type=["png","jpg","jpeg"], key="rnflt_conv")
    if rnflt_conv_files:
        if st.button("Convert RNFLT images → .npz (download)", key="conv_rnflt"):
            try:
                stacks = []
                for f in rnflt_conv_files:
                    im = Image.open(f).convert("L")
                    arr = np.array(im).astype(np.float32)
                    stacks.append(arr)
                vol = np.stack(stacks, axis=0)
                buf = io.BytesIO()
                np.savez_compressed(buf, volume=vol)
                buf.seek(0)
                st.success(f"Packed {len(stacks)} slices into volume {vol.shape}")
                st.download_button("⬇️ Download RNFLT .npz", data=buf.getvalue(), file_name="rnflt_volume.npz", mime="application/octet-stream")
            except Exception as e:
                st.error(f"Conversion error: {e}")

    st.markdown("---")
    st.subheader("B-Scan Input & Tools")
    bscan_input_mode = st.radio("B-scan input type", ["Image (recommended)", "NPZ (sequence of B-scan slices)"])
    st.markdown("Image → NPZ converter (B-scan slices)")
    bscan_conv_files = st.file_uploader("Upload B-scan slices (PNG/JPG) for NPZ", accept_multiple_files=True, type=["png","jpg","jpeg"], key="bscan_conv")
    if bscan_conv_files:
        if st.button("Convert B-scan images → .npz (download)", key="conv_bscan"):
            try:
                stacks = []
                for f in bscan_conv_files:
                    im = Image.open(f).convert("L")
                    arr = np.array(im).astype(np.float32)
                    stacks.append(arr)
                vol = np.stack(stacks, axis=0)
                buf = io.BytesIO()
                np.savez_compressed(buf, volume=vol)
                buf.seek(0)
                st.success(f"Packed {len(stacks)} slices into volume {vol.shape}")
                st.download_button("⬇️ Download B-scan .npz", data=buf.getvalue(), file_name="bscan_volume.npz", mime="application/octet-stream")
            except Exception as e:
                st.error(f"Conversion error: {e}")

    st.markdown("---")
    st.markdown("⚠️ Recommended: RNFLT as NPZ for full maps. B-scan works as an image or a sequence.")
# end sidebar

# -----------------------
# Main UI layout (uploads) — follow chosen input modes
# -----------------------
colA, colB = st.columns(2)
with colA:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🩺 RNFLT Map Analysis")
    if rnflt_input_mode == "NPZ (recommended)":
        rnflt_file = st.file_uploader("Upload RNFLT file (.npz)", type=["npz"], key="rnflt_npz")
        rnflt_arr = None
        rnflt_metrics = None
        if rnflt_file:
            rnflt_arr, rnflt_metrics = process_npz(rnflt_file)
    else:
        rnflt_img = st.file_uploader("Upload RNFLT image (single) (png/jpg)", type=["png","jpg","jpeg"], key="rnflt_img")
        rnflt_arr = None
        rnflt_metrics = None
        rnflt_pil = None
        if rnflt_img:
            try:
                pil = Image.open(rnflt_img).convert("L")
                rr = np.array(pil).astype(float)
                vals = rr.flatten()
                rnflt_metrics = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals)), "min": float(np.nanmin(vals)), "max": float(np.nanmax(vals))}
                rnflt_arr = rr
                rnflt_pil = pil
            except Exception as e:
                st.error(f"RNFLT image read error: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

with colB:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("👁️ B-Scan Slice Analysis")
    if bscan_input_mode == "Image (recommended)":
        bscan_file = st.file_uploader("Upload B-Scan Image (jpg/png)", type=["jpg","png","jpeg"], key="bscan_img")
        bscan_npz_file = None
    else:
        # NPZ mode for B-scan (sequence/volume)
        bscan_npz_file = st.file_uploader("Upload B-scan volume (.npz)", type=["npz"], key="bscan_npz")
        bscan_file = None
    st.markdown("</div>", unsafe_allow_html=True)

threshold = st.slider("Thin-zone threshold (µm)", 5, 50, 10)

# -----------------------
# Analysis logic (unchanged except it uses rnflt_arr / bscan_file or bscan_npz_file)
# -----------------------
if (('rnflt_arr' in locals() and rnflt_arr is not None) or rnflt_file or (bscan_file is not None) or (bscan_npz_file is not None)):
    figs = []
    severity_overall = 0
    st.markdown("<hr>", unsafe_allow_html=True)

    # RNFLT Processing
    if 'rnflt_arr' in locals() and rnflt_arr is not None:
        # if we have metrics from earlier
        try:
            metrics = rnflt_metrics
            X = np.array([[metrics["mean"], metrics["std"], metrics["min"], metrics["max"]]])
            if scaler is not None and kmeans is not None:
                Xs = scaler.transform(X)
                cluster = int(kmeans.predict(Xs)[0])
                label_r = "Glaucoma-like" if cluster == thin_cluster else "Healthy-like"
            else:
                cluster = "?"
                label_r = "Unknown (no model)"
            if avg_healthy is not None:
                diff, risk, sev = compute_risk_map(rnflt_arr, avg_healthy, -threshold)
            else:
                diff = rnflt_arr - np.nanmean(rnflt_arr)
                risk = np.where(diff < -threshold, diff, np.nan)
                sev = np.nanpercentile(np.nan_to_num(diff), 75)
            severity_overall = max(severity_overall, sev)
            m1, m2, m3, m4 = st.columns([2,2,2,2])
            m1.markdown(f"<div class='metric-label'>Status</div><div class='large-metric'>{'🚨' if 'Glaucoma' in label_r else '✅'} {label_r}</div>", unsafe_allow_html=True)
            m2.markdown(f"<div class='metric-label'>Mean RNFLT</div><div class='large-metric'>{metrics['mean']:.2f}</div>", unsafe_allow_html=True)
            m3.markdown(f"<div class='metric-label'>Std Dev</div><div class='large-metric'>{metrics['std']:.2f}</div>", unsafe_allow_html=True)
            m4.markdown(f"<div class='metric-label'>Cluster</div><div class='large-metric'>{cluster}</div>", unsafe_allow_html=True)
            st.markdown(render_severity(sev), unsafe_allow_html=True)

            fig, axes = plt.subplots(1,3,figsize=(18,6),constrained_layout=True)
            im0=axes[0].imshow(rnflt_arr,cmap='turbo');axes[0].axis('off');axes[0].set_title("Uploaded RNFLT")
            plt.colorbar(im0,ax=axes[0],shrink=0.85)
            im1=axes[1].imshow(diff,cmap='bwr',vmin=-30,vmax=30);axes[1].axis('off');axes[1].set_title("Difference (vs Healthy)")
            plt.colorbar(im1,ax=axes[1],shrink=0.85)
            im2=axes[2].imshow(risk,cmap='hot');axes[2].axis('off');axes[2].set_title("Risk Map")
            plt.colorbar(im2,ax=axes[2],shrink=0.85)
            fig.patch.set_facecolor("#050612")
            st.pyplot(fig)
            figs.append(fig)
        except Exception as e:
            st.error(f"Error in RNFLT section: {e}")

    # RNFLT NPZ uploader fallback (older code path)
    if rnflt_file and ('rnflt_arr' not in locals() or rnflt_arr is None):
        if scaler is not None:
            rnflt, metrics = process_npz(rnflt_file)
            if rnflt is not None:
                try:
                    X = np.array([[metrics["mean"], metrics["std"], metrics["min"], metrics["max"]]])
                    Xs = scaler.transform(X)
                    cluster = int(kmeans.predict(Xs)[0])
                    label_r = "Glaucoma-like" if cluster == thin_cluster else "Healthy-like"
                except Exception:
                    cluster = "?"
                    label_r = "Unknown"
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
    # If user uploaded an NPZ for B-scan (sequence), take first slice as representative
    if 'bscan_npz_file' in locals() and bscan_npz_file is not None:
        try:
            bscan_vol, _ = process_npz(bscan_npz_file)
            if bscan_vol is not None:
                # bscan_vol already flattened to 2D by process_npz if 3D
                image_pil = Image.fromarray(np.uint8(255 * (bscan_vol - np.nanmin(bscan_vol)) / (np.nanmax(bscan_vol) - np.nanmin(bscan_vol) + 1e-9)))
                batch, proc = preprocess_bscan(image_pil)
                if b_model is not None:
                    pred_raw = b_model.predict(batch, verbose=0)[0][0]
                    label_b = "Glaucoma-like" if pred_raw > 0.5 else "Healthy-like"
                    conf = pred_raw*100 if label_b=="Glaucoma-like" else (1-pred_raw)*100
                else:
                    label_b = "Unknown"
                    conf = 0.0
                severity_overall = max(severity_overall, conf)
                st.markdown("<hr>", unsafe_allow_html=True)
                m1, m2 = st.columns(2)
                m1.markdown(f"<div class='metric-label'>CNN Prediction</div><div class='large-metric'>{'🚨' if 'Glaucoma' in label_b else '✅'} {label_b}</div>", unsafe_allow_html=True)
                m2.markdown(f"<div class='metric-label'>Confidence</div><div class='large-metric'>{conf:.2f}%</div>", unsafe_allow_html=True)
                st.markdown(render_severity(conf), unsafe_allow_html=True)
                st.image([image_pil,], caption=["B-scan slice (from volume)"], use_column_width=True)
        except Exception as e:
            st.error(f"B-scan NPZ read error: {e}")

    # B-scan as single image (recommended)
    if bscan_file:
        try:
            image_pil = Image.open(bscan_file).convert("L")
            batch, proc = preprocess_bscan(image_pil)
            try:
                pred_raw = b_model.predict(batch, verbose=0)[0][0] if b_model is not None else 0.0
                label_b = "Glaucoma-like" if pred_raw > 0.5 else "Healthy-like"
                conf = pred_raw*100 if label_b=="Glaucoma-like" else (1-pred_raw)*100
            except Exception:
                pred_raw = 0.0
                label_b = "Unknown"
                conf = 0.0

            severity_overall = max(severity_overall, conf)

            st.markdown("<hr>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.markdown(f"<div class='metric-label'>CNN Prediction</div><div class='large-metric'>{'🚨' if 'Glaucoma' in label_b else '✅'} {label_b}</div>", unsafe_allow_html=True)
            m2.markdown(f"<div class='metric-label'>Confidence</div><div class='large-metric'>{conf:.2f}%</div>", unsafe_allow_html=True)
            st.markdown(render_severity(conf), unsafe_allow_html=True)

            heat = gradcam(batch, b_model) if b_model is not None else None
            if heat is not None:
                heat_r = cv2.resize(heat, (224,224))
                hm = (heat_r * 255).astype(np.uint8)
                hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
                overlay = (np.stack([proc]*3, axis=-1)*255).astype(np.uint8)
                overlay = cv2.addWeighted(overlay, 0.6, hm_color, 0.4, 0)
                st.image([image_pil, overlay], caption=["Original B-Scan", "Grad-CAM Overlay"], use_column_width=True)
            else:
                st.image(image_pil, caption="Original B-Scan", use_column_width=True)
        except Exception as e:
            st.error(f"B-scan error: {e}")

    # Combined Severity Summary
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align:center'>Overall Severity Index</h4>", unsafe_allow_html=True)
    st.markdown(render_severity(severity_overall), unsafe_allow_html=True)

    # Downloads + Persistent Save
    if figs:
        png_bytes = fig_to_png(figs[0])
        pdf_bytes = create_pdf(figs, metadata={
            'title': 'OCULAIRE Report',
            'subtitle': 'Glaucoma analysis',
            'patient': patient_name or '-',
            'patient_id': patient_id or '-',
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': rnflt_metrics or {},
            'severity': float(severity_overall)
        })
        st.markdown("<div class='download-btns'>", unsafe_allow_html=True)
        st.download_button("📸 Download RNFLT PNG", data=png_bytes, file_name="oculaire_rnflt.png", mime="image/png")
        st.download_button("📄 Download Full Report (PDF)", data=pdf_bytes, file_name="oculaire_report.pdf", mime="application/pdf")

        # Save run button (persist to SQLite)
        if st.button("💾 Save run to history & store PDF"):
            try:
                metrics_to_store = rnflt_metrics or {}
                save_run(patient_name or '-', patient_id or '-', metrics_to_store, severity_overall, pdf_bytes)
                st.success("Saved run to local history (SQLite). You can view it in 'Saved Runs' below.")
            except Exception as e:
                st.error(f"Error saving run: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # If user saved runs, show list and allow download
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Saved Runs (recent)")
    runs = list_runs(10)
    if runs:
        for r in runs:
            rid, rpatient, rpid, rts, rmetrics, rsev = r
            cols = st.columns([3,1,1])
            with cols[0]:
                st.markdown(f"**{rpatient}** (ID: {rpid}) — {rts}")
                st.markdown(f"Severity: {rsev:.2f}% — Metrics: {rmetrics}")
            with cols[1]:
                if st.button(f"⬇️ Download PDF #{rid}", key=f"dl_{rid}"):
                    pdfb = load_run_pdf(rid)
                    if pdfb:
                        st.download_button(f"Download run {rid}", data=pdfb, file_name=f"oculaire_run_{rid}.pdf", mime="application/pdf")
            with cols[2]:
                if st.button(f"🗑️ Delete #{rid}", key=f"del_{rid}"):
                    # quick delete (not implemented fully) — simple UX: inform user to clear DB manually for now
                    st.warning("Delete not implemented in this demo. To remove rows, delete oculaire_runs.db or run cleanup script.")
    else:
        st.info("No saved runs yet. Save a run after analysis using the 'Save run to history' button.")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:var(--muted);padding:6px;'>OCULAIRE Neon Lab v5 — For research use only</div>", unsafe_allow_html=True)

# -----------------------
# Floating expander (Streamlit-only UI) — behaves like expander but neon
# -----------------------
st.markdown('<div class="floating-expander">', unsafe_allow_html=True)
with st.expander("💬 Ask AI assistant", expanded=False):
    st.markdown("<div class='chat-header'>🤖 Glaucoma Q&A Assistant</div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:var(--muted); font-size:13px; margin-bottom:12px;'>Ask me anything about glaucoma, OCT imaging, RNFLT, or eye health!</p>", unsafe_allow_html=True)

    # display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-msg'><strong>You:</strong> {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-msg'><strong>🤖:</strong> {msg['content']}</div>", unsafe_allow_html=True)

    # input area
    user_question = st.text_input("Your question:", key="chat_input",
                                  placeholder="e.g., What is glaucoma? How does OCT detect it?",
                                  label_visibility="collapsed", help="Type your glaucoma or OCT question here")

    col1, col2 = st.columns([4,1])
    with col1:
        send_btn = st.button("📤 Send", use_container_width=True)
    with col2:
        clear_btn = st.button("🗑️", use_container_width=True)

    if send_btn and user_question:
        if not API_KEY:
            st.error("❌ API key not configured. See sidebar.")
        else:
            with st.spinner("🔍 Searching for answers..."):
                # append user query and call assistant
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                reply = ask_glaucoma_assistant(user_question, st.session_state.chat_history, API_KEY)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
            # rerun to show updated conversation
            try:
                st.experimental_rerun()
            except Exception:
                # older/newer streamlit may use st.rerun() — attempt that fallback
                try:
                    st.rerun()
                except Exception:
                    pass

    if clear_btn:
        st.session_state.chat_history = []
        try:
            st.experimental_rerun()
        except Exception:
            try:
                st.rerun()
            except Exception:
                pass

st.markdown('</div>', unsafe_allow_html=True)
