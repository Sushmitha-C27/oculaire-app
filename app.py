# app.py
# OCULAIRE — Updated: Progression Analysis, Scan Quality, Clinical/Research Mode,
# Chatbot context with scan metrics, Dark/Neon theme toggle
# Dependencies: streamlit, numpy, joblib, tensorflow, matplotlib, pillow, cv2, reportlab, sqlite3

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

# ReportLab for PDF
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# Try to import google.generativeai
try:
    import google.generativeai as genai
    USE_SDK = True
except Exception:
    USE_SDK = False

# -----------------------
# Page Config
# -----------------------
st.set_page_config(page_title="OCULAIRE: Updated", layout="wide", page_icon="👁️")

# -----------------------
# Session state defaults
# -----------------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'  # 'neon' or 'dark'
if 'mode' not in st.session_state:
    st.session_state.mode = 'clinical'  # 'clinical' or 'research'

# -----------------------
# Database path & init
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
            quality REAL,
            bscan_label TEXT,
            bscan_conf REAL,
            png BLOB
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# -----------------------
# Helpers: API key
# -----------------------
def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY", None)

API_KEY = get_api_key()

# -----------------------
# Theme CSS (dark / neon toggle)
# -----------------------
def inject_css(theme='dark'):
    if theme == 'neon':
        neon_css = """
        <style>
        :root {
          --bg: #001400;
          --panel: rgba(0,255,0,0.06);
          --accent: #39ff14;
          --muted: #aaffaa;
        }
        .stApp { background: radial-gradient(circle at 20% 20%, #002200, #000000 80%); color: #ccffcc; }
        .card { background: rgba(0,255,0,0.05); border:1px solid rgba(0,255,0,0.2); border-radius:12px; padding:12px; }
        h1 { font-size:48px; letter-spacing:8px; color:transparent; background: linear-gradient(90deg,#39ff14,#a8ff7d); -webkit-background-clip:text; animation: intenseGlow 1.4s infinite; }
        @keyframes intenseGlow { 0% { text-shadow: 0 0 18px #39ff14; } 50% { text-shadow: 0 0 48px #39ff14; } 100% { text-shadow: 0 0 18px #39ff14; } }
        .sev-inner { background: linear-gradient(90deg,#39ff14,#b8ff8a); box-shadow: 0 0 30px #39ff14; }
        footer { visibility:hidden; }
        </style>
        """
        st.markdown(neon_css, unsafe_allow_html=True)
    else:
        dark_css = """
        <style>
        :root { --bg:#020208; --panel:#0a0f25; --accent:#00f5ff; --muted:#a4b1c9; }
        .stApp { background: radial-gradient(circle at 20% 20%, #091133, #020208 90%); color: #e6faff; }
        .card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:12px; padding:12px; }
        h1 { font-size:42px; color:transparent; background:linear-gradient(90deg,#00f5ff,#ff40c4); -webkit-background-clip:text; }
        .sev-inner { background: linear-gradient(90deg,#00f5ff,#ff40c4); box-shadow: 0 0 30px rgba(0,245,255,0.6); }
        footer { visibility:hidden; }
        </style>
        """
        st.markdown(dark_css, unsafe_allow_html=True)

# inject initial theme
inject_css(st.session_state.theme)

# -----------------------
# Matplotlib theme
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
})

# -----------------------
# Model loading (cached)
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
# Image / NPZ helpers (kept concise)
# -----------------------
def process_npz(f):
    try:
        buf = io.BytesIO(f.getvalue())
        data = np.load(buf, allow_pickle=True)
        arr = data["volume"] if "volume" in data else data[data.files[0]]
        if arr.ndim == 3:
            arr = arr[0, :, :]
        vals = arr.flatten().astype(float)
        metrics = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals)), "min": float(np.nanmin(vals)), "max": float(np.nanmax(vals))}
        return arr, metrics
    except Exception as e:
        st.error(f"Error reading NPZ: {e}")
        return None, None

def preprocess_bscan(image_pil, size=(224,224)):
    arr = np.array(image_pil.convert('L'))
    arr = np.clip(arr, 0, np.percentile(arr, 99))
    arr = (arr - arr.min())/(arr.max()-arr.min()+1e-6)
    arr_res = cv2.resize(arr, size, interpolation=cv2.INTER_NEAREST)
    arr_rgb = np.repeat(arr_res[...,None],3,axis=-1)
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
            loss = preds[:,0]
        grads = tape.gradient(loss, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0,1,2))
        conv_out = conv_out[0]
        heat = conv_out @ pooled[..., tf.newaxis]
        heat = tf.squeeze(heat)
        heat = tf.maximum(heat, 0)/(tf.reduce_max(heat)+1e-6)
        return heat.numpy()
    except Exception:
        return None

# -----------------------
# New feature: Scan Quality Scoring
# -----------------------
def compute_image_quality(img_arr):
    """
    img_arr: 2D numpy grayscale array [0..255] or float normalized 0..1
    Returns quality_score (0-100) and components dict
    Components:
      - snr: signal-to-noise (simple heuristic)
      - contrast: RMS contrast
      - blur: variance of Laplacian (higher = sharper)
    """
    # normalize to 0..255
    a = img_arr.astype(np.float32)
    if a.max() <= 1.1:
        a = a * 255.0
    # SNR: mean / std
    mean = np.mean(a)
    std = np.std(a) + 1e-9
    snr = mean / std
    # map snr to 0..100 roughly
    snr_score = np.clip((snr / 10.0) * 100, 0, 100)

    # contrast: RMS contrast normalized
    rms = np.sqrt(np.mean((a - mean)**2))
    contrast_score = np.clip((rms / 80.0) * 100, 0, 100)

    # blur: variance of Laplacian
    lap = cv2.Laplacian(a.astype(np.uint8), cv2.CV_64F)
    var_lap = lap.var()
    blur_score = np.clip((var_lap / 200.0) * 100, 0, 100)

    # weighted combination
    quality = 0.45 * snr_score + 0.35 * contrast_score + 0.20 * blur_score
    components = {"snr": float(snr_score), "contrast": float(contrast_score), "sharpness": float(blur_score)}
    return float(np.clip(quality,0,100)), components

# -----------------------
# Severity renderer (same pattern)
# -----------------------
def render_severity(pct):
    pct = max(0.0, min(100.0, float(pct)))
    html = f"""
    <div class='sev-wrap' style='text-align:center;margin-top:10px;'>
      <div style='width:100%;max-width:900px;margin:auto;height:24px;background:rgba(255,255,255,0.03);border-radius:14px;border:1px solid rgba(255,255,255,0.04);overflow:hidden;'>
        <div id='sev_inner' style='height:100%; width:0%; background: linear-gradient(90deg, #00f5ff, #ff40c4); transform-origin:left center; transition: width 0.9s linear;'></div>
      </div>
      <div style='margin-top:8px; color: #a4b1c9; font-weight:700;'>{pct:.1f}% Severity</div>
    </div>
    <script>
      setTimeout(function(){{
        var el = document.getElementById('sev_inner'); if (el) el.style.width = '{pct:.1f}%';
      }}, 60);
    </script>
    """
    return html

# -----------------------
# Progression analysis (simple trend)
# -----------------------
def list_runs(limit=50):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, patient, patient_id, timestamp, metrics, severity, quality FROM runs ORDER BY id DESC LIMIT ?', (limit,))
    rows = c.fetchall()
    conn.close()
    return rows

def save_run(patient, patient_id, metrics, severity, quality, bscan_label=None, bscan_conf=None, png_bytes=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO runs (patient, patient_id, timestamp, metrics, severity, quality, bscan_label, bscan_conf, png) VALUES (?,?,?,?,?,?,?,?,?)',
              (patient, patient_id, datetime.utcnow().isoformat(), json.dumps(metrics), float(severity), float(quality), bscan_label, float(bscan_conf) if bscan_conf is not None else None, sqlite3.Binary(png_bytes) if png_bytes else None))
    conn.commit()
    conn.close()

def compute_progression(patient_id, limit=10):
    """
    Returns a dict with arrays (timestamps, mean_values, severity) if multiple runs exist for patient_id.
    Simple linear delta per day and delta percent are provided.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT timestamp, metrics, severity FROM runs WHERE patient_id=? ORDER BY id ASC LIMIT ?', (patient_id, limit))
    rows = c.fetchall()
    conn.close()
    if not rows or len(rows) < 2:
        return None
    timestamps = []
    means = []
    severities = []
    for ts, metrics_json, sev in rows:
        timestamps.append(datetime.fromisoformat(ts))
        try:
            mj = json.loads(metrics_json)
            means.append(float(mj.get('mean', np.nan)))
        except Exception:
            means.append(np.nan)
        severities.append(float(sev))
    return {"timestamps": timestamps, "means": np.array(means), "severities": np.array(severities)}

# -----------------------
# Chatbot function (includes full scan context when available)
# -----------------------
MODEL_NAME = "models/gemini-2.5-flash"  # keep as in your code

def ask_glaucoma_assistant(question, history, api_key, context_summary=None):
    """
    Calls Gemini or fallback to REST. Includes context_summary string (metrics, quality, summary)
    """
    if not api_key:
        return "⚠️ Please configure your Gemini API key."

    system_instruction = """You are a specialized medical AI assistant focused on glaucoma.
Only answer about glaucoma/OCT/RNFLT/B-scan when possible. Keep answers concise (<200 words), include a short educational disclaimer."""
    # include context
    prompt = system_instruction + "\n\n"
    if context_summary:
        prompt += "Patient scan context:\n" + context_summary + "\n\n"
    # include last chat messages for context
    for msg in history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        prompt += f"{role}: {msg['content']}\n"
    prompt += f"\nUser question: {question}\nAssistant:"
    try:
        if USE_SDK:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(MODEL_NAME)
            chat_history = []
            for msg in history[-6:]:
                role = "user" if msg["role"] == "user" else "model"
                chat_history.append({"role": role, "parts":[msg["content"]]})
            chat = model.start_chat(history=chat_history)
            response = chat.send_message(prompt)
            return response.text
        else:
            # fallback to REST endpoint (requires key)
            url = f"https://generativelanguage.googleapis.com/v1beta/{MODEL_NAME}:generateContent?key={api_key}"
            payload = {
                "contents":[{"parts":[prompt]}],
                "generationConfig":{"temperature":0.2, "maxOutputTokens":300}
            }
            r = requests.post(url, json=payload, timeout=25)
            if r.status_code == 200:
                data = r.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return f"🔺 API error {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return f"🔺 Error calling assistant: {e}"

# -----------------------
# PDF generator (Option C style, but simplified)
# -----------------------
def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=150)
    buf.seek(0)
    return buf.getvalue()

def generate_full_pdf(figs, rnflt_metrics=None, rnflt_cluster=None, rnflt_severity=None, bscan_label=None, bscan_conf=None, patient_name='-', patient_id='-', mode='clinical'):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=40, bottomMargin=40, leftMargin=50, rightMargin=50)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("Title", parent=styles["Title"], fontSize=26, textColor=colors.HexColor("#00eaff"), alignment=1)
    subtitle = ParagraphStyle("Subtitle", parent=styles["Heading2"], fontSize=12, textColor=colors.HexColor("#a6d4ff"), alignment=1)
    header_green = ParagraphStyle("HeaderGreen", parent=styles["Heading2"], textColor=colors.HexColor("#39ff14"), fontSize=14)
    body_blue = ParagraphStyle("BodyBlue", parent=styles["BodyText"], textColor=colors.HexColor("#0d6efd"), fontSize=10, leading=14)
    body = ParagraphStyle("Body", parent=styles["BodyText"], textColor=colors.HexColor("#e6faff"), fontSize=10, leading=14)
    story = []

    # COVER
    story.append(Paragraph("OCULAIRE", title))
    story.append(Paragraph("AI-Powered Glaucoma Screening Report", subtitle))
    story.append(Spacer(1, 12))
    meta = [
        ["Report Generated:", datetime.now().strftime("%B %d, %Y — %I:%M %p")],
        ["Analysis Type:", "RNFLT + B-Scan" if rnflt_metrics and bscan_label else ("RNFLT" if rnflt_metrics else "B-Scan")],
        ["AI Model Version:", "OCULAIRE v5.0 (2024)"],
        ["Report ID:", "OCU-" + datetime.now().strftime("%Y%m%d%H%M%S")],
        ["Patient:", f"{patient_name} (ID: {patient_id})"]
    ]
    meta_table = Table(meta, colWidths=[160,320])
    meta_table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.Color(0.05,0.05,0.08)),("TEXTCOLOR",(0,0),(-1,-1),colors.HexColor("#e6faff")),("BOX",(0,0),(-1,-1),1,colors.HexColor("#00eaff"))]))
    story.append(meta_table)
    story.append(Spacer(1,12))

    # Executive summary - color-coded
    if rnflt_cluster == "Glaucoma-like" or bscan_label == "Glaucoma-like":
        bg = colors.HexColor("#ff3b3b")
        text = "⚠️ ABNORMAL PATTERNS DETECTED"
        rl = ("HIGH" if (rnflt_severity or 0) >= 60 else "MODERATE" if (rnflt_severity or 0) >= 30 else "LOW-MODERATE")
    else:
        bg = colors.HexColor("#39ff14")
        text = "✅ NORMAL PATTERNS DETECTED"
        rl = "LOW"
    exec_par = Paragraph(f"<b>Status:</b> {text}<br/><b>Risk Level:</b> {rl}<br/><b>Severity Index:</b> {(rnflt_severity or 0):.1f}%<br/><b>CNN Confidence:</b> {(bscan_conf or 0):.1f}%", body if mode=='clinical' else body_blue)
    exec_box = Table([[exec_par]], colWidths=[400])
    exec_box.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),bg),("TEXTCOLOR",(0,0),(-1,-1),colors.white),("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10)]))
    story.append(exec_box)
    story.append(PageBreak())

    # Clinical interpretation + RNFLT stats
    story.append(Paragraph("CLINICAL INTERPRETATION", header_green))
    story.append(Spacer(1,8))
    if rnflt_cluster == "Glaucoma-like":
        story.append(Paragraph("The AI has detected patterns consistent with glaucomatous changes in RNFLT.", body))
    else:
        story.append(Paragraph("RNFLT pattern appears within normal ranges.", body))
    story.append(Spacer(1,10))
    if rnflt_metrics:
        tbl = [["Mean Thickness", f"{rnflt_metrics['mean']:.2f} μm"], ["Std Dev", f"{rnflt_metrics['std']:.2f} μm"], ["Min", f"{rnflt_metrics['min']:.2f} μm"], ["Max", f"{rnflt_metrics['max']:.2f} μm"]]
        table = Table(tbl, colWidths=[200,200])
        table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.Color(0.02,0.02,0.02)),("TEXTCOLOR",(0,0),(-1,-1),colors.HexColor("#ccffcc")),("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#39ff14"))]))
        story.append(table)
    story.append(PageBreak())

    # Images (each fig appended)
    story.append(Paragraph("DETAILED VISUAL ANALYSIS", header_green))
    story.append(Spacer(1,6))
    for fig in figs:
        png = fig_to_png(fig)
        img = RLImage(io.BytesIO(png), width=6.5*inch, height=3.2*inch)
        story.append(img)
        story.append(Spacer(1,10))
    story.append(PageBreak())

    # Recommendations
    story.append(Paragraph("RECOMMENDATIONS & ACTION PLAN", header_green))
    story.append(Spacer(1,8))
    if rnflt_cluster == "Glaucoma-like":
        recs = ["Schedule ophthalmologist visit within 1–2 weeks.", "Request IOP, OCT, visual fields.", "Avoid heavy lifting and inverted poses."]
    else:
        recs = ["Continue annual eye exams.", "Document this baseline for future comparison."]
    for r in recs:
        story.append(Paragraph(f"- {r}", body))
    story.append(PageBreak())

    # Disclaimers & refs
    story.append(Paragraph("IMPORTANT MEDICAL DISCLAIMER", header_green))
    story.append(Spacer(1,6))
    story.append(Paragraph("This OCULAIRE report is for research/educational purposes only. It is not a medical diagnosis. Consult an ophthalmologist.", body))
    story.append(Spacer(1,8))
    docs = ["Weinreb RN et al., JAMA 2014", "Tham YC et al., Ophthalmology 2014", "European Glaucoma Society Guidelines 2021"]
    for d in docs:
        story.append(Paragraph(f"- {d}", ParagraphStyle("refs", parent=styles["BodyText"], fontSize=9, textColor=colors.HexColor("#d0ffd0"))))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# -----------------------
# UI: Sidebar controls (mode, theme)
# -----------------------
with st.sidebar:
    st.header("Settings")
    theme = st.radio("Theme", ["dark", "neon"], index=0 if st.session_state.theme=='dark' else 1)
    if theme != st.session_state.theme:
        st.session_state.theme = theme
        inject_css(st.session_state.theme)
    mode = st.radio("Mode", ["clinical", "research"], index=0 if st.session_state.mode=='clinical' else 1)
    st.session_state.mode = mode
    st.markdown("---")
    st.markdown("🔑 API Key (optional for chatbot)")
    if API_KEY:
        st.success("Gemini API key found")
    else:
        st.info("Add GEMINI_API_KEY to secrets or env to enable chatbot")

# -----------------------
# Header (fast glow if neon)
# -----------------------
st.markdown("<div style='text-align:center;margin-top:8px;'><h1>OCULAIRE</h1><p style='color:var(--muted)'>AI-Powered Glaucoma Detection Dashboard</p></div>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# -----------------------
# Sidebar: Uploads + converters
# -----------------------
with st.sidebar:
    st.subheader("Uploads & Tools")
    patient_name = st.text_input("Patient name", key="patient_name")
    patient_id = st.text_input("Patient ID", key="patient_id")
    st.markdown("---")
    st.subheader("RNFLT Input")
    rnflt_input_mode = st.radio("RNFLT input", ["NPZ (recommended)", "Single Image"])
    rnflt_conv_files = st.file_uploader("Convert RNFLT slices → .npz", type=["png","jpg","jpeg"], accept_multiple_files=True, key="rnflt_conv")
    if rnflt_conv_files:
        if st.button("Convert RNFLT slices"):
            try:
                stacks=[]
                for f in rnflt_conv_files:
                    im = Image.open(f).convert('L')
                    stacks.append(np.array(im).astype(np.float32))
                vol = np.stack(stacks,axis=0)
                buf = io.BytesIO()
                np.savez_compressed(buf, volume=vol)
                buf.seek(0)
                st.success(f"Packed {vol.shape[0]} slices")
                st.download_button("Download RNFLT .npz", data=buf.getvalue(), file_name="rnflt_volume.npz")
            except Exception as e:
                st.error(f"Conv error: {e}")

    st.markdown("---")
    st.subheader("B-Scan Input")
    bscan_input_mode = st.radio("B-scan input", ["Image (recommended)", "NPZ (multi-slice)"])
    bscan_conv_files = st.file_uploader("Convert B-scan slices → .npz", type=["png","jpg","jpeg"], accept_multiple_files=True, key="bscan_conv")
    if bscan_conv_files:
        if st.button("Convert B-scan slices"):
            try:
                stacks=[]
                for f in bscan_conv_files:
                    im = Image.open(f).convert('L')
                    stacks.append(np.array(im).astype(np.float32))
                vol = np.stack(stacks,axis=0)
                buf = io.BytesIO()
                np.savez_compressed(buf, volume=vol)
                buf.seek(0)
                st.success(f"Packed {vol.shape[0]} slices")
                st.download_button("Download B-scan .npz", data=buf.getvalue(), file_name="bscan_volume.npz")
            except Exception as e:
                st.error(f"Conv error: {e}")

# -----------------------
# Main layout: Upload panels
# -----------------------
colA, colB = st.columns(2)
with colA:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🩺 RNFLT Map Analysis")
    rnflt_arr=None; rnflt_metrics=None; rnflt_pil=None
    if rnflt_input_mode=="NPZ (recommended)":
        rnflt_file = st.file_uploader("Upload RNFLT .npz", type=["npz"], key="rnflt_main")
        if rnflt_file:
            rnflt_arr, rnflt_metrics = process_npz(rnflt_file)
    else:
        rnflt_img = st.file_uploader("Upload RNFLT image (png/jpg)", type=["png","jpg","jpeg"], key="rnflt_img_main")
        if rnflt_img:
            pil = Image.open(rnflt_img).convert('L')
            rnflt_pil = pil
            arr = np.array(pil).astype(float)
            rnflt_arr = arr
            vals=arr.flatten()
            rnflt_metrics={"mean":float(np.nanmean(vals)),"std":float(np.nanstd(vals)),"min":float(np.nanmin(vals)),"max":float(np.nanmax(vals))}
    st.markdown("</div>", unsafe_allow_html=True)

with colB:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("👁️ B-Scan Analysis")
    bscan_file=None; bscan_npz=None
    if bscan_input_mode=="Image (recommended)":
        bscan_file = st.file_uploader("Upload B-scan image", type=["png","jpg","jpeg"], key="bscan_main")
    else:
        bscan_npz = st.file_uploader("Upload B-scan NPZ", type=["npz"], key="bscan_npz_main")
    st.markdown("</div>", unsafe_allow_html=True)

# threshold slider (keep in main content)
threshold = st.slider("Thin-zone threshold (µm)", 5, 50, 10)

# -----------------------
# Analysis trigger
# -----------------------
analysis_trigger = (rnflt_arr is not None) or (bscan_file is not None) or (bscan_npz is not None)

# Storage for figs and summary
figs = []
context_summary = ""

if analysis_trigger:
    st.markdown("<hr>", unsafe_allow_html=True)
    severity_overall = 0.0
    # RNFLT processing
    if rnflt_arr is not None:
        try:
            metrics = rnflt_metrics or {}
            # cluster prediction if available
            if scaler is not None and kmeans is not None:
                X = np.array([[metrics.get("mean",0), metrics.get("std",0), metrics.get("min",0), metrics.get("max",0)]])
                try:
                    Xs = scaler.transform(X)
                    cluster = int(kmeans.predict(Xs)[0])
                    label_r = "Glaucoma-like" if cluster == thin_cluster else "Healthy-like"
                except Exception:
                    label_r = "Unknown"
            else:
                label_r = "Unknown"
            # diff & risk
            if avg_healthy is not None:
                diff, risk, sev = compute_risk_map(rnflt_arr, avg_healthy, -threshold)
            else:
                diff = rnflt_arr - np.nanmean(rnflt_arr)
                risk = np.where(diff < -threshold, diff, np.nan)
                # define small sev as percent thin area:
                sev = (np.isfinite(risk).sum() / rnflt_arr.size) * 100
            severity_overall = max(severity_overall, float(sev))
            # compute quality score if we have an image
            img_for_quality = rnflt_arr
            q_score, q_comp = compute_image_quality(img_for_quality)
            # display metrics
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Status", label_r)
            c2.metric("Mean RNFLT", f"{metrics.get('mean',0):.2f}")
            c3.metric("Std Dev", f"{metrics.get('std',0):.2f}")
            c4.metric("Cluster", str(label_r))
            st.markdown(render_severity(sev), unsafe_allow_html=True)
            # RNFLT figure (3-panel)
            fig, axes = plt.subplots(1,3,figsize=(16,5))
            im0 = axes[0].imshow(rnflt_arr, cmap='turbo'); axes[0].axis('off'); axes[0].set_title("RNFLT Map")
            plt.colorbar(im0, ax=axes[0], shrink=0.7)
            im1 = axes[1].imshow(diff, cmap='bwr', vmin=-30, vmax=30); axes[1].axis('off'); axes[1].set_title("Difference vs Healthy")
            plt.colorbar(im1, ax=axes[1], shrink=0.7)
            im2 = axes[2].imshow(risk, cmap='hot'); axes[2].axis('off'); axes[2].set_title("Risk Map")
            plt.colorbar(im2, ax=axes[2], shrink=0.7)
            fig.patch.set_facecolor("#050612")
            st.pyplot(fig)
            figs.append(fig)
            # show quality
            st.markdown("### Scan Quality")
            st.write(f"**Quality score:** {q_score:.1f}% — components: SNR {q_comp['snr']:.1f}, Contrast {q_comp['contrast']:.1f}, Sharpness {q_comp['sharpness']:.1f}")
            # small explanation
            if q_score < 40:
                st.warning("Low scan quality detected — consider re-acquiring with better focus/contrast.")
            elif q_score < 70:
                st.info("Acceptable quality but consider improvements for clinical use.")
            else:
                st.success("High scan quality.")
            # add to context summary
            context_summary += f"RNFLT summary: Status={label_r}, Mean={metrics.get('mean',np.nan):.1f}, Severity={sev:.1f}%, Quality={q_score:.1f}%.\n"
        except Exception as e:
            st.error(f"RNFLT Error: {e}")

    # B-scan processing
    if bscan_file or bscan_npz:
        try:
            if bscan_file:
                pil = Image.open(bscan_file).convert('L')
            else:
                # if npz: extract first slice and convert to PIL
                vol, _ = process_npz(bscan_npz)
                img_slice = vol if vol.ndim==2 else vol[0,:,:]
                pil = Image.fromarray(np.uint8(255*(img_slice - np.nanmin(img_slice))/(np.nanmax(img_slice)-np.nanmin(img_slice)+1e-9)))
            batch, proc = preprocess_bscan(pil)
            # compute quality
            proc_255 = (proc*255).astype(np.uint8) if proc.max() <= 1.1 else proc.astype(np.uint8)
            q_score_b, q_comp_b = compute_image_quality(proc_255)
            # model prediction
            if b_model:
                pred_raw = float(b_model.predict(batch, verbose=0)[0][0])
                label_b = "Glaucoma-like" if pred_raw > 0.5 else "Healthy-like"
                conf = pred_raw*100 if label_b=="Glaucoma-like" else (1-pred_raw)*100
            else:
                label_b = "Unknown"; conf = 0.0
            severity_overall = max(severity_overall, conf)
            col1,col2 = st.columns(2)
            col1.metric("CNN Prediction", label_b)
            col2.metric("Confidence", f"{conf:.1f}%")
            st.markdown(render_severity(conf), unsafe_allow_html=True)
            heat = gradcam(batch, b_model) if b_model else None
            if heat is not None:
                heat_r = cv2.resize(heat, (proc.shape[1], proc.shape[0]))
                hm = (heat_r*255).astype(np.uint8)
                hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
                overlay = (np.stack([proc]*3,axis=-1)*255).astype(np.uint8)
                overlay = cv2.addWeighted(overlay, 0.6, hm_color, 0.4, 0)
                st.image([pil, overlay], caption=["Original B-Scan", "Grad-CAM Overlay"], use_column_width=True)
                # store fig
                fig2, ax2 = plt.subplots(1,2,figsize=(12,5))
                ax2[0].imshow(pil, cmap='gray'); ax2[0].axis('off'); ax2[0].set_title("B-Scan")
                ax2[1].imshow(overlay); ax2[1].axis('off'); ax2[1].set_title("Grad-CAM")
                fig2.patch.set_facecolor("#050612")
                figs.append(fig2)
            else:
                st.image(pil, caption="B-scan", use_column_width=True)
            # show quality for bscan
            st.markdown("### B-scan Quality")
            st.write(f"**Quality score:** {q_score_b:.1f}% — SNR {q_comp_b['snr']:.1f}, Contrast {q_comp_b['contrast']:.1f}, Sharpness {q_comp_b['sharpness']:.1f}")
            if q_score_b < 40: st.warning("Low B-scan quality detected.")
            elif q_score_b < 70: st.info("Acceptable B-scan quality.")
            else: st.success("High B-scan quality.")
            context_summary += f"B-scan summary: Status={label_b}, Confidence={conf:.1f}%, Quality={q_score_b:.1f}%.\n"
        except Exception as e:
            st.error(f"B-scan Error: {e}")

    # combined summary
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Overall Severity Index")
    st.markdown(render_severity(severity_overall), unsafe_allow_html=True)
    # store png of first fig for DB
    png_bytes = None
    if figs:
        try:
            png_bytes = fig_to_png(figs[0])
        except Exception:
            png_bytes = None
    # save run to DB button & auto save option in research mode
    if st.button("💾 Save Run to History"):
        try:
            save_run(patient_name or "-", patient_id or "-", rnflt_metrics or {}, severity_overall, (q_score if 'q_score' in locals() else (q_score_b if 'q_score_b' in locals() else 0.0)), bscan_label if 'label_b' in locals() else None, (conf if 'conf' in locals() else None), png_bytes)
            st.success("Saved run to local history.")
        except Exception as e:
            st.error(f"Save error: {e}")
    # automatically save in research mode (optional)
    if st.session_state.mode == 'research':
        # lightweight autosave (if patient_id provided) - avoid duplicates by timestamp uniqueness
        try:
            if patient_id:
                save_run(patient_name or "-", patient_id or "-", rnflt_metrics or {}, severity_overall, (q_score if 'q_score' in locals() else (q_score_b if 'q_score_b' in locals() else 0.0)), bscan_label if 'label_b' in locals() else None, (conf if 'conf' in locals() else None), png_bytes)
        except Exception:
            pass

    # prepare PDF generation session state
    st.session_state['pdf_figs'] = figs
    st.session_state['pdf_rnflt_metrics'] = rnflt_metrics
    st.session_state['pdf_rnflt_cluster'] = label_r if 'label_r' in locals() else None
    st.session_state['pdf_rnflt_severity'] = severity_overall
    st.session_state['pdf_bscan_label'] = label_b if 'label_b' in locals() else None
    st.session_state['pdf_bscan_conf'] = conf if 'conf' in locals() else 0.0
    st.session_state['pdf_context_summary'] = context_summary

    # progression analysis if patient_id present
    if patient_id:
        prog = compute_progression(patient_id, limit=20)
        if prog:
            st.markdown("### Progression Analysis")
            # simple line plots for mean RNFLT and severity
            figp, axp = plt.subplots(1,2,figsize=(12,3))
            dates = prog['timestamps']
            axp[0].plot(dates, prog['means'], marker='o'); axp[0].set_title("Mean RNFLT over time"); axp[0].tick_params(axis='x', rotation=25)
            axp[1].plot(dates, prog['severities'], marker='o'); axp[1].set_title("Severity over time"); axp[1].tick_params(axis='x', rotation=25)
            figp.patch.set_facecolor("#050612")
            st.pyplot(figp)
            # compute trends
            dt_days = (dates[-1] - dates[0]).days or 1
            delta_mean = float(prog['means'][-1] - prog['means'][0])
            delta_sev = float(prog['severities'][-1] - prog['severities'][0])
            st.write(f"Change in mean RNFLT over {dt_days} days: {delta_mean:.2f} μm")
            st.write(f"Change in severity over {dt_days} days: {delta_sev:.2f}%")
        else:
            st.info("Not enough history for progression analysis (need ≥2 runs).")

    # Downloads and PDF generation
    if figs:
        png_bytes = fig_to_png(figs[0])
        st.download_button("📸 Download RNFLT PNG", data=png_bytes, file_name="oculaire_rnflt.png")
        if st.button("📄 Generate Full Medical Report (PDF)"):
            try:
                generated = generate_full_pdf(figs,
                                              rnflt_metrics=rnflt_metrics,
                                              rnflt_cluster=label_r if 'label_r' in locals() else None,
                                              rnflt_severity=severity_overall,
                                              bscan_label=label_b if 'label_b' in locals() else None,
                                              bscan_conf=conf if 'conf' in locals() else 0.0,
                                              patient_name=patient_name or "-",
                                              patient_id=patient_id or "-",
                                              mode=st.session_state.mode)
                st.success("PDF generated.")
                st.download_button("⬇️ Download Full PDF Report", data=generated, file_name="OCULAIRE_Report.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"PDF generation error: {e}")

# -----------------------
# Chatbot area (floating expander)
# -----------------------
st.markdown('<div class="floating-expander">', unsafe_allow_html=True)
with st.expander("💬 Ask AI assistant", expanded=False):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🤖 Glaucoma Q&A Assistant")
    st.markdown("<small style='color:var(--muted)'>The assistant will include the latest scan summary when available.</small>", unsafe_allow_html=True)
    # show basic context
    if 'pdf_context_summary' in st.session_state and st.session_state['pdf_context_summary']:
        st.info("Context included with questions: " + st.session_state['pdf_context_summary'])
    # chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-msg'><strong>You:</strong> {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-msg'><strong>🤖:</strong> {msg['content']}</div>", unsafe_allow_html=True)
    question = st.text_input("Your question:", key="chat_input", placeholder="e.g., Interpret the RNFLT severity and what to do next")
    col1, col2 = st.columns([4,1])
    with col1:
        send = st.button("📤 Send")
    with col2:
        clear = st.button("🗑️ Clear")
    if send and question:
        # assemble context summary (from session_state)
        context = st.session_state.get('pdf_context_summary', "")
        st.session_state.chat_history.append({"role":"user","content":question})
        reply = ask_glaucoma_assistant(question, st.session_state.chat_history, API_KEY, context_summary=context)
        st.session_state.chat_history.append({"role":"assistant","content":reply})
        st.experimental_rerun()
    if clear:
        st.session_state.chat_history = []
        st.experimental_rerun()
    st.markdown("</div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# Saved runs viewer (bottom)
# -----------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("Saved Runs (recent)")
runs = list_runs(limit=20)
if runs:
    for r in runs:
        rid, rpatient, rpid, rts, rmetrics, rsev, rquality = r
        cols = st.columns([4,1,1])
        with cols[0]:
            st.markdown(f"**{rpatient}** (ID: {rpid}) — {rts}")
            st.markdown(f"Severity: {rsev:.2f}% — Quality: {rquality:.1f}%")
            # show metrics collapsed
            with st.expander("View metrics & download PNG"):
                st.write(json.loads(rmetrics) if rmetrics else {})
                # download stored png if available
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("SELECT png FROM runs WHERE id=?", (rid,))
                row = c.fetchone()
                conn.close()
                if row and row[0]:
                    st.download_button(f"Download PNG #{rid}", data=row[0], file_name=f"run_{rid}.png", mime="image/png")
        with cols[1]:
            if st.button(f"⬇️ Download PDF for #{rid}", key=f"dl_{rid}"):
                # regenerate PDF from stored png and metrics (quick cover)
                try:
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute("SELECT metrics, severity, bscan_label, bscan_conf, png FROM runs WHERE id=?", (rid,))
                    rr = c.fetchone()
                    conn.close()
                    metrics_j = json.loads(rr[0]) if rr and rr[0] else {}
                    sever = rr[1] if rr else 0.0
                    bl = rr[2] if rr else None
                    bc = rr[3] if rr else 0.0
                    pngb = rr[4] if rr and rr[4] else None
                    figs_local = []
                    if pngb:
                        # create matplotlib fig to embed
                        im = Image.open(io.BytesIO(pngb))
                        figL, axL = plt.subplots(figsize=(6.5,3))
                        axL.imshow(im); axL.axis('off')
                        figs_local.append(figL)
                    pdfgen = generate_full_pdf(figs_local, rnflt_metrics=metrics_j, rnflt_cluster=None, rnflt_severity=sever, bscan_label=bl, bscan_conf=bc, patient_name=rpatient, patient_id=rpid, mode=st.session_state.mode)
                    st.download_button(f"Download Report #{rid}", data=pdfgen, file_name=f"OCULAIRE_run_{rid}.pdf", mime="application/pdf")
                except Exception as e:
                    st.error(f"Error generating PDF: {e}")
        with cols[2]:
            if st.button(f"🗑️ Delete #{rid}", key=f"del_{rid}"):
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("DELETE FROM runs WHERE id=?", (rid,))
                conn.commit()
                conn.close()
                st.experimental_rerun()
else:
    st.info("No saved runs yet. Run an analysis and save to build history.")

# End of file
