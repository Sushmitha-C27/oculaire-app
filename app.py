# app.py — OCULAIRE Option C (Neon Green UI) — Full app
# ============================================================
import os
import io
import time
import json
import sqlite3
import numpy as np
import joblib
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import requests
import streamlit as st
from datetime import datetime

# Optional: TensorFlow model for B-scan (if available)
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# ReportLab for PDF generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image as RLImage
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# Try to import google.generativeai (Gemini SDK)
try:
    import google.generativeai as genai
    USE_SDK = True
except Exception:
    USE_SDK = False

# ============================================================
# Page config + theme
# ============================================================
st.set_page_config(page_title="OCULAIRE: Neon Glaucoma Detection Dashboard", layout="wide", page_icon="👁️")

# Neon green CSS + fast glow header (Option C)
st.markdown("""
<style>
:root{
  --green:#39ff14;
  --green-soft:#aaffaa;
  --bg:#000f00;
  --muted:#ccffcc;
}

/* App */
.stApp {
  background: radial-gradient(circle at 20% 20%, #002200, #000000 80%);
  color: var(--muted);
  font-family: 'Plus Jakarta Sans', sans-serif;
}

/* header */
h1.oculaire {
  font-size:72px;
  font-weight:900;
  letter-spacing:12px;
  background: linear-gradient(90deg, #39ff14, #b8ff8a);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: intenseGlow 1.4s infinite ease-in-out;
  text-align:center;
  margin-bottom: -6px;
}
h3.oculaire-sub { color: #aaffaa; text-align:center; margin-top:4px; }

/* card */
.card { background: rgba(0,255,0,0.05); border-radius:12px; padding:14px; border:1px solid rgba(0,255,0,0.08); }

/* severity */
.sev-inner { background: linear-gradient(90deg,#39ff14,#b8ff8a); box-shadow:0 0 35px #39ff14; transition: width 0.9s ease-in-out; height:100%; border-radius:18px; }

/* chat bubbles */
.user-msg { background: rgba(0,255,0,0.12); border-left:4px solid #39ff14; padding:10px; border-radius:8px; }
.assistant-msg { background: rgba(100,255,150,0.12); border-left:4px solid #7dff7d; padding:10px; border-radius:8px; }

/* floating expander */
.floating-expander details { background: rgba(0,0,0,0.55) !important; border-radius:12px !important; }
.footer-note { text-align:center; color:#99ff99; padding:6px; }
@keyframes intenseGlow {
  0% { transform: scale(1); text-shadow: 0 0 18px #39ff14, 0 0 30px #39ff14; }
  50% { transform: scale(1.06); text-shadow: 0 0 50px #39ff14, 0 0 80px #39ff14; }
  100% { transform: scale(1); text-shadow: 0 0 18px #39ff14, 0 0 30px #39ff14; }
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="oculaire">OCULAIRE</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="oculaire-sub">Illuminating Vision. Detecting Glaucoma. — Neon Lab v5</h3>', unsafe_allow_html=True)
st.markdown("---")

# ============================================================
# Session state defaults
# ============================================================
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdf_figs' not in st.session_state:
    st.session_state.pdf_figs = []
if 'pdf_data' not in st.session_state:
    st.session_state.pdf_data = {}
if 'trigger_pdf' not in st.session_state:
    st.session_state.trigger_pdf = False

# ============================================================
# API helper
# ============================================================
def get_api_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", None)

API_KEY = get_api_key()

# ============================================================
# Matplotlib theme for figures
# ============================================================
plt.style.use('dark_background')
plt.rcParams.update({
    "figure.facecolor": "#020802",
    "axes.facecolor": "#020802",
    "axes.edgecolor": "#39ff14",
    "axes.labelcolor": "#aaffaa",
    "xtick.color": "#39ff14",
    "ytick.color": "#39ff14",
    "text.color": "#ccffcc",
    "font.size": 11,
})

# ============================================================
# Robust SQLite helpers (safe, create schema, WAL mode)
# ============================================================
DB_PATH = os.path.join('.', 'oculaire_runs.db')

def _get_conn(retries=3, delay=0.2, timeout=20.0):
    last_exc = None
    for attempt in range(retries):
        try:
            conn = sqlite3.connect(DB_PATH, timeout=timeout, check_same_thread=False)
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")
            cur.execute("PRAGMA foreign_keys = ON;")
            conn.commit()
            cur.close()
            return conn
        except Exception as e:
            last_exc = e
            time.sleep(delay * (attempt + 1))
    raise last_exc

from contextlib import contextmanager
@contextmanager
def db_cursor():
    conn = None
    cur = None
    try:
        conn = _get_conn()
        cur = conn.cursor()
        yield conn, cur
        conn.commit()
    except Exception:
        if conn:
            conn.rollback()
        raise
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def init_db():
    try:
        with db_cursor() as (conn, c):
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
    except Exception as e:
        st.error(f"Database initialization error: {e}")

def save_run(patient, patient_id, metrics, severity, quality=None, bscan_label=None, bscan_conf=None, png_bytes=None):
    try:
        with db_cursor() as (conn, c):
            c.execute('''
                INSERT INTO runs (patient, patient_id, timestamp, metrics, severity, quality, bscan_label, bscan_conf, png)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                patient or '-', patient_id or '-', datetime.utcnow().isoformat(),
                json.dumps(metrics or {}), float(severity or 0.0), float(quality or 0.0),
                bscan_label, float(bscan_conf) if bscan_conf is not None else None,
                sqlite3.Binary(png_bytes) if png_bytes else None
            ))
            return c.lastrowid
    except Exception as e:
        st.error(f"Error saving run: {e}")
        return None

def list_runs(limit=20):
    try:
        with db_cursor() as (conn, c):
            c.execute('SELECT id, patient, patient_id, timestamp, metrics, severity, quality FROM runs ORDER BY id DESC LIMIT ?', (limit,))
            return c.fetchall()
    except Exception as e:
        st.warning(f"Could not read saved runs: {e}")
        return []

def load_run_png(run_id):
    try:
        with db_cursor() as (conn, c):
            c.execute('SELECT png FROM runs WHERE id=?', (run_id,))
            row = c.fetchone()
            return row[0] if row and row[0] else None
    except Exception as e:
        st.warning(f"Could not load run {run_id}: {e}")
        return None

init_db()

# ============================================================
# Model loading & safe fallbacks
# ============================================================
@st.cache_resource
def load_models():
    b_model = None
    scaler = kmeans = avg_healthy = avg_glaucoma = thin_cluster = None

    # Load b-scan model if TF available and file present
    try:
        if TF_AVAILABLE and os.path.exists("bscan_cnn.h5"):
            b_model = tf.keras.models.load_model("bscan_cnn.h5", compile=False)
    except Exception:
        b_model = None

    # Load RNFLT artifacts if available
    try:
        if os.path.exists("rnflt_scaler.joblib"):
            scaler = joblib.load("rnflt_scaler.joblib")
        if os.path.exists("rnflt_kmeans.joblib"):
            kmeans = joblib.load("rnflt_kmeans.joblib")
        if os.path.exists("avg_map_healthy.npy"):
            avg_healthy = np.load("avg_map_healthy.npy")
        if os.path.exists("avg_map_glaucoma.npy"):
            avg_glaucoma = np.load("avg_map_glaucoma.npy")
        if avg_healthy is not None and avg_glaucoma is not None:
            thin_cluster = 0 if np.nanmean(avg_healthy) > np.nanmean(avg_glaucoma) else 1
    except Exception:
        scaler = kmeans = avg_healthy = avg_glaucoma = thin_cluster = None

    return b_model, scaler, kmeans, avg_healthy, avg_glaucoma, thin_cluster

b_model, scaler, kmeans, avg_healthy, avg_glaucoma, thin_cluster = load_models()

# ============================================================
# Helpers (npz processing, risk calc, gradcam, PDF helpers)
# ============================================================
def process_npz(uploaded_file):
    try:
        buf = io.BytesIO(uploaded_file.getvalue())
        data = np.load(buf, allow_pickle=True)
        arr = data["volume"] if "volume" in data else data[data.files[0]]
        if arr.ndim == 3:
            arr = arr[0, :, :]
        arr = arr.astype(float)
        vals = arr.flatten()
        metrics = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals)), "min": float(np.nanmin(vals)), "max": float(np.nanmax(vals))}
        return arr, metrics
    except Exception as e:
        st.error(f"Error reading NPZ: {e}")
        return None, None

def compute_risk_map(rnflt, healthy, threshold=-10):
    try:
        if rnflt.shape != healthy.shape:
            healthy = cv2.resize(healthy, (rnflt.shape[1], rnflt.shape[0]))
        diff = rnflt - healthy
        risk = np.where(diff < threshold, diff, np.nan)
        total = np.isfinite(diff).sum()
        risky = np.isfinite(risk).sum()
        severity = (risky / total) * 100 if total else 0.0
        return diff, risk, severity
    except Exception:
        return rnflt - np.nanmean(rnflt), np.where((rnflt - np.nanmean(rnflt)) < -threshold, rnflt - np.nanmean(rnflt), np.nan), 0.0

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
        if model is None: return None
        last_conv = None
        for layer in reversed(model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                last_conv = layer.name
                break
        if not last_conv: return None
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

def fig_to_png(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=dpi)
    buf.seek(0)
    return buf.getvalue()

# ============================================================
# PDF generator (reportlab) — description/body text styled blue
# ============================================================
def generate_full_pdf(figs, rnflt_metrics=None, rnflt_cluster=None, rnflt_severity=None, bscan_label=None, bscan_conf=None):
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("reportlab is required for PDF generation. Install reportlab and restart.")
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=36, bottomMargin=36, leftMargin=50, rightMargin=50)
    styles = getSampleStyleSheet()

    # Title/Body styles — body colored blue as requested
    title_style = ParagraphStyle("TitleGlow", parent=styles["Title"], fontSize=26, textColor=colors.HexColor("#00eaff"), alignment=1)
    subtitle_style = ParagraphStyle("Subtitle", parent=styles["Heading2"], fontSize=12, textColor=colors.HexColor("#ff40c4"), alignment=1)
    header_green = ParagraphStyle("HeaderGreen", parent=styles["Heading2"], fontSize=14, textColor=colors.HexColor("#39ff14"))
    # *** BODY in BLUE as requested ***
    body_blue = ParagraphStyle("BodyBlue", parent=styles["BodyText"], fontSize=11, leading=14, textColor=colors.HexColor("#0b6aff"))
    body_small_blue = ParagraphStyle("SmallBlue", parent=styles["BodyText"], fontSize=9, leading=12, textColor=colors.HexColor("#0b6aff"))

    story = []

    # Page 1 — cover + metadata + executive summary
    story.append(Paragraph("OCULAIRE", title_style))
    story.append(Paragraph("AI-Powered Glaucoma Screening Report", subtitle_style))
    story.append(Spacer(1, 16))

    report_id = "OCU-" + datetime.now().strftime("%Y%m%d%H%M%S")
    gen_date = datetime.now().strftime("%B %d, %Y — %I:%M %p")

    metadata = [
        ["Report Generated:", gen_date],
        ["Analysis Type:", "RNFLT + B-Scan" if (figs and len(figs)>0) else "RNFLT or B-Scan"],
        ["AI Model Version:", "OCULAIRE v5.0 (2024)"],
        ["Report ID:", report_id],
    ]
    meta_table = Table(metadata, colWidths=[150, 300])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1), colors.Color(0.05,0.05,0.08)),
        ("TEXTCOLOR",(0,0),(-1,-1), colors.HexColor("#e6faff")),
        ("INNERGRID",(0,0),(-1,-1),0.3, colors.HexColor("#00eaff")),
        ("BOX",(0,0),(-1,-1),1, colors.HexColor("#00eaff")),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # Exec summary (color-coded red/green)
    if (rnflt_cluster == "Glaucoma-like") or (bscan_label == "Glaucoma-like"):
        risk_color = colors.HexColor("#ff6b6b")
        risk_text = "⚠️ ABNORMAL PATTERNS DETECTED"
        risk_level = ("HIGH" if (rnflt_severity or 0) >= 60 else "MODERATE" if (rnflt_severity or 0) >= 30 else "LOW-MODERATE")
    else:
        risk_color = colors.HexColor("#39ff14")
        risk_text = "✅ NORMAL PATTERNS DETECTED"
        risk_level = "LOW"

    exec_para = Paragraph(f"<b>Status:</b> {risk_text}<br/><b>Risk Level:</b> {risk_level}<br/><b>Severity Index:</b> {(rnflt_severity or 0):.1f}%<br/><b>CNN Confidence:</b> {(bscan_conf or 0):.1f}%", body_blue)
    box_table = Table([[exec_para]], colWidths=[450])
    box_table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1), risk_color),
        ("TEXTCOLOR",(0,0),(-1,-1), colors.white),
        ("BOX",(0,0),(-1,-1),1.2, colors.black),
        ("LEFTPADDING",(0,0),(-1,-1),12),
        ("RIGHTPADDING",(0,0),(-1,-1),12),
        ("TOPPADDING",(0,0),(-1,-1),10),
        ("BOTTOMPADDING",(0,0),(-1,-1),10),
    ]))
    story.append(box_table)
    story.append(PageBreak())

    # Page 2 — Clinical interpretation & RNFLT stats
    story.append(Paragraph("CLINICAL INTERPRETATION", header_green))
    story.append(Spacer(1,8))
    if rnflt_cluster == "Glaucoma-like":
        story.append(Paragraph("The AI analysis has detected <b>patterns consistent with glaucomatous changes</b> in your RNFL structure. These include thinning in clinically significant regions compared to the healthy reference.", body_blue))
    else:
        story.append(Paragraph("The RNFL pattern appears <b>within normal ranges</b>. No significant signs of glaucomatous thinning are identified at this time.", body_blue))
    story.append(Spacer(1,10))
    story.append(Paragraph("<b>Key Findings</b>", body_blue))
    story.append(Spacer(1,6))
    if rnflt_cluster == "Glaucoma-like":
        findings = [
            "- RNFL thinning detected in critical sectors.",
            f"- { (rnflt_severity or 0):.1f }% of retinal area flagged as at-risk.",
            "- Pattern deviation from healthy baseline exceeds threshold.",
            "- Suggestive of early-to-moderate glaucomatous damage."
        ]
    else:
        findings = [
            "- RNFL thickness within expected clinical ranges.",
            "- No significant thinning detected.",
            "- Pattern matches healthy reference distribution."
        ]
    for f in findings:
        story.append(Paragraph(f, body_blue))
    story.append(Spacer(1,12))
    story.append(Paragraph("<b>RNFLT Measurements</b>", header_green))
    story.append(Spacer(1,6))
    if rnflt_metrics:
        rnflt_tbl = [
            ["Mean Thickness", f"{rnflt_metrics.get('mean',0):.2f} μm"],
            ["Standard Deviation", f"{rnflt_metrics.get('std',0):.2f} μm"],
            ["Minimum", f"{rnflt_metrics.get('min',0):.2f} μm"],
            ["Maximum", f"{rnflt_metrics.get('max',0):.2f} μm"],
        ]
        rnflt_table = Table(rnflt_tbl, colWidths=[220,220])
        rnflt_table.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1), colors.Color(0.02,0.05,0.18)),
            ("TEXTCOLOR",(0,0),(-1,-1), colors.HexColor("#0b6aff")),
            ("GRID",(0,0),(-1,-1),0.5, colors.HexColor("#00eaff")),
        ]))
        story.append(rnflt_table)
    story.append(PageBreak())

    # Page 3 — Images
    story.append(Paragraph("DETAILED VISUAL ANALYSIS", header_green))
    story.append(Spacer(1,8))
    for fig in figs:
        try:
            png = fig_to_png(fig)
            img = RLImage(io.BytesIO(png), width=6.5*inch, height=3.2*inch)
            story.append(img)
            story.append(Spacer(1,12))
        except Exception:
            pass
    story.append(PageBreak())

    # Page 4 — Symptoms & Risks
    story.append(Paragraph("SYMPTOMS & RISK FACTORS", header_green))
    story.append(Spacer(1,8))
    story.append(Paragraph("<b>Symptoms to Monitor</b>", body_blue))
    symptoms = [
        "Gradual peripheral vision loss",
        "Blurred vision or halos around lights",
        "Difficulty adjusting to darkness",
        "Eye discomfort or headaches"
    ]
    for s in symptoms:
        story.append(Paragraph(f"- {s}", body_blue))
    story.append(Spacer(1,8))
    story.append(Paragraph("<b>Major Risk Factors</b>", body_blue))
    risks = ["Age >60", "Family history", "High IOP", "Thin corneas", "High myopia", "Diabetes / Hypertension"]
    for r in risks:
        story.append(Paragraph(f"- {r}", body_blue))
    story.append(PageBreak())

    # Page 5 — Recommendations
    story.append(Paragraph("RECOMMENDATIONS & ACTION PLAN", header_green))
    story.append(Spacer(1,8))
    if rnflt_cluster == "Glaucoma-like":
        recs = [
            "Schedule ophthalmologist visit within 1-2 weeks.",
            "Request tonometry, visual fields, gonioscopy and dilated optic nerve exam.",
            "Bring this report to your appointment."
        ]
    else:
        recs = [
            "Continue routine annual eye exams.",
            "Document this baseline for future comparison.",
            "Consider earlier follow-up if any symptoms occur."
        ]
    for r in recs:
        story.append(Paragraph(f"- {r}", body_blue))
    story.append(Spacer(1,10))
    story.append(Paragraph("<b>Lifestyle Recommendations</b>", body_blue))
    lifestyle = ["Antioxidant-rich diet", "Regular aerobic activity", "Protect eyes from UV", "Limit caffeine, avoid smoking", "Sleep 7–9 hours"]
    for l in lifestyle:
        story.append(Paragraph(f"- {l}", body_blue))
    story.append(PageBreak())

    # Page 6 — Disclaimer & References
    story.append(Paragraph("IMPORTANT MEDICAL DISCLAIMER", header_green))
    story.append(Spacer(1,8))
    story.append(Paragraph("This OCULAIRE report is generated by an AI screening tool for research and educational purposes only. It is not a clinical diagnosis. Always consult an ophthalmologist for medical decisions.", body_blue))
    story.append(Spacer(1,8))
    story.append(Paragraph("<b>Methodology & References</b>", header_green))
    refs = ["Weinreb RN et al., JAMA 2014", "Tham YC et al., Ophthalmology 2014", "European Glaucoma Society Guidelines 2021"]
    for ref in refs:
        story.append(Paragraph(f"- {ref}", body_small_blue))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ============================================================
# Chatbot (Gemini / fallback)
# ============================================================
MODEL_NAME = "models/gemini-2.5-flash"

def ask_glaucoma_assistant(question, history, api_key):
    if not api_key:
        return "⚠️ API key not configured (Gemini). Please add GEMINI_API_KEY to secrets/environment."
    system_instruction = """You are a medical assistant focused ONLY on glaucoma, OCT, RNFLT, and ocular imaging. Provide concise educational answers and include a disclaimer that you are not providing medical advice."""
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
            # simple HTTP fallback to Google Generative Language API (if allowed)
            convo = ""
            for msg in history[-6:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                convo += f"{role}: {msg['content']}\n\n"
            full_prompt = f"{system_instruction}\n\n{convo}\nUser: {question}\n\nAssistant:"
            url = f"https://generativelanguage.googleapis.com/v1beta/{MODEL_NAME}:generateContent?key={api_key}"
            resp = requests.post(url, headers={"Content-Type":"application/json"}, json={
                "contents":[{"parts":[{"text": full_prompt}]}],
                "generationConfig": {"temperature": 0.7, "maxOutputTokens": 500}
            }, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return f"Error calling API ({resp.status_code}): {resp.text[:200]}"
    except Exception as e:
        return f"Error: {e}"

# ============================================================
# Sidebar UI: inputs, converters, API status
# ============================================================
with st.sidebar:
    st.markdown("### 🔑 API Status")
    if API_KEY:
        st.success("Gemini API key configured")
    else:
        st.error("No Gemini API key (chatbot disabled)")

    st.markdown("---")
    st.markdown("## 🩺 RNFLT Input & Tools")
    rnflt_input_mode = st.radio("RNFLT input type", ["NPZ (recommended)", "Single Image"])

    rnflt_conv_files = st.file_uploader("Upload RNFLT slices (PNG/JPG) to convert to NPZ", accept_multiple_files=True, type=["png","jpg","jpeg"])
    if rnflt_conv_files:
        if st.button("Convert RNFLT slices → .npz"):
            try:
                stacks = [np.array(Image.open(f).convert("L")).astype(np.float32) for f in rnflt_conv_files]
                vol = np.stack(stacks, axis=0)
                buf = io.BytesIO()
                np.savez_compressed(buf, volume=vol)
                buf.seek(0)
                st.success(f"Packed {len(stacks)} slices into volume {vol.shape}")
                st.download_button("⬇️ Download RNFLT .npz", data=buf.getvalue(), file_name="rnflt_volume.npz")
            except Exception as e:
                st.error(f"Conversion error: {e}")

    st.markdown("---")
    st.markdown("## 👁️ B-Scan Input & Tools")
    bscan_input_mode = st.radio("B-scan input type", ["Image (recommended)", "NPZ (sequence)"])

    bscan_conv_files = st.file_uploader("Upload B-scan slices (PNG/JPG) to convert to NPZ", accept_multiple_files=True, type=["png","jpg","jpeg"])
    if bscan_conv_files:
        if st.button("Convert B-scan slices → .npz"):
            try:
                stacks = [np.array(Image.open(f).convert("L")).astype(np.float32) for f in bscan_conv_files]
                vol = np.stack(stacks, axis=0)
                buf = io.BytesIO()
                np.savez_compressed(buf, volume=vol)
                buf.seek(0)
                st.success(f"Packed {len(stacks)} slices into volume {vol.shape}")
                st.download_button("⬇️ Download B-scan .npz", data=buf.getvalue(), file_name="bscan_volume.npz")
            except Exception as e:
                st.error(f"Conversion error: {e}")

    st.markdown("---")
    threshold = st.slider("Thin-zone threshold (µm)", min_value=5, max_value=50, value=10)

# ============================================================
# Main UI: upload panels + analysis trigger
# ============================================================
colA, colB = st.columns(2)

# RNFLT panel
with colA:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🩺 RNFLT Map Analysis")
    rnflt_arr = None
    rnflt_metrics = None
    if rnflt_input_mode == "NPZ (recommended)":
        rnflt_file = st.file_uploader("Upload RNFLT .npz file", type=["npz"], key="rnflt_npz_main")
        if rnflt_file:
            rnflt_arr, rnflt_metrics = process_npz(rnflt_file)
    else:
        rnflt_image = st.file_uploader("Upload single RNFLT image", type=["png","jpg","jpeg"], key="rnflt_img_main")
        if rnflt_image:
            pil = Image.open(rnflt_image).convert("L")
            arr = np.array(pil).astype(float)
            rnflt_arr = arr
            rnflt_metrics = {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr)), "min": float(np.nanmin(arr)), "max": float(np.nanmax(arr))}
    st.markdown("</div>", unsafe_allow_html=True)

# B-scan panel
with colB:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("👁️ B-Scan Analysis")
    bscan_file = None
    bscan_npz = None
    if bscan_input_mode == "Image (recommended)":
        bscan_file = st.file_uploader("Upload B-scan image", type=["png","jpg","jpeg"], key="bscan_img_main")
    else:
        bscan_npz = st.file_uploader("Upload B-scan NPZ volume", type=["npz"], key="bscan_npz_main")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Start analysis if any input present
analysis_trigger = (rnflt_arr is not None) or (bscan_file is not None) or (bscan_npz is not None)

# local variables to pass to PDF later
pdf_figs_local = []
pdf_context = {}

if analysis_trigger:
    st.markdown("## Analysis")
    severity_overall = 0.0
    figs = []

    # RNFLT analysis
    label_r = None
    if rnflt_arr is not None:
        try:
            metrics = rnflt_metrics or {}
            if scaler is not None and kmeans is not None:
                X = np.array([[metrics.get("mean",0), metrics.get("std",0), metrics.get("min",0), metrics.get("max",0)]])
                try:
                    Xs = scaler.transform(X)
                    cluster = int(kmeans.predict(Xs)[0])
                    label_r = "Glaucoma-like" if cluster == thin_cluster else "Healthy-like"
                except Exception:
                    cluster = "?"
                    label_r = "Unknown"
            else:
                cluster = "?"
                label_r = "Unknown"

            if avg_healthy is not None:
                diff, risk, sev = compute_risk_map(rnflt_arr, avg_healthy, -threshold)
            else:
                diff = rnflt_arr - np.nanmean(rnflt_arr)
                risk = np.where(diff < -threshold, diff, np.nan)
                sev = 0.0

            severity_overall = max(severity_overall, float(sev))

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Status", label_r)
            c2.metric("Mean RNFLT", f"{metrics.get('mean',0):.2f}")
            c3.metric("Std Dev", f"{metrics.get('std',0):.2f}")
            c4.metric("Cluster", str(cluster))

            st.markdown(render_severity := """
            <div style="width:100%; text-align:center; margin-top:10px;">
                <div style="width:100%; max-width:900px; margin:auto; height:24px; background:rgba(0,255,0,0.08); border-radius:18px; border:1px solid rgba(0,255,0,0.18); overflow:hidden;">
                    <div style="height:100%; width:40%; background:linear-gradient(90deg,#39ff14,#b8ff8a); box-shadow:0 0 30px #39ff14;"></div>
                </div>
                <div style="margin-top:8px; font-size:16px; color:#b8ffb8;"><b>{:.1f}% Severity</b></div>
            </div>
            """.format(sev), unsafe_allow_html=True)

            # Build RNFLT figure (3 panel)
            fig, axes = plt.subplots(1, 3, figsize=(16,5))
            im0 = axes[0].imshow(rnflt_arr, cmap='turbo'); axes[0].axis('off'); axes[0].set_title("RNFLT Map")
            plt.colorbar(im0, ax=axes[0], shrink=0.8)
            im1 = axes[1].imshow(diff, cmap='bwr', vmin=-30, vmax=30); axes[1].axis('off'); axes[1].set_title("Difference vs Healthy")
            plt.colorbar(im1, ax=axes[1], shrink=0.8)
            im2 = axes[2].imshow(risk, cmap='hot'); axes[2].axis('off'); axes[2].set_title("Risk Map")
            plt.colorbar(im2, ax=axes[2], shrink=0.8)
            fig.patch.set_facecolor("#020802")
            st.pyplot(fig)
            figs.append(fig)
        except Exception as e:
            st.error(f"RNFLT processing error: {e}")

    # B-scan analysis
    label_b = None
    conf = 0.0
    if bscan_file:
        try:
            pil = Image.open(bscan_file).convert("L")
            batch, proc = preprocess_bscan(pil)
            if b_model is not None:
                try:
                    pred_raw = float(b_model.predict(batch, verbose=0)[0][0])
                except Exception:
                    pred_raw = 0.0
                label_b = "Glaucoma-like" if pred_raw > 0.5 else "Healthy-like"
                conf = pred_raw * 100 if label_b == "Glaucoma-like" else (1-pred_raw)*100
            else:
                label_b = "Unknown"
                conf = 0.0
            severity_overall = max(severity_overall, conf)

            col1, col2 = st.columns(2)
            col1.metric("CNN Prediction", label_b)
            col2.metric("Confidence", f"{conf:.2f}%")

            # Grad-CAM visualization if model exists
            heat = gradcam(batch, b_model) if b_model is not None else None
            if heat is not None:
                heat_r = cv2.resize(heat, (proc.shape[1], proc.shape[0]))
                hm = (heat_r * 255).astype(np.uint8)
                hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
                overlay = (np.stack([proc]*3, axis=-1)*255).astype(np.uint8)
                overlay = cv2.addWeighted(overlay, 0.6, hm_color, 0.4, 0)
                st.image([pil, overlay], caption=["Original B-Scan", "Grad-CAM Overlay"], use_column_width=True)

                fig2, ax2 = plt.subplots(1, 2, figsize=(12,5))
                ax2[0].imshow(pil, cmap='gray'); ax2[0].axis('off'); ax2[0].set_title("B-Scan")
                ax2[1].imshow(overlay); ax2[1].axis('off'); ax2[1].set_title("Grad-CAM Overlay")
                fig2.patch.set_facecolor("#020802")
                figs.append(fig2)
            else:
                st.image(pil, caption="B-Scan (original)", use_column_width=True)
        except Exception as e:
            st.error(f"B-scan processing error: {e}")

    # Overall severity display
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align:center; color:#39ff14;'>Overall Severity Index</h3>", unsafe_allow_html=True)
    # re-use simple progress display
    st.markdown(f"<div style='text-align:center; color:#b8ffb8; font-weight:700;'>{severity_overall:.1f}%</div>", unsafe_allow_html=True)

    # Offer downloads & save context for PDF
    if len(figs) > 0:
        png_bytes = fig_to_png(figs[0])
        st.download_button("📸 Download RNFLT PNG", data=png_bytes, file_name="oculaire_rnflt.png")
        # Save into session_state to pass to PDF generator
        st.session_state.pdf_figs = figs
        st.session_state.pdf_data = {
            "rnflt_metrics": rnflt_metrics,
            "rnflt_cluster": label_r,
            "rnflt_severity": severity_overall,
            "bscan_label": label_b,
            "bscan_conf": conf
        }

        if st.button("📄 Generate Full Medical Report (PDF)"):
            st.session_state.trigger_pdf = True

    # Allow saving run to DB
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Save run to history (local SQLite)")
    patient_name = st.text_input("Patient name (optional)", key="patient_name_save")
    patient_id = st.text_input("Patient ID (optional)", key="patient_id_save")
    if st.button("💾 Save run"):
        try:
            png_save = fig_to_png(figs[0]) if len(figs)>0 else None
            save_run(patient_name or '-', patient_id or '-', rnflt_metrics or {}, severity_overall, quality=0.0, bscan_label=label_b, bscan_conf=conf, png_bytes=png_save)
            st.success("Run saved to local history.")
        except Exception as e:
            st.error(f"Error saving run: {e}")

# ============================================================
# Generate PDF if triggered
# ============================================================
if st.session_state.get("trigger_pdf", False):
    try:
        figs_for_pdf = st.session_state.get("pdf_figs", [])
        pdf_ctx = st.session_state.get("pdf_data", {})
        final_pdf = generate_full_pdf(
            figs=figs_for_pdf,
            rnflt_metrics=pdf_ctx.get("rnflt_metrics"),
            rnflt_cluster=pdf_ctx.get("rnflt_cluster"),
            rnflt_severity=pdf_ctx.get("rnflt_severity"),
            bscan_label=pdf_ctx.get("bscan_label"),
            bscan_conf=pdf_ctx.get("bscan_conf")
        )
        st.session_state.trigger_pdf = False
        st.success("PDF generated.")
        st.download_button("📄 Download Full PDF Report (6 pages)", data=final_pdf, file_name="OCULAIRE_Full_Report.pdf", mime="application/pdf", use_container_width=True)
    except Exception as e:
        st.error(f"PDF generation error: {e}")

# ============================================================
# Saved runs list
# ============================================================
st.markdown("---")
st.subheader("Saved Runs (recent)")
runs = list_runs(limit=20)
if runs:
    for r in runs:
        rid, rpatient, rpid, rts, rmetrics, rsev, rqual = r
        cols = st.columns([3,1,1])
        with cols[0]:
            st.markdown(f"**{rpatient}** (ID: {rpid}) — {rts}")
            st.markdown(f"Severity: {rsev:.2f}% — Metrics: {rmetrics}")
        with cols[1]:
            if st.button(f"⬇️ Download PNG #{rid}", key=f"dl_{rid}"):
                pngb = load_run_png(rid)
                if pngb:
                    st.download_button(f"Download run {rid} PNG", data=pngb, file_name=f"oculaire_run_{rid}.png", mime="image/png")
        with cols[2]:
            if st.button(f"🗑️ Delete #{rid}", key=f"del_{rid}"):
                # simple delete function
                try:
                    with db_cursor() as (conn, c):
                        c.execute("DELETE FROM runs WHERE id=?", (rid,))
                    st.success(f"Deleted run {rid} (refresh to see changes).")
                except Exception as e:
                    st.error(f"Delete error: {e}")
else:
    st.info("No saved runs yet — run analysis and save a run to populate history.")

# ============================================================
# Floating chatbot expander
# ============================================================
st.markdown('<div class="floating-expander">', unsafe_allow_html=True)
with st.expander("💬 Ask AI assistant (glaucoma)"):
    st.markdown("<div class='chat-header'>🤖 Glaucoma Q&A Assistant</div>", unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-msg'><strong>You:</strong> {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-msg'><strong>🤖:</strong> {msg['content']}</div>", unsafe_allow_html=True)

    user_q = st.text_input("Your question:", key="chat_input", placeholder="e.g., What does RNFLT thinning mean?")
    col1, col2 = st.columns([4,1])
    with col1:
        send_btn = st.button("📤 Send")
    with col2:
        clear_btn = st.button("🗑️ Clear chat")

    if send_btn and user_q:
        if not API_KEY:
            st.error("No Gemini API key configured. Add GEMINI_API_KEY in secrets or env to enable chatbot.")
        else:
            st.session_state.chat_history.append({"role":"user","content":user_q})
            reply = ask_glaucoma_assistant(user_q, st.session_state.chat_history, API_KEY)
            st.session_state.chat_history.append({"role":"assistant","content":reply})
            st.experimental_rerun()

    if clear_btn:
        st.session_state.chat_history = []
        st.experimental_rerun()

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="footer-note">OCULAIRE Neon Lab v5 — For research use only</div>', unsafe_allow_html=True)
