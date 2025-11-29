# ============================================================
#  OCULAIRE — Neon Green UI + Fast Glow Header + Chatbot
#  Extended: Scan Quality Score + Improved Chatbot (explain scan) + QR code in PDF
# ============================================================

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
from datetime import datetime

# ReportLab for multi-page PDF (ensure installed in environment)
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# Optional qrcode for embedding QR in PDF
try:
    import qrcode
    QR_AVAILABLE = True
except Exception:
    QR_AVAILABLE = False

# Try Gemini SDK
try:
    import google.generativeai as genai
    USE_SDK = True
except Exception:
    USE_SDK = False

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="OCULAIRE: Glaucoma Detection Dashboard",
    layout="wide",
    page_icon="👁️"
)

# -----------------------
# Neon Green Theme + Fast Glow Header (UI)
# -----------------------
st.markdown("""
<style>

:root {
  --green: #39ff14;
  --green-soft: #aaffaa;
  --panel-green: rgba(0,255,0,0.08);
  --bg-soft: #000f00;
}

/* App Background */
.stApp {
  background: radial-gradient(circle at 20% 20%, #002200, #000000 80%);
  color: #ccffcc;
  font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Neon cards */
.card {
  background: rgba(0,255,0,0.05);
  border: 1px solid rgba(0,255,0,0.3);
  border-radius: 12px;
  padding: 18px;
  box-shadow: 0 0 22px rgba(0,255,0,0.2);
}

/* Sidebar */
.css-1d391kg, .css-1lcbmhc {
  background: rgba(0,255,0,0.08) !important;
  border-right: 1px solid rgba(0,255,0,0.2);
}

label, h2, h3, h4, h5 {
  color: var(--green) !important;
}

/* FAST intense neon pulse */
@keyframes intenseGlow {
  0% { transform: scale(1); text-shadow: 0 0 18px var(--green), 0 0 30px var(--green); }
  25% { transform: scale(1.06); text-shadow: 0 0 32px var(--green), 0 0 55px var(--green); }
  50% { transform: scale(1.1); text-shadow: 0 0 50px var(--green), 0 0 80px var(--green); }
  75% { transform: scale(1.05); text-shadow: 0 0 30px var(--green), 0 0 60px var(--green); }
  100% { transform: scale(1); text-shadow: 0 0 20px var(--green), 0 0 40px var(--green); }
}

/* Severity bar */
.sev-inner {
  background: linear-gradient(90deg, #39ff14, #b8ff8a);
  box-shadow: 0 0 35px #39ff14;
  transition: width 0.9s ease-in-out;
}

/* Chat bubbles */
.user-msg {
  background: rgba(0,255,0,0.12);
  border-left: 3px solid var(--green);
  padding: 12px; border-radius: 8px;
  color: #e8ffe8;
}
.assistant-msg {
  background: rgba(100,255,150,0.15);
  border-left: 3px solid #7dff7d;
  padding: 12px; border-radius: 8px;
  color: #e8ffe8;
}

</style>

<!-- FAST GLOWING NEON TITLE -->
<div style='text-align:center; margin-top:20px;'>
  <h1 style="
    font-size:72px;
    font-weight:900;
    letter-spacing:12px;
    background: linear-gradient(90deg, #39ff14, #b8ff8a);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    animation: intenseGlow 1.4s infinite ease-in-out;
  ">
    OCULAIRE
  </h1>

  <h3 style="color:#aaffaa; margin-top:-10px;">
    Illuminating Vision. Detecting Glaucoma.
  </h3>
</div>

""", unsafe_allow_html=True)

# -----------------------
# Session State defaults (core objects)
# -----------------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'trigger_pdf' not in st.session_state:
    st.session_state['trigger_pdf'] = False
if 'pdf_figs' not in st.session_state:
    st.session_state['pdf_figs'] = []
if 'pdf_context' not in st.session_state:
    st.session_state['pdf_context'] = {}

# -----------------------
# API Key helper
# -----------------------
def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    env = os.getenv("GEMINI_API_KEY")
    if env:
        return env
    return None

API_KEY = get_api_key()

# -----------------------
# Matplotlib Theme
# -----------------------
plt.style.use('dark_background')
plt.rcParams.update({
    "figure.facecolor": "#020802",
    "axes.facecolor": "#020802",
    "axes.edgecolor": "#39ff14",
    "axes.labelcolor": "#aaffaa",
    "xtick.color": "#39ff14",
    "ytick.color": "#39ff14",
    "text.color": "#ccffcc",
    "font.size": 12,
})

# ============================================================
#  Processing & helpers (load models, npz, preprocess, gradcam)
# ============================================================

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

def process_npz(f):
    try:
        buf = io.BytesIO(f.getvalue())
        data = np.load(buf, allow_pickle=True)
        arr = data["volume"] if "volume" in data else data[data.files[0]]
        if arr.ndim == 3:
            arr = arr[0, :, :]
        arr = arr.astype(float)
        vals = arr.flatten()
        metrics = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals)),
            "min": float(np.nanmin(vals)),
            "max": float(np.nanmax(vals))
        }
        return arr, metrics
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
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=150)
    buf.seek(0)
    return buf.getvalue()

def convert_images_to_npz(files):
    try:
        stacks = []
        for f in files:
            im = Image.open(f).convert("L")
            arr = np.array(im).astype(np.float32)
            stacks.append(arr)
        volume = np.stack(stacks, axis=0)
        buf = io.BytesIO()
        np.savez_compressed(buf, volume=volume)
        buf.seek(0)
        return buf.getvalue(), volume.shape
    except Exception as e:
        st.error(f"Conversion error: {e}")
        return None, None

# ============================================================
# New: AI Scan Quality Score (SNR + Blur)
# ============================================================
def estimate_snr(img_gray):
    arr = np.array(img_gray).astype(np.float32)
    if arr.max() > 1.0:
        arr_n = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
    else:
        arr_n = arr
    mean_signal = np.nanmean(arr_n)
    std_noise = np.nanstd(arr_n)
    if std_noise < 1e-6:
        return 100.0
    snr = mean_signal / std_noise
    # map SNR to 0..100
    snr_score = 100 * (np.tanh((snr - 1.0) / 1.5) * 0.5 + 0.5)
    return float(np.clip(snr_score, 0, 100))

def estimate_sharpness(img_gray):
    arr = np.array(img_gray).astype(np.uint8)
    lap = cv2.Laplacian(arr, cv2.CV_64F)
    var_lap = float(np.var(lap))
    sharp_score = 100 * (np.tanh((var_lap - 30) / 120) * 0.5 + 0.5)
    return float(np.clip(sharp_score, 0, 100)), var_lap

def compute_quality_from_image_pil(pil_img):
    try:
        gray = pil_img.convert("L")
        snr_s = estimate_snr(np.array(gray))
        sharp_s, var_lap = estimate_sharpness(gray)
        combined = 0.6 * sharp_s + 0.4 * snr_s
        return float(np.clip(combined, 0, 100)), {"snr_score": snr_s, "sharp_score": sharp_s, "var_lap": var_lap}
    except Exception:
        return 0.0, {"snr_score": 0.0, "sharp_score": 0.0, "var_lap": 0.0}

def compute_quality_from_array(arr):
    try:
        a = arr.copy().astype(np.float32)
        a = a - np.nanmin(a)
        if np.nanmax(a) > 0:
            a = (a / (np.nanmax(a))) * 255.0
        else:
            a = a * 0.0
        a = np.nan_to_num(a).astype(np.uint8)
        pil = Image.fromarray(a)
        return compute_quality_from_image_pil(pil)
    except Exception:
        return 0.0, {"snr_score": 0.0, "sharp_score": 0.0, "var_lap": 0.0}

# -----------------------
# Severity renderer
# -----------------------
def render_severity(pct):
    pct = max(0.0, min(100.0, float(pct)))
    html = f"""
    <div style="width:100%; text-align:center; margin-top:20px;">
        <div style="width:100%; max-width:900px; margin:auto; height:28px;
                    background:rgba(0,255,0,0.1); border-radius:18px;
                    border:1px solid rgba(0,255,0,0.25); overflow:hidden;">
            <div id="sev_inner" style="
                height:100%; width:0%;
                background:linear-gradient(90deg,#39ff14,#b8ff8a);
                box-shadow:0 0 30px #39ff14;
                transition:width 1s ease-out;">
            </div>
        </div>
        <div style="margin-top:10px; font-size:20px; color:#b8ffb8;">
            <b>{pct:.1f}% Severity</b>
        </div>
    </div>

    <script>
    setTimeout(function(){{
        var el = document.getElementById("sev_inner");
        if (el) {{
            el.style.width = "{pct:.1f}%";
        }}
    }}, 80);
    </script>
    """
    return html

# ============================================================
# Sidebar UI
# ============================================================
with st.sidebar:
    st.markdown("### 🔑 API Status")
    if API_KEY:
        st.success("Gemini API key configured")
    else:
        st.error("API key missing")
        st.info("Add GEMINI_API_KEY to secrets or environment.")

    st.markdown("---")
    st.markdown("## 🩺 RNFLT Input & Tools")
    rnflt_input_mode = st.radio("RNFLT input type", ["NPZ (recommended)", "Single Image"]) 

    rnflt_conv = st.file_uploader("Upload RNFLT slices → convert to .npz", accept_multiple_files=True, type=["png","jpg","jpeg"]) 
    if rnflt_conv:
        if st.button("Convert RNFLT slices to NPZ"):
            data, shape = convert_images_to_npz(rnflt_conv)
            if data:
                st.success(f"Packed slices into volume shape {shape}")
                st.download_button("⬇️ Download RNFLT volume", data=data, file_name="rnflt_volume.npz")

    st.markdown("---")
    st.markdown("## 👁️ B-Scan Input & Tools")
    bscan_input_mode = st.radio("B-scan input type", ["Image (recommended)", "NPZ (multi-slice)"])

    bscan_conv = st.file_uploader("Upload B-scan slices → convert to .npz", accept_multiple_files=True, type=["png","jpg","jpeg"]) 
    if bscan_conv:
        if st.button("Convert B-scan slices to NPZ"):
            data, shape = convert_images_to_npz(bscan_conv)
            if data:
                st.success(f"Packed B-scan slices into {shape}")
                st.download_button("⬇️ Download B-scan volume", data=data, file_name="bscan_volume.npz")

    st.markdown("---")
    threshold = st.slider("Thin-zone threshold (µm)", min_value=5, max_value=50, value=10)

# ============================================================
# Main layout: RNFLT and B-scan upload panels
# ============================================================
colA, colB = st.columns(2)

with colA:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🩺 RNFLT Map Analysis")

    rnflt_arr = None
    rnflt_metrics = None
    rnflt_pil = None

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
            rnflt_pil = pil
            vals = arr.flatten()
            rnflt_metrics = {
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals)),
                "min": float(np.nanmin(vals)),
                "max": float(np.nanmax(vals)),
            }
    st.markdown("</div>", unsafe_allow_html=True)

with colB:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("👁️ B-Scan Slice Analysis")

    bscan_file = None
    bscan_npz = None

    if bscan_input_mode == "Image (recommended)":
        bscan_file = st.file_uploader("Upload B-scan image", type=["png","jpg","jpeg"], key="bscan_img_main")
    else:
        bscan_npz = st.file_uploader("Upload B-scan NPZ volume", type=["npz"], key="bscan_npz_main")
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Run analysis if any input present
# ============================================================
analysis_trigger = (rnflt_arr is not None) or (bscan_file is not None) or (bscan_npz is not None)

if analysis_trigger:
    st.markdown("<hr>", unsafe_allow_html=True)

    run_figs = []
    run_rnflt_metrics = rnflt_metrics
    run_label_r = None
    run_label_b = None
    run_conf = 0.0
    run_severity = 0.0

    # RNFLT analysis
    if rnflt_arr is not None:
        try:
            metrics = rnflt_metrics or {}
            if scaler is not None and kmeans is not None and metrics:
                X = np.array([[metrics["mean"], metrics["std"], metrics["min"], metrics["max"]]])
                Xs = scaler.transform(X)
                cluster = int(kmeans.predict(Xs)[0])
                run_label_r = "Glaucoma-like" if cluster == thin_cluster else "Healthy-like"
            else:
                cluster = "?"
                run_label_r = "Unknown"

            if avg_healthy is not None:
                diff, risk, sev = compute_risk_map(rnflt_arr, avg_healthy, -threshold)
            else:
                diff = rnflt_arr - np.nanmean(rnflt_arr)
                risk = np.where(diff < -threshold, diff, np.nan)
                sev = 0.0

            run_severity = max(run_severity, sev)

            # Quality for RNFLT map
            q_rnflt, qmeta_rnflt = compute_quality_from_array(rnflt_arr)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Status", run_label_r)
            if metrics:
                c2.metric("Mean RNFLT", f"{metrics['mean']:.2f}")
                c3.metric("Std Dev", f"{metrics['std']:.2f}")
            else:
                c2.metric("Mean RNFLT", "-")
                c3.metric("Std Dev", "-")
            c4.metric("Cluster", str(cluster))

            st.markdown(f"**RNFLT Scan Quality:** {q_rnflt:.1f}% — (sharp:{qmeta_rnflt['sharp_score']:.1f}, snr:{qmeta_rnflt['snr_score']:.1f})")

            fig, axes = plt.subplots(1, 3, figsize=(18,6))
            im0 = axes[0].imshow(rnflt_arr, cmap='turbo'); axes[0].axis('off'); axes[0].set_title("RNFLT Map")
            plt.colorbar(im0, ax=axes[0], shrink=0.85)
            im1 = axes[1].imshow(diff, cmap='bwr', vmin=-30, vmax=30); axes[1].axis('off'); axes[1].set_title("Difference vs Healthy")
            plt.colorbar(im1, ax=axes[1], shrink=0.85)
            im2 = axes[2].imshow(risk, cmap='hot'); axes[2].axis('off'); axes[2].set_title("Risk Map")
            plt.colorbar(im2, ax=axes[2], shrink=0.85)
            fig.patch.set_facecolor("#020802")
            st.pyplot(fig)
            run_figs.append(fig)

            run_rnflt_metrics = metrics
        except Exception as e:
            st.error(f"RNFLT Error: {e}")

    # B-scan analysis
    if bscan_file:
        try:
            pil = Image.open(bscan_file).convert("L")
            batch, proc = preprocess_bscan(pil)
            # compute quality
            q_bscan, qmeta_bscan = compute_quality_from_image_pil(pil)

            if b_model is not None:
                pred_raw = float(b_model.predict(batch, verbose=0)[0][0])
                run_label_b = "Glaucoma-like" if pred_raw > 0.5 else "Healthy-like"
                run_conf = pred_raw*100 if run_label_b == "Glaucoma-like" else (1 - pred_raw)*100
            else:
                run_label_b = "Unknown"
                run_conf = 0.0

            run_severity = max(run_severity, run_conf)

            col1, col2 = st.columns(2)
            col1.metric("CNN Prediction", run_label_b)
            col2.metric("Confidence", f"{run_conf:.2f}%")
            st.markdown(f"**B-scan Quality:** {q_bscan:.1f}% — (sharp:{qmeta_bscan['sharp_score']:.1f}, snr:{qmeta_bscan['snr_score']:.1f})")
            st.markdown(render_severity(run_conf), unsafe_allow_html=True)

            heat = gradcam(batch, b_model) if b_model else None
            if heat is not None:
                heat_r = cv2.resize(heat, (proc.shape[1], proc.shape[0]))
                hm = (heat_r * 255).astype(np.uint8)
                hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
                overlay = (np.stack([proc]*3, axis=-1) * 255).astype(np.uint8)
                overlay = cv2.addWeighted(overlay, 0.6, hm_color, 0.4, 0)
                st.image([pil, overlay], caption=["Original B-Scan", "Grad-CAM Overlay"], use_column_width=True)
                fig2, ax2 = plt.subplots(1,2,figsize=(14,6))
                ax2[0].imshow(pil, cmap='gray'); ax2[0].axis('off'); ax2[0].set_title("B-Scan")
                ax2[1].imshow(overlay); ax2[1].axis('off'); ax2[1].set_title("Grad-CAM Overlay")
                fig2.patch.set_facecolor("#020802")
                run_figs.append(fig2)
            else:
                st.image(pil, caption="Original B-Scan", use_column_width=True)
        except Exception as e:
            st.error(f"B-Scan Error: {e}")

    # overall severity display
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; color:#39ff14;'>Overall Severity Index</h3>", unsafe_allow_html=True)
    st.markdown(render_severity(run_severity), unsafe_allow_html=True)

    # save run context into session_state for PDF + chatbot
    if run_figs:
        st.session_state['pdf_figs'] = run_figs
        st.session_state['pdf_context'] = {
            "rnflt_metrics": run_rnflt_metrics,
            "rnflt_cluster": run_label_r,
            "rnflt_severity": run_severity,
            "bscan_label": run_label_b,
            "bscan_conf": run_conf,
            # quality: average of available
            "quality": float(np.mean([v for v in [ (compute_quality_from_array(rnflt_arr)[0] if rnflt_arr is not None else None),
                                                  (compute_quality_from_image_pil(Image.open(bscan_file).convert('L'))[0] if bscan_file else None) ] if v is not None]) if True else 0.0)
        }
        # quick downloads
        png_bytes = fig_to_png(run_figs[0])
        st.download_button("📸 Download RNFLT PNG", data=png_bytes, file_name="oculaire_rnflt.png")
        if st.button("📄 Generate Full Medical Report (PDF)"):
            st.session_state['trigger_pdf'] = True

# ============================================================
# Improved Chatbot: includes optional 'explain my scan' context
# ============================================================
MODEL_NAME = "models/gemini-2.5-flash"

def build_scan_summary_for_prompt():
    ctx = st.session_state.get('pdf_context', {})
    parts = []
    if not ctx:
        return "No scan loaded in session."
    rm = ctx.get('rnflt_metrics')
    if rm:
        parts.append(f"RNFLT Mean {rm.get('mean',0):.2f} µm, Std {rm.get('std',0):.2f} µm, Min {rm.get('min',0):.2f}, Max {rm.get('max',0):.2f}.")
    rc = ctx.get('rnflt_cluster')
    if rc:
        parts.append(f"RNFLT label: {rc}.")
    sev = ctx.get('rnflt_severity')
    if sev is not None:
        parts.append(f"Severity index: {sev:.1f}%.")
    bl = ctx.get('bscan_label')
    if bl:
        parts.append(f"B-scan label: {bl} (confidence {ctx.get('bscan_conf',0):.1f}%).")
    q = ctx.get('quality')
    if q is not None:
        parts.append(f"Scan quality: {q:.1f}% (higher is better).")
    return " ".join(parts)

def ask_glaucoma_assistant(question, history, api_key):
    """Call Google Gemini API with glaucoma-specific context. If question asks to 'explain' include scan summary."""
    if not api_key or not api_key.strip():
        return "⚠️ Please configure your Google Gemini API key (see sidebar)."

    system_instruction = """You are a specialized medical AI assistant focused exclusively on glaucoma, OCT imaging, RNFLT measurements, optic nerve health, intraocular pressure, and glaucoma diagnosis/treatment.
Answer concisely (<= 200 words) and include a brief disclaimer that this is educational information not medical advice.
If user asks to 'explain my scan' or similar, use the provided scan summary context to explain results and provide practical next steps (tests to request, urgency)."""

    # detect explain intent
    q_lower = question.lower()
    include_scan = any(kw in q_lower for kw in ["explain my scan", "explain my result", "explain the scan", "explain my rnflt", "explain my b-scan", "explain this scan"])
    scan_summary = build_scan_summary_for_prompt() if include_scan else ""

    try:
        if USE_SDK:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(MODEL_NAME)
            chat_history = []
            for msg in history[-8:]:
                role = "user" if msg["role"] == "user" else "assistant"
                chat_history.append({"role": role, "parts": [msg["content"]]})
            chat = model.start_chat(history=chat_history)
            prompt = system_instruction + ("\n\nScan summary: " + scan_summary if include_scan else "")
            prompt += f"\n\nUser question: {question}"
            response = chat.send_message(prompt)
            return response.text
        else:
            conversation_context = ""
            for msg in history[-8:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                conversation_context += f"{role}: {msg['content']}\n\n"
            full_prompt = system_instruction + ("\n\nScan summary: " + scan_summary if include_scan else "") + f"\n\n{conversation_context}User: {question}\n\nAssistant:"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": full_prompt}]}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 500}
                },
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            elif response.status_code == 403:
                return "🔑 API key invalid or restricted."
            else:
                return f"❌ Error ({response.status_code}): {response.text[:200]}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ============================================================
# Chat UI — floating expander
# ============================================================
st.markdown('<div class="floating-expander">', unsafe_allow_html=True)
with st.expander("💬 Ask AI assistant", expanded=False):
    st.markdown("<div class='chat-header'>🤖 Glaucoma Q&A Assistant</div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#b8ffb8; font-size:13px; margin-bottom:12px;'>Ask me anything about glaucoma, OCT imaging, RNFLT, or request 'explain my scan'</p>", unsafe_allow_html=True)

    # show history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-msg'><strong>You:</strong> {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-msg'><strong>🤖:</strong> {msg['content']}</div>", unsafe_allow_html=True)

    user_question = st.text_input("Your question:", key="chat_input", placeholder="e.g., Explain my scan; What does RNFLT thinning mean?", label_visibility="collapsed")
    col1, col2 = st.columns([4,1])
    with col1:
        send_btn = st.button("📤 Send", use_container_width=True)
    with col2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)

    if send_btn and user_question:
        if not API_KEY:
            st.error("❌ API key not configured. See sidebar.")
        else:
            with st.spinner("🔍 Generating answer..."):
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                reply = ask_glaucoma_assistant(user_question, st.session_state.chat_history, API_KEY)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
            try:
                st.experimental_rerun()
            except Exception:
                pass

    if clear_btn:
        st.session_state.chat_history = []
        try:
            st.experimental_rerun()
        except Exception:
            pass

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PDF generation engine (with clinical blue body and QR)
# ============================================================
def generate_full_pdf(figs,
                      rnflt_metrics=None,
                      rnflt_cluster=None,
                      rnflt_severity=None,
                      bscan_label=None,
                      bscan_conf=None,
                      quality=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=40, bottomMargin=40, leftMargin=50, rightMargin=50)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("TitleGlow", parent=styles["Title"], fontSize=28, textColor=colors.HexColor("#00eaff"), leading=32, alignment=1)
    subtitle = ParagraphStyle("Subtitle", parent=styles["Heading2"], fontSize=14, textColor=colors.HexColor("#ff40c4"), alignment=1, leading=18)
    header_green = ParagraphStyle("HeaderGreen", parent=styles["Heading2"], textColor=colors.HexColor("#39ff14"), fontSize=18)
    clinical_blue = ParagraphStyle("ClinicalBlue", parent=styles["BodyText"], fontSize=11, leading=16, textColor=colors.HexColor("#0b6aff"))
    body_small_blue = ParagraphStyle("SmallBlue", parent=styles["BodyText"], fontSize=9, leading=12, textColor=colors.HexColor("#0b6aff"))

    story = []

    # COVER
    story.append(Paragraph("OCULAIRE", title))
    story.append(Paragraph("AI-Powered Glaucoma Screening Report", subtitle))
    story.append(Spacer(1, 12))

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

    # Executive summary box
    if (rnflt_cluster == "Glaucoma-like") or (bscan_label == "Glaucoma-like"):
        risk_color = colors.HexColor("#ff6b6b")
        risk_text = "⚠️ ABNORMAL PATTERNS DETECTED"
        risk_level = ("HIGH" if (rnflt_severity or 0) >= 60 else "MODERATE" if (rnflt_severity or 0) >= 30 else "LOW-MODERATE")
    else:
        risk_color = colors.HexColor("#39ff14")
        risk_text = "✅ NORMAL PATTERNS DETECTED"
        risk_level = "LOW"

    exec_para = Paragraph(f"<b>Status:</b> {risk_text}<br/><b>Risk Level:</b> {risk_level}<br/><b>Severity Index:</b> {(rnflt_severity or 0):.1f}%<br/><b>CNN Confidence:</b> {(bscan_conf or 0):.1f}%<br/><b>Scan Quality:</b> {(quality or 0):.1f}%", clinical_blue)
    box_table = Table([[exec_para]], colWidths=[450])
    box_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), risk_color),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.white),
        ("BOX", (0,0), (-1,-1), 1.5, colors.black),
        ("LEFTPADDING", (0,0), (-1,-1), 12),
        ("RIGHTPADDING", (0,0), (-1,-1), 12),
        ("TOPPADDING", (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
    ]))
    story.append(box_table)
    story.append(Spacer(1, 12))

    # QR code (if available) — link points to a placeholder URL containing report_id
    if QR_AVAILABLE:
        try:
            qr_url = f"https://example.com/oculaire/report/{report_id}"
            qr = qrcode.make(qr_url)
            qr_buf = io.BytesIO()
            qr.save(qr_buf, format="PNG")
            qr_buf.seek(0)
            img = RLImage(qr_buf, width=1.25*inch, height=1.25*inch)
            # place to the right of metadata (simple append)
            story.append(Spacer(1, 6))
            story.append(img)
            story.append(Spacer(1, 6))
        except Exception:
            pass

    story.append(PageBreak())

    # PAGE 2: Clinical interpretation and RNFLT stats (blue descriptions)
    story.append(Paragraph("CLINICAL INTERPRETATION", header_green))
    story.append(Spacer(1, 10))
    if rnflt_cluster == "Glaucoma-like":
        story.append(Paragraph("The AI analysis has detected <b>patterns consistent with glaucomatous changes</b> in your retinal nerve fiber layer structure. This includes thinning relative to the healthy reference database.", clinical_blue))
    else:
        story.append(Paragraph("The retinal nerve fiber layer thickness pattern appears <b>within normal ranges</b>. No significant signs of glaucomatous thinning are identified.", clinical_blue))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Key Findings</b>", clinical_blue)); story.append(Spacer(1,6))

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
            "- Pattern matches healthy reference population."
        ]
    for f in findings:
        story.append(Paragraph(f, clinical_blue))
    story.append(Spacer(1,12))

    story.append(Paragraph("<b>RNFLT Measurements</b>", header_green))
    story.append(Spacer(1,8))
    if rnflt_metrics:
        rnflt_tbl = [
            ["Mean Thickness", f"{rnflt_metrics.get('mean',0):.2f} μm"],
            ["Standard Deviation", f"{rnflt_metrics.get('std',0):.2f} μm"],
            ["Minimum", f"{rnflt_metrics.get('min',0):.2f} μm"],
            ["Maximum", f"{rnflt_metrics.get('max',0):.2f} μm"],
        ]
        rnflt_table = Table(rnflt_tbl, colWidths=[200,200])
        rnflt_table.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1), colors.Color(0.02,0.05,0.18)),
            ("TEXTCOLOR",(0,0),(-1,-1), colors.HexColor("#0b6aff")),
            ("GRID",(0,0),(-1,-1),0.5, colors.HexColor("#00eaff")),
        ]))
        story.append(rnflt_table)
    story.append(PageBreak())

    # PAGE 3: Figures
    story.append(Paragraph("DETAILED VISUAL ANALYSIS", header_green))
    story.append(Spacer(1,8))
    for fig in figs or []:
        try:
            png = fig_to_png(fig)
            img = RLImage(io.BytesIO(png), width=6.5*inch, height=3.2*inch)
            story.append(img)
            story.append(Spacer(1,12))
        except Exception:
            pass
    story.append(PageBreak())

    # PAGE 4: Symptoms & Risk Factors
    story.append(Paragraph("SYMPTOMS & RISK FACTORS", header_green)); story.append(Spacer(1,8))
    story.append(Paragraph("<b>Symptoms to Monitor</b>", clinical_blue))
    symptoms = ["Gradual peripheral vision loss", "Blurred vision or halos", "Difficulty adjusting to darkness", "Eye discomfort or headaches"]
    for s in symptoms: story.append(Paragraph(f"- {s}", clinical_blue))
    story.append(Spacer(1,8))
    story.append(Paragraph("<b>Major Risk Factors</b>", clinical_blue))
    risks = ["Age >60", "Family history", "High IOP", "Thin corneas", "High myopia", "Diabetes / Hypertension"]
    for r in risks: story.append(Paragraph(f"- {r}", clinical_blue))
    story.append(PageBreak())

    # PAGE 5: Recommendations & Lifestyle
    story.append(Paragraph("RECOMMENDATIONS & ACTION PLAN", header_green)); story.append(Spacer(1,8))
    if rnflt_cluster == "Glaucoma-like":
        recs = ["Schedule ophthalmologist visit within 1–2 weeks.", "Request tonometry, visual fields, gonioscopy and dilated optic nerve exam.", "Bring this report to your appointment."]
    else:
        recs = ["Continue routine annual eye exams.", "Document this baseline for future comparison.", "Consider earlier follow-up if any symptoms occur."]
    for r in recs: story.append(Paragraph(f"- {r}", clinical_blue))
    story.append(Spacer(1,8))
    story.append(Paragraph("<b>Lifestyle Recommendations</b>", clinical_blue))
    lifestyle = ["Antioxidant-rich diet", "Regular aerobic activity", "Protect from UV exposure", "Limit caffeine, avoid smoking", "Sleep 7–9 hours"]
    for l in lifestyle: story.append(Paragraph(f"- {l}", clinical_blue))
    story.append(PageBreak())

    # PAGE 6: Disclaimers & References
    story.append(Paragraph("IMPORTANT MEDICAL DISCLAIMER", header_green)); story.append(Spacer(1,8))
    story.append(Paragraph("This OCULAIRE report is generated by an AI screening tool for research and educational purposes only. It is not a clinical diagnosis. Always consult an ophthalmologist for medical decisions.", clinical_blue))
    story.append(Spacer(1,8))
    story.append(Paragraph("<b>Methodology & References</b>", header_green))
    refs = ["Weinreb RN et al., JAMA 2014", "Tham YC et al., Ophthalmology 2014", "European Glaucoma Society Guidelines 2021"]
    for ref in refs: story.append(Paragraph(f"- {ref}", body_small_blue))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ============================================================
# If user pressed PDF trigger, create & offer download
# ============================================================
if st.session_state.get("trigger_pdf", False):
    stored_figs = st.session_state.get("pdf_figs", None)
    stored_ctx = st.session_state.get("pdf_context", {})
    if stored_figs:
        try:
            final_pdf = generate_full_pdf(
                figs=stored_figs,
                rnflt_metrics=stored_ctx.get("rnflt_metrics"),
                rnflt_cluster=stored_ctx.get("rnflt_cluster"),
                rnflt_severity=stored_ctx.get("rnflt_severity"),
                bscan_label=stored_ctx.get("bscan_label"),
                bscan_conf=stored_ctx.get("bscan_conf"),
                quality=stored_ctx.get("quality")
            )
            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader("📄 Download Full OCULAIRE Medical Report")
            st.download_button(label="📄 Download Full PDF Report (6 Pages)", data=final_pdf, file_name="OCULAIRE_Full_Report.pdf", mime="application/pdf", use_container_width=True)
        except Exception as e:
            st.error(f"PDF generation error: {e}")
    else:
        st.warning("No visuals found to create a full report. Upload RNFLT or B-scan and generate again.")
    st.session_state["trigger_pdf"] = False

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:#b8ffb8;padding:6px;'>OCULAIRE Neon Lab v5 — For research use only</div>", unsafe_allow_html=True)
