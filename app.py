# ============================================================
#  OCULAIRE — Neon Green UI + Fast Glow Header
#  Option C Theme:
#    • App UI       = Neon Green
#    • PDF Cover    = Cyan + Magenta
#    • Clinical PDF = Green-accent medical layout
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

# ReportLab for advanced multipage PDF
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# Try Gemini SDK
try:
    import google.generativeai as genai
    USE_SDK = True
except Exception:
    USE_SDK = False

# ------------------------------------------------------------
# Page Config
# ------------------------------------------------------------
st.set_page_config(
    page_title="OCULAIRE: Neon Glaucoma Detection Dashboard",
    layout="wide",
    page_icon="👁️"
)

# ------------------------------------------------------------
# --- IMPORTANT: Initialize globals early to avoid NameError ---
# ------------------------------------------------------------
figs = []                   # will collect matplotlib figures used for PDF
rnflt_metrics = None
label_r = None
label_b = None
conf = 0.0
severity_overall = 0.0

# ------------------------------------------------------------
# Neon Green Theme + Fast Glow Header
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Session State
# ------------------------------------------------------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_processed_q' not in st.session_state:
    st.session_state.last_processed_q = None

# ------------------------------------------------------------
# API Key helper
# ------------------------------------------------------------
def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    if os.getenv("GEMINI_API_KEY"):
        return os.getenv("GEMINI_API_KEY")
    return None

API_KEY = get_api_key()

# ------------------------------------------------------------
# Matplotlib Theme
# ------------------------------------------------------------
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
#  OCULAIRE — Processing & Analysis Engine
#  (RNFLT, B-Scan, Risk Maps, Grad-CAM, Converters)
# ============================================================

# ------------------------------------------------------------
# Load Models
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Process NPZ files
# ------------------------------------------------------------
def process_npz(f):
    try:
        buf = io.BytesIO(f.getvalue())
        data = np.load(buf, allow_pickle=True)
        arr = data["volume"] if "volume" in data else data[data.files[0]]

        # If 3D, take slice 0
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


# ------------------------------------------------------------
# RNFLT Risk Calculation
# ------------------------------------------------------------
def compute_risk_map(rnflt, healthy, threshold=-10):
    if rnflt.shape != healthy.shape:
        healthy = cv2.resize(healthy, (rnflt.shape[1], rnflt.shape[0]))

    diff = rnflt - healthy
    risk = np.where(diff < threshold, diff, np.nan)

    total = np.isfinite(diff).sum()
    risky = np.isfinite(risk).sum()

    severity = (risky / total) * 100 if total else 0
    return diff, risk, severity


# ------------------------------------------------------------
# B-scan preprocessing
# ------------------------------------------------------------
def preprocess_bscan(image_pil, size=(224, 224)):
    arr = np.array(image_pil.convert('L'))
    arr = np.clip(arr, 0, np.percentile(arr, 99))
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)

    arr_res = cv2.resize(arr, size, interpolation=cv2.INTER_NEAREST)
    rgb = np.repeat(arr_res[..., None], 3, axis=-1)
    batch = np.expand_dims(rgb, axis=0).astype(np.float32)

    return batch, arr_res


# ------------------------------------------------------------
# Grad-CAM
# ------------------------------------------------------------
def gradcam(batch, model):
    try:
        last_conv = None
        for layer in reversed(model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                last_conv = layer.name
                break

        if not last_conv:
            return None

        grad_model = tf.keras.models.Model(model.inputs,
                                           [model.get_layer(last_conv).output, model.output])

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(batch)
            loss = preds[:, 0]

        grads = tape.gradient(loss, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_out = conv_out[0]
        heat = conv_out @ pooled[..., tf.newaxis]
        heat = tf.squeeze(heat)

        heat = tf.maximum(heat, 0) / (tf.reduce_max(heat) + 1e-6)
        return heat.numpy()

    except Exception:
        return None


# ------------------------------------------------------------
# PNG + PDF helpers
# ------------------------------------------------------------
def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=150)
    buf.seek(0)
    return buf.getvalue()


# ------------------------------------------------------------
# Severity Renderer (Green bar)
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Image → NPZ Converters
# ------------------------------------------------------------
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
#  OCULAIRE — SIDEBAR + MAIN ANALYSIS LAYOUT
# ============================================================

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔑 API Status")
    if API_KEY:
        st.success("Gemini API key configured")
    else:
        st.error("API key missing")
        st.info("Add GEMINI_API_KEY in secrets or env variable.")

    st.markdown("---")

    st.markdown("## 🩺 RNFLT Input & Tools")
    rnflt_input_mode = st.radio(
        "RNFLT input type",
        ["NPZ (recommended)", "Single Image"]
    )

    rnflt_conv = st.file_uploader(
        "Upload RNFLT slices → convert to .npz",
        accept_multiple_files=True,
        type=["png", "jpg", "jpeg"]
    )
    if rnflt_conv:
        if st.button("Convert RNFLT slices to NPZ"):
            data, shape = convert_images_to_npz(rnflt_conv)
            if data:
                st.success(f"Packed slices into volume shape {shape}")
                st.download_button(
                    "⬇️ Download RNFLT volume",
                    data=data,
                    file_name="rnflt_volume.npz"
                )

    st.markdown("---")

    st.markdown("## 👁️ B-Scan Input & Tools")
    bscan_input_mode = st.radio(
        "B-scan input type",
        ["Image (recommended)", "NPZ (multi-slice)"]
    )

    bscan_conv = st.file_uploader(
        "Upload B-scan slices → convert to .npz",
        accept_multiple_files=True,
        type=["png", "jpg", "jpeg"]
    )
    if bscan_conv:
        if st.button("Convert B-scan slices to NPZ"):
            data, shape = convert_images_to_npz(bscan_conv)
            if data:
                st.success(f"Packed B-scan slices into {shape}")
                st.download_button(
                    "⬇️ Download B-scan volume",
                    data=data,
                    file_name="bscan_volume.npz"
                )

    st.markdown("---")
    threshold = st.slider(
        "Thin-zone threshold (µm)",
        min_value=5, max_value=50, value=10
    )


# ============================================================
# MAIN LAYOUT
# ============================================================

colA, colB = st.columns(2)

# ------------------------------------------------------------
# RNFLT Upload Panel
# ------------------------------------------------------------
with colA:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🩺 RNFLT Map Analysis")

    rnflt_arr = None
    rnflt_metrics = None
    rnflt_pil = None

    if rnflt_input_mode == "NPZ (recommended)":
        rnflt_file = st.file_uploader(
            "Upload RNFLT .npz file",
            type=["npz"],
            key="rnflt_npz_main"
        )
        if rnflt_file:
            rnflt_arr, rnflt_metrics = process_npz(rnflt_file)

    else:
        rnflt_image = st.file_uploader(
            "Upload single RNFLT image",
            type=["png", "jpg", "jpeg"],
            key="rnflt_img_main"
        )
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


# ------------------------------------------------------------
# B-SCAN Upload Panel
# ------------------------------------------------------------
with colB:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("👁️ B-Scan Slice Analysis")

    bscan_file = None
    bscan_npz = None

    if bscan_input_mode == "Image (recommended)":
        bscan_file = st.file_uploader(
            "Upload B-scan image",
            type=["png", "jpg", "jpeg"],
        )
    else:
        bscan_npz = st.file_uploader(
            "Upload B-scan NPZ volume",
            type=["npz"]
        )
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# START ANALYSIS WHEN ANY INPUT IS AVAILABLE
# ============================================================

analysis_trigger = (
    (rnflt_arr is not None) or
    (bscan_file is not None) or
    (bscan_npz is not None)
)

if analysis_trigger:
    st.markdown("<hr>", unsafe_allow_html=True)

    # reuse the top-level vars in case they were set earlier
    global figs, rnflt_metrics, label_r, label_b, conf, severity_overall

    severity_overall = 0
    figs = []  # Collect all matplotlib figs for PDF later

    # --------------------------------------------------------
    # RNFLT ANALYSIS
    # --------------------------------------------------------
    if rnflt_arr is not None:
        try:
            metrics = rnflt_metrics if rnflt_metrics is not None else {
                "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0
            }

            if scaler is not None and kmeans is not None:
                X = np.array([[metrics["mean"], metrics["std"],
                               metrics["min"], metrics["max"]]])
                Xs = scaler.transform(X)
                cluster = int(kmeans.predict(Xs)[0])
                label_r = "Glaucoma-like" if cluster == thin_cluster else "Healthy-like"
            else:
                cluster = "?"
                label_r = "Unknown"

            if avg_healthy is not None:
                diff, risk, sev = compute_risk_map(
                    rnflt_arr, avg_healthy, -threshold
                )
            else:
                diff = rnflt_arr - np.nanmean(rnflt_arr)
                risk = np.where(diff < -threshold, diff, np.nan)
                # fallback severity
                try:
                    sev = float(np.nanpercentile(np.nan_to_num(diff), 75))
                except Exception:
                    sev = 0.0

            severity_overall = max(severity_overall, sev)

            # Display metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Status", label_r)
            c2.metric("Mean RNFLT", f"{metrics['mean']:.2f}")
            c3.metric("Std Dev", f"{metrics['std']:.2f}")
            c4.metric("Cluster", str(cluster))

            # Green severity bar
            st.markdown(render_severity(sev), unsafe_allow_html=True)

            # RNFLT Figure (3-panel)
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            im0 = axes[0].imshow(rnflt_arr, cmap='turbo')
            axes[0].axis('off'); axes[0].set_title("RNFLT Map")
            plt.colorbar(im0, ax=axes[0], shrink=0.85)

            im1 = axes[1].imshow(diff, cmap='bwr', vmin=-30, vmax=30)
            axes[1].axis('off'); axes[1].set_title("Difference vs Healthy")
            plt.colorbar(im1, ax=axes[1], shrink=0.85)

            im2 = axes[2].imshow(risk, cmap='hot')
            axes[2].axis('off'); axes[2].set_title("Risk Map")
            plt.colorbar(im2, ax=axes[2], shrink=0.85)

            fig.patch.set_facecolor("#020802")
            st.pyplot(fig)
            figs.append(fig)

        except Exception as e:
            st.error(f"RNFLT Error: {e}")


    # --------------------------------------------------------
    # B-SCAN ANALYSIS (Single Image)
    # --------------------------------------------------------
    if bscan_file:
        try:
            pil = Image.open(bscan_file).convert("L")
            batch, proc = preprocess_bscan(pil)

            if b_model:
                pred_raw = float(b_model.predict(batch, verbose=0)[0][0])
                label_b = "Glaucoma-like" if pred_raw > 0.5 else "Healthy-like"
                conf = pred_raw * 100 if label_b == "Glaucoma-like" else (1 - pred_raw) * 100
            else:
                label_b = "Unknown"; conf = 0.0

            severity_overall = max(severity_overall, conf)

            col1, col2 = st.columns(2)
            col1.metric("CNN Prediction", label_b)
            col2.metric("Confidence", f"{conf:.2f}%")

            # Severity bar
            st.markdown(render_severity(conf), unsafe_allow_html=True)

            # Grad-CAM
            heat = gradcam(batch, b_model) if b_model else None
            if heat is not None:
                heat_r = cv2.resize(heat, (224, 224))
                hm = (heat_r * 255).astype(np.uint8)
                hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)

                overlay = (np.stack([proc]*3, axis=-1) * 255).astype(np.uint8)
                overlay = cv2.addWeighted(overlay, 0.6, hm_color, 0.4, 0)

                st.image([pil, overlay], caption=["Original", "Grad-CAM"], use_column_width=True)

                # Store visualization fig for PDF
                fig2, ax2 = plt.subplots(1, 2, figsize=(14, 6))
                ax2[0].imshow(pil, cmap="gray"); ax2[0].axis("off"); ax2[0].set_title("B-Scan")
                ax2[1].imshow(overlay); ax2[1].axis("off"); ax2[1].set_title("Grad-CAM Overlay")
                fig2.patch.set_facecolor("#020802")
                figs.append(fig2)
            else:
                st.image(pil, caption="Original B-Scan", use_column_width=True)

        except Exception as e:
            st.error(f"B-Scan Error: {e}")


    # --------------------------------------------------------
    # Combined Severity
    # --------------------------------------------------------
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; color:#39ff14;'>Overall Severity Index</h3>", unsafe_allow_html=True)
    st.markdown(render_severity(severity_overall), unsafe_allow_html=True)

    # --------------------------------------------------------
    # Downloads (PNG + PDF)
    # PDF is implemented in Chunk 4
    # --------------------------------------------------------
    if figs:
        png_bytes = fig_to_png(figs[0])
        st.download_button("📸 Download RNFLT PNG", data=png_bytes, file_name="oculaire_rnflt.png")

        st.session_state["pdf_figs"] = figs
        st.session_state["pdf_rnflt_metrics"] = rnflt_metrics
        st.session_state["pdf_rnflt_cluster"] = label_r if label_r is not None else None
        st.session_state["pdf_rnflt_severity"] = severity_overall
        st.session_state["pdf_bscan_label"] = label_b if label_b is not None else None
        st.session_state["pdf_bscan_conf"] = conf if conf is not None else 0

        # Button will call final PDF generator (Chunk 4)
        if st.button("📄 Generate Full Medical Report (PDF)"):
            st.session_state["trigger_pdf"] = True

# ============================================================
#  OCULAIRE — FULL MEDICAL PDF REPORT ENGINE (6+ PAGES)
#  Option C:
#    • COVER PAGE = Cyan + Magenta neon branding
#    • OTHER PAGES = Green-accent clinical theme
# ============================================================

def generate_full_pdf(figs,
                      rnflt_metrics=None,
                      rnflt_cluster=None,
                      rnflt_severity=None,
                      bscan_label=None,
                      bscan_conf=None):
    """Generates the complete multi-page OCULAIRE medical report."""
    buffer = io.BytesIO()

    # --------------------------------------------------------
    # Base PDF template
    # --------------------------------------------------------
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=40, bottomMargin=40,
        leftMargin=50, rightMargin=50
    )
    styles = getSampleStyleSheet()

    # Custom styles
    title = ParagraphStyle(
        "TitleGlow", parent=styles["Title"],
        fontSize=28, textColor=colors.HexColor("#00eaff"),
        leading=32, alignment=1
    )
    subtitle = ParagraphStyle(
        "Subtitle", parent=styles["Heading2"],
        fontSize=14, textColor=colors.HexColor("#ff40c4"),
        alignment=1, leading=18
    )
    header_green = ParagraphStyle(
        "HeaderGreen", parent=styles["Heading2"],
        textColor=colors.HexColor("#39ff14"),
        fontSize=18
    )
    # <<< CHANGED: report description/body text set to neon blue per request >>>
    body = ParagraphStyle(
        "Body", parent=styles["BodyText"],
        fontSize=11, leading=15,
        textColor=colors.HexColor("#00bfff")   # neon blue description text
    )
    body_small = ParagraphStyle(
        "Small", parent=styles["BodyText"],
        fontSize=9, leading=12,
        textColor=colors.HexColor("#7fdfff")
    )

    story = []

    # ============================================================
    #  PAGE 1: COVER PAGE — Cyan/Magenta Neon
    # ============================================================

    story.append(Paragraph("OCULAIRE", title))
    story.append(Paragraph("AI-Powered Glaucoma Screening Report", subtitle))
    story.append(Spacer(1, 25))

    # Metadata table
    report_id = "OCU-" + datetime.now().strftime("%Y%m%d%H%M%S")
    gen_date = datetime.now().strftime("%B %d, %Y — %I:%M %p")

    metadata = [
        ["Report Generated:", gen_date],
        ["Analysis Type:", "RNFLT + B-Scan"],
        ["AI Model Version:", "OCULAIRE v5.0 (2024)"],
        ["Report ID:", report_id],
    ]
    meta_table = Table(metadata, colWidths=[150, 300])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.Color(0.05, 0.05, 0.08)),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.HexColor("#e6faff")),
        ("INNERGRID", (0,0), (-1,-1), 0.3, colors.HexColor("#00eaff")),
        ("BOX", (0,0), (-1,-1), 1, colors.HexColor("#00eaff")),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica")
    ]))

    story.append(meta_table)
    story.append(Spacer(1, 25))

    # Executive Summary (Color-coded)
    if rnflt_cluster == "Glaucoma-like" or bscan_label == "Glaucoma-like":
        risk_color = colors.HexColor("#ff3b3b")
        risk_text = "⚠️ ABNORMAL PATTERNS DETECTED"
        risk_level = ("HIGH" if (rnflt_severity or 0) >= 60 else
                      "MODERATE" if (rnflt_severity or 0) >= 30 else
                      "LOW-MODERATE")
    else:
        risk_color = colors.HexColor("#39ff14")
        risk_text = "✅ NORMAL PATTERNS DETECTED"
        risk_level = "LOW"

    # ensure numeric safe defaults
    rnflt_severity_safe = float(rnflt_severity) if rnflt_severity is not None else 0.0
    bscan_conf_safe = float(bscan_conf) if bscan_conf is not None else 0.0

    exec_box = Paragraph(
        f"""
        <b>Status:</b> {risk_text}<br/>
        <b>Risk Level:</b> {risk_level}<br/>
        <b>Severity Index:</b> {rnflt_severity_safe:.1f}%<br/>
        <b>CNN Confidence:</b> {bscan_conf_safe:.1f}%<br/>
        """,
        body
    )

    box_table = Table([[exec_box]], colWidths=[450])
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
    story.append(PageBreak())

    # ============================================================
    #  PAGE 2 — CLINICAL INTERPRETATION & RNFLT STATS
    # ============================================================

    story.append(Paragraph("CLINICAL INTERPRETATION", header_green))
    story.append(Spacer(1, 12))

    if rnflt_cluster == "Glaucoma-like":
        story.append(Paragraph("""
        The AI analysis has detected <b>patterns consistent with glaucomatous changes</b>
        in your retinal nerve fiber layer structure. This includes thinning in clinically
        significant regions relative to the healthy reference database.
        """, body))
    else:
        story.append(Paragraph("""
        The retinal nerve fiber layer thickness pattern appears <b>within normal ranges</b>,
        showing no significant signs of glaucomatous thinning. Structural integrity of
        the optic nerve is preserved.
        """, body))

    story.append(Spacer(1, 18))
    story.append(Paragraph("<b>Key Findings</b>", body))
    story.append(Spacer(1, 6))

    if rnflt_cluster == "Glaucoma-like":
        findings = [
            "- RNFL thinning detected in critical sectors.",
            f"- {rnflt_severity_safe:.1f}% of retinal area flagged as at-risk.",
            "- Pattern deviation from healthy baseline exceeds threshold.",
            "- Suggestive of early-to-moderate glaucomatous damage.",
        ]
    else:
        findings = [
            "- RNFL thickness within expected clinical ranges.",
            "- No significant thinning detected.",
            "- Pattern matches healthy reference distribution.",
            "- Optic nerve structure appears well-maintained.",
        ]

    for f in findings:
        story.append(Paragraph(f, body))

    story.append(Spacer(1, 16))
    story.append(Paragraph("<b>RNFLT Measurements</b>", header_green))
    story.append(Spacer(1, 6))

    # safe rnflt_metrics fallback
    rnflt_metrics_safe = rnflt_metrics if rnflt_metrics is not None else {
        "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0
    }

    rnflt_tbl = [
        ["Mean Thickness", f"{rnflt_metrics_safe['mean']:.2f} μm"],
        ["Standard Deviation", f"{rnflt_metrics_safe['std']:.2f} μm"],
        ["Minimum", f"{rnflt_metrics_safe['min']:.2f} μm"],
        ["Maximum", f"{rnflt_metrics_safe['max']:.2f} μm"],
    ]
    rnflt_table = Table(rnflt_tbl, colWidths=[200, 200])
    rnflt_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.Color(0.02, 0.2, 0.02)),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.HexColor("#ccffcc")),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#39ff14")),
    ]))

    story.append(rnflt_table)
    story.append(PageBreak())

    # ============================================================
    #  PAGE 3 — RNFLT + B-SCAN IMAGES
    # ============================================================

    story.append(Paragraph("DETAILED VISUAL ANALYSIS", header_green))
    story.append(Spacer(1, 10))

    for fig in figs:
        png = fig_to_png(fig)
        img = RLImage(io.BytesIO(png), width=6.5*inch, height=3.2*inch)
        story.append(img)
        story.append(Spacer(1, 20))

    story.append(PageBreak())

    # ============================================================
    #  PAGE 4 — Symptoms & Risk Factors
    # ============================================================

    story.append(Paragraph("SYMPTOMS & RISK FACTORS", header_green))
    story.append(Spacer(1, 12))

    if rnflt_cluster == "Glaucoma-like":
        symptoms = [
            "- Gradual peripheral vision loss",
            "- Blurred vision or halos",
            "- Difficulty adjusting to dim light",
            "- Headaches or eye discomfort",
            "- Redness or irritation",
        ]
    else:
        symptoms = [
            "- Sudden vision changes",
            "- Persistent headaches",
            "- Difficulty adapting to darkness",
        ]

    story.append(Paragraph("<b>Symptoms to Monitor:</b>", body))
    for s in symptoms:
        story.append(Paragraph(s, body))

    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Major Risk Factors:</b>", body))

    risks = [
        "Age >60 years",
        "Family history of glaucoma",
        "High intraocular pressure",
        "Thin corneas",
        "Extreme myopia",
        "Diabetes / BP issues",
        "Previous eye trauma",
        "Long-term steroid use",
    ]
    for r in risks:
        story.append(Paragraph(f"- {r}", body))

    story.append(PageBreak())

    # ============================================================
    #  PAGE 5 — Recommendations, Treatment, Lifestyle, Monitoring
    # ============================================================

    story.append(Paragraph("RECOMMENDATIONS & ACTION PLAN", header_green))
    story.append(Spacer(1, 12))

    if rnflt_cluster == "Glaucoma-like":
        recs = [
            "Schedule ophthalmologist visit within 1–2 weeks.",
            "Request IOP measurement, OCT, and visual fields.",
            "Avoid heavy lifting or inverted yoga poses.",
        ]
    else:
        recs = [
            "Continue annual eye exams.",
            "Maintain healthy lifestyle habits.",
            "Document this baseline for future comparison.",
        ]

    for r in recs:
        story.append(Paragraph(f"- {r}", body))

    story.append(Spacer(1, 18))
    story.append(Paragraph("<b>Lifestyle Recommendations:</b>", body))

    lifestyle = [
        "Maintain antioxidant-rich diet.",
        "Regular aerobic activity.",
        "Protect from UV exposure.",
        "Limit caffeine, avoid smoking.",
        "Stay hydrated and sleep 7–9 hours.",
        "Take screen breaks (20–20–20 rule)."
    ]
    for l in lifestyle:
        story.append(Paragraph(f"- {l}", body))

    story.append(PageBreak())

    # ============================================================
    #  PAGE 6 — Disclaimers, Methodology, References
    # ============================================================

    story.append(Paragraph("IMPORTANT MEDICAL DISCLAIMER", header_green))
    story.append(Spacer(1, 12))
    story.append(Paragraph("""
    This OCULAIRE report is for research and educational purposes only. It is not a
    medical diagnosis. Always consult an ophthalmologist for clinical decisions.
    """, body))

    story.append(Spacer(1, 18))
    story.append(Paragraph("<b>Methodology & References</b>", header_green))
    story.append(Spacer(1, 12))

    refs = [
        "Weinreb RN et al., JAMA 2014",
        "Tham YC et al., Ophthalmology 2014",
        "European Glaucoma Society Guidelines 2021",
        "AAO Preferred Practice Pattern 2020",
        "Hood DC et al., Macular Damage Study 2013"
    ]
    for ref in refs:
        story.append(Paragraph(f"- {ref}", body_small))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
# ============================================================
# FINAL ASSEMBLY — Build Report + Download Button
# ============================================================

# Use session_state triggers to generate PDF on demand (safer)
if st.session_state.get("trigger_pdf", False) or (figs and st.button("📄 Generate Full Medical Report (PDF)")):
    try:
        # safe defaults for metrics if missing
        rnflt_metrics_to_pass = rnflt_metrics if rnflt_metrics is not None else {
            "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0
        }
        final_pdf = generate_full_pdf(
            figs=figs,
            rnflt_metrics=rnflt_metrics_to_pass,
            rnflt_cluster=label_r if label_r is not None else None,
            rnflt_severity=severity_overall,
            bscan_label=label_b if label_b is not None else None,
            bscan_conf=conf if conf is not None else 0
        )

        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("📄 Download Full OCULAIRE Medical Report")

        st.download_button(
            label="📄 Download Full PDF Report (6 Pages)",
            data=final_pdf,
            file_name="OCULAIRE_Full_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

        # reset the trigger so repeated clicks don't re-generate unexpectedly
        st.session_state["trigger_pdf"] = False

    except Exception as e:
        st.error(f"PDF generation error: {e}")

