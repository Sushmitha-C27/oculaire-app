# app.py — OCULAIRE Neon Lab v6 (single-file)
# Run: streamlit run app.py

import os
import io
import time
import json
import requests
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import cv2

# Try to import google.generativeai (optional). If present we use SDK style; otherwise use REST.
try:
    import google.generativeai as genai
    USE_SDK = True
except Exception:
    USE_SDK = False

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="OCULAIRE: Neon Glaucoma Detection Dashboard",
                   layout="wide",
                   page_icon="👁️",
                   initial_sidebar_state="expanded")

# -----------------------
# Session-state init
# -----------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_raw_reply" not in st.session_state:
    st.session_state.last_raw_reply = None
if "ui_tick" not in st.session_state:
    st.session_state.ui_tick = 0

# -----------------------
# Helpers - API key
# -----------------------
def get_api_key():
    # priority: streamlit secrets -> environment -> None
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY")

API_KEY = get_api_key()

# -----------------------
# Matplotlib neon defaults
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
# CSS / Theme (neon + floating expander)
# -----------------------
st.markdown(
    """
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

/* Header styles */
.header { text-align:center; margin-top:10px; margin-bottom:10px; }
.header h1 {
  font-size:42px; font-weight:900; letter-spacing:3px;
  background: linear-gradient(90deg, var(--neonA), var(--neonB));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  text-shadow: 0 0 20px rgba(0,245,255,0.8), 0 0 35px rgba(255,64,196,0.5);
}
.header h3 { color:var(--muted); font-weight:400; font-size:15px; }

/* Card */
.card {
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
  border:1px solid rgba(255,255,255,0.05);
  box-shadow: 0 0 25px rgba(0,245,255,0.05), 0 0 35px rgba(255,64,196,0.05);
  border-radius:12px; padding:16px; margin-bottom:16px;
}

/* Severity bar: longer and faster beat */
.sev-wrap { margin-top:16px; }
.sev-outer { height:18px; width:92%; margin: 0 auto; background: rgba(255,255,255,0.03); border-radius:14px; overflow:hidden; }
.sev-inner {
  height:100%; width:0%;
  background: linear-gradient(90deg,var(--neonA),var(--neonB));
  border-radius:14px;
  box-shadow: 0 0 25px rgba(0,245,255,0.6), 0 0 25px rgba(255,64,196,0.5);
  transition: width 0.6s ease-in-out;
}
.sev-chip {
  margin-top:6px; display:inline-block;
  padding:8px 14px; border-radius:16px;
  font-weight:800; font-size:14px; color:#021617;
  background: linear-gradient(90deg, rgba(0,245,255,0.95), rgba(255,64,196,0.95));
  box-shadow: 0 0 30px rgba(0,245,255,0.25), 0 0 40px rgba(255,64,196,0.2);
  animation: pulse 1s infinite;
}
@keyframes pulse { 0%{transform:scale(1);} 50%{transform:scale(1.06);} 100%{transform:scale(1);} }

/* Floating expander neon style */
.floating-expander {
  position: fixed !important;
  bottom: 18px !important;
  right: 18px !important;
  width: 460px !important;
  max-width: 92vw !important;
  z-index: 9999 !important;
  animation: float 3s ease-in-out infinite !important;
}
.floating-expander details {
  border-radius: 14px !important;
  border: 2px solid rgba(0,245,255,0.18) !important;
  background: linear-gradient(180deg, rgba(10,15,37,0.98), rgba(2,2,8,0.98)) !important;
  box-shadow: 0 10px 40px rgba(0,0,0,0.6) !important;
}
.floating-expander details[open] { box-shadow: 0 0 60px rgba(0,245,255,0.6), 0 0 90px rgba(255,64,196,0.4) !important; }

/* summary style and neon icon */
.floating-expander details summary {
  background: linear-gradient(135deg, rgba(0,245,255,0.2), rgba(255,64,196,0.2)) !important;
  padding: 14px !important;
  border-radius: 12px !important;
  cursor: pointer !important;
  font-weight: 800 !important;
  font-size: 16px !important;
  color: #e6faff !important;
  display: flex !important;
  align-items: center !important;
  gap: 10px !important;
}
.floating-expander details summary::before { content: "💬"; font-size:22px; margin-right:6px; animation: pulse 1.5s infinite; }

/* chat inside */
.chat-msg-user { padding:10px; border-radius:8px; background: rgba(0,245,255,0.04); margin-bottom:8px; }
.chat-msg-assistant { padding:10px; border-radius:8px; background: rgba(255,64,196,0.04); margin-bottom:8px; }

/* hide default footer */
footer { visibility:hidden; }
@keyframes float { 0%,100%{ transform: translateY(0px);} 50%{ transform: translateY(-6px);} }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Chat assistant implementation (Gemini)
# -----------------------
MODEL_NAME = "models/gemini-2.5-pro"  # change if needed

def ask_glaucoma_assistant(question, history, api_key):
    """
    Send question + short conversation history to Gemini (SDK or REST fallback).
    Returns assistant text (string) or raises exception.
    """
    if not api_key or not api_key.strip():
        raise RuntimeError("No Gemini API key configured. Put GEMINI_API_KEY in Streamlit secrets or environment.")
    system_instruction = (
        "You are a specialist assistant for glaucoma, OCT and RNFLT. "
        "Answer only glaucoma/OCT/RNFLT related educational questions concisely (<=200 words). "
        "Include a short disclaimer to consult a clinician."
    )

    # Build a short conversation context (last 6 messages)
    ctx = []
    for msg in history[-6:]:
        role = "user" if msg["role"] == "user" else "assistant"
        ctx.append(f"{role}: {msg['content']}")

    prompt = system_instruction + "\n\n" + "\n".join(ctx) + f"\n\nUser: {question}\n\nAssistant:"

    # SDK path (if available)
    if USE_SDK:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(MODEL_NAME)
            chat_history = []
            for msg in history[-6:]:
                role = "user" if msg["role"] == "user" else "model"
                chat_history.append({"role": role, "parts": [msg["content"]]})
            chat = model.start_chat(history=chat_history)
            resp = chat.send_message(f"{system_instruction}\n\nUser question: {question}")
            text = resp.text
            return text if isinstance(text, str) else str(text)
        except Exception as e:
            # try REST fallback if SDK fails
            # proceed to REST section below after logging
            st.sidebar.error(f"SDK call failed: {e}")

    # REST fallback
    url = "https://generativelanguage.googleapis.com/v1beta/{MODEL_NAME}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.6, "maxOutputTokens": 400}
    }
    resp = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=30)
    if resp.status_code == 200:
        j = resp.json()
        try:
            text = j["candidates"][0]["content"]["parts"][0]["text"]
            return text
        except Exception as e:
            raise RuntimeError(f"Malformed response JSON: {e}\n\n{json.dumps(j)[:800]}")
    elif resp.status_code == 403:
        raise RuntimeError("API Key invalid or restricted (403). Check key & restrictions.")
    else:
        raise RuntimeError(f"API Error {resp.status_code}: {resp.text[:800]}")

# -----------------------
# Load ML models/resources (cached)
# -----------------------
@st.cache_resource
def load_models_and_resources():
    b_model = None
    scaler = kmeans = avg_healthy = avg_glaucoma = thin_cluster = None
    # load bscan model (optional)
    try:
        b_model = None
        if os.path.exists("bscan_cnn.h5"):
            import tensorflow as tf
            b_model = tf.keras.models.load_model("bscan_cnn.h5", compile=False)
    except Exception:
        b_model = None
    # load other resources (optional)
    try:
        scaler = joblib.load("rnflt_scaler.joblib") if os.path.exists("rnflt_scaler.joblib") else None
        kmeans = joblib.load("rnflt_kmeans.joblib") if os.path.exists("rnflt_kmeans.joblib") else None
        avg_healthy = np.load("avg_map_healthy.npy") if os.path.exists("avg_map_healthy.npy") else None
        avg_glaucoma = np.load("avg_map_glaucoma.npy") if os.path.exists("avg_map_glaucoma.npy") else None
        thin_cluster = 0 if (avg_healthy is not None and np.nanmean(avg_healthy) > np.nanmean(avg_glaucoma)) else 1
    except Exception:
        scaler = kmeans = avg_healthy = avg_glaucoma = thin_cluster = None
    return b_model, scaler, kmeans, avg_healthy, avg_glaucoma, thin_cluster

b_model, scaler, kmeans, avg_healthy, avg_glaucoma, thin_cluster = load_models_and_resources()

# -----------------------
# Helper: RNFLT & B-scan processing helpers
# -----------------------
def process_npz_file_like(file_like):
    try:
        buf = io.BytesIO(file_like.getvalue())
        data = np.load(buf, allow_pickle=True)
        if "volume" in data:
            arr = data["volume"]
        else:
            arr = data[data.files[0]]
        # if 3D, take first slice if needed
        if arr.ndim == 3:
            # if volume: choose mean across first axis or take [0]
            arr2 = arr
            # If shape (N,H,W) and N>1 we can average or choose first. We'll choose mean of all slices to produce RNFLT map.
            arr = np.nanmean(arr2, axis=0)
        vals = arr.flatten().astype(float)
        metrics = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals)), "min": float(np.nanmin(vals)), "max": float(np.nanmax(vals))}
        return arr, metrics
    except Exception as e:
        st.error(f"NPZ read error: {e}")
        return None, None

def process_rnflt_image_file_like(file_like):
    try:
        pil = Image.open(file_like).convert("L")
        arr = np.array(pil).astype(float)
        vals = arr.flatten()
        metrics = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals)), "min": float(np.nanmin(vals)), "max": float(np.nanmax(vals))}
        return arr, metrics, pil
    except Exception as e:
        st.error(f"RNFLT image read error: {e}")
        return None, None, None

def preprocess_bscan_pil(image_pil, size=(320,320)):
    arr = np.array(image_pil.convert("L")).astype(np.float32)
    arr = np.clip(arr, 0, np.percentile(arr, 99))
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    arr_res = cv2.resize(arr, size, interpolation=cv2.INTER_LINEAR)
    arr_rgb = np.repeat(arr_res[..., None], 3, axis=-1)
    batch = np.expand_dims(arr_rgb, axis=0).astype(np.float32)
    return batch, arr_res

def gradcam_for_model(batch, model):
    try:
        import tensorflow as tf
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

# -----------------------
# Utility: PNG/PDF outputs
# -----------------------
def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.getvalue()

def create_pdf_bytes(figs):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        for f in figs:
            pdf.savefig(f, bbox_inches="tight", facecolor=f.get_facecolor())
    buf.seek(0)
    return buf.getvalue()

# -----------------------
# Sidebar: RNFLT / B-scan options and converter
# -----------------------
st.sidebar.title("RNFLT & B-scan Tools")

# RNFLT input mode
rnflt_input_mode = st.sidebar.radio("RNFLT input type", ("NPZ (recommended)", "Image (single RNFLT image)"))

st.sidebar.markdown("---")
st.sidebar.subheader("Image → NPZ converter (pack slices)")
st.sidebar.markdown("If you only have RNFLT slice images, upload them (ordered). I'll pack into `volume` and let you download `.npz`.")
conv_files = st.sidebar.file_uploader("Upload RNFLT slice images (ordered)", accept_multiple_files=True, type=["png","jpg","jpeg"], help="Upload slices in order; limit per file depends on deployment.")
if conv_files:
    if st.sidebar.button("Convert to .npz and prepare download"):
        try:
            slices = []
            for f in conv_files:
                im = Image.open(f).convert("L")
                arr = np.array(im).astype(np.float32)
                slices.append(arr)
            vol = np.stack(slices, axis=0)
            outbuf = io.BytesIO()
            np.savez_compressed(outbuf, volume=vol)
            outbuf.seek(0)
            st.sidebar.success(f"Packed {len(slices)} slices → volume shape {vol.shape}")
            st.sidebar.download_button("⬇️ Download RNFLT volume (.npz)", data=outbuf.getvalue(), file_name="rnflt_volume.npz", mime="application/octet-stream")
        except Exception as e:
            st.sidebar.error(f"Conversion error: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("B-scan options")
bscan_allow_predict = st.sidebar.checkbox("Enable B-scan prediction (if model available)", value=True)
st.sidebar.markdown("If `bscan_cnn.h5` is missing, predictions are disabled but upload preview still works.")

st.sidebar.markdown("---")
st.sidebar.markdown("🔑 API key status & debug")
st.sidebar.write("Gemini API key present:", bool(API_KEY))
st.sidebar.write("Using SDK:", USE_SDK)
st.sidebar.write("Last raw reply:", (st.session_state.last_raw_reply[:300] + "...") if st.session_state.last_raw_reply and len(st.session_state.last_raw_reply) > 300 else (st.session_state.last_raw_reply or "—"))

# -----------------------
# Header
# -----------------------
st.markdown("""
<div class="header">
  <h1>👁️ OCULAIRE</h1>
  <h3>AI-Powered Glaucoma Detection Dashboard — Neon Lab v6</h3>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# -----------------------
# Main upload UI
# -----------------------
display_width = 640
colA, colB = st.columns(2)

with colA:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🩺 RNFLT Map Analysis")
    if rnflt_input_mode == "NPZ (recommended)":
        rnflt_file = st.file_uploader("Upload RNFLT file (.npz)", type=["npz"])
        rnflt_arr = None
        rnflt_metrics = None
        if rnflt_file:
            rnflt_arr, rnflt_metrics = process_npz_file_like(rnflt_file)
    else:
        rnflt_img_file = st.file_uploader("Upload RNFLT image (single grayscale RNFLT)", type=["png","jpg","jpeg"])
        rnflt_arr = None
        rnflt_metrics = None
        rnflt_pil = None
        if rnflt_img_file:
            rnflt_arr, rnflt_metrics, rnflt_pil = process_rnflt_image_file_like(rnflt_img_file)
    st.markdown("</div>", unsafe_allow_html=True)

with colB:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("👁️ B-Scan Slice Analysis")
    bscan_file = st.file_uploader("Upload B-Scan image", type=["jpg","png","jpeg"])
    st.markdown("</div>", unsafe_allow_html=True)

threshold = st.slider("Thin-zone threshold (µm)", 5, 50, 10)

# -----------------------
# Analysis (RNFLT & B-scan)
# -----------------------
if (rnflt_arr is not None) or (bscan_file is not None):
    figs = []
    severity_overall = 0.0
    st.markdown("<hr>", unsafe_allow_html=True)

    # RNFLT block
    if rnflt_arr is not None:
        try:
            metrics = rnflt_metrics
            X = np.array([[metrics["mean"], metrics["std"], metrics["min"], metrics["max"]]])
            # cluster / label
            if scaler is not None and kmeans is not None:
                Xs = scaler.transform(X)
                cluster = int(kmeans.predict(Xs)[0])
                label_r = "Glaucoma-like" if cluster == thin_cluster else "Healthy-like"
            else:
                cluster = "?"
                label_r = "Unknown"
            # compute risk/diff
            if avg_healthy is not None:
                healthy_map = avg_healthy
                if rnflt_arr.shape != healthy_map.shape:
                    healthy_map = cv2.resize(healthy_map, (rnflt_arr.shape[1], rnflt_arr.shape[0]))
                diff = rnflt_arr - healthy_map
            else:
                diff = rnflt_arr - np.nanmean(rnflt_arr)
            risk = np.where(diff < -threshold, diff, np.nan)
            total = np.isfinite(diff).sum()
            risky = np.isfinite(risk).sum()
            sev = (risky / total) * 100 if total else 0.0
            severity_overall = max(severity_overall, float(sev))

            # metrics display
            m1, m2, m3, m4 = st.columns([2,2,2,2])
            m1.markdown(f"<div style='color:var(--muted); font-size:12px;'>Status</div><div style='font-weight:900; font-size:22px; color:#fff; text-shadow:0 0 12px rgba(0,245,255,0.6);'>{'🚨' if 'Glaucoma' in label_r else '✅'} {label_r}</div>", unsafe_allow_html=True)
            m2.markdown(f"<div style='color:var(--muted); font-size:12px;'>Mean RNFLT</div><div style='font-weight:800; font-size:22px; color:#fff;'>{metrics['mean']:.2f}</div>", unsafe_allow_html=True)
            m3.markdown(f"<div style='color:var(--muted); font-size:12px;'>Std Dev</div><div style='font-weight:800; font-size:22px; color:#fff;'>{metrics['std']:.2f}</div>", unsafe_allow_html=True)
            m4.markdown(f"<div style='color:var(--muted); font-size:12px;'>Cluster</div><div style='font-weight:800; font-size:22px; color:#fff;'>{cluster}</div>", unsafe_allow_html=True)

            # preview RNFLT image
            try:
                rnflt_normalized = (rnflt_arr - np.nanmin(rnflt_arr)) / (np.nanmax(rnflt_arr) - np.nanmin(rnflt_arr) + 1e-9)
                rnflt_show = Image.fromarray(np.uint8(255 * rnflt_normalized))
                st.image(rnflt_show, caption="RNFLT map (preview)", width=display_width)
            except Exception:
                pass

            # severity UI
            def render_sev_html(pct):
                pct = max(0.0, min(100.0, float(pct)))
                return f"""
                <div class='sev-wrap'>
                  <div class='sev-outer'><div id='sev_inner' class='sev-inner'></div></div>
                  <div style='text-align:center'><div class='sev-chip'>{pct:.1f}%</div></div>
                </div>
                <script>
                  (function(){{
                    setTimeout(function(){{
                      var el=document.getElementById('sev_inner');
                      if(el) el.style.width='{pct:.1f}%';
                    }},80);
                  }})();
                </script>
                """
            st.markdown(render_sev_html(severity_overall), unsafe_allow_html=True)

            # debug plots
            fig, axes = plt.subplots(1,3,figsize=(18,5), constrained_layout=True)
            axes[0].imshow(rnflt_arr, cmap='turbo'); axes[0].set_title("Uploaded RNFLT"); axes[0].axis('off')
            axes[1].imshow(diff, cmap='bwr', vmin=-30, vmax=30); axes[1].set_title("Difference (vs Healthy)"); axes[1].axis('off')
            axes[2].imshow(risk, cmap='hot'); axes[2].set_title("Risk Map"); axes[2].axis('off')
            for ax in axes:
                ax.set_facecolor('#050612')
            fig.patch.set_facecolor("#050612")
            st.pyplot(fig)
            figs.append(fig)
        except Exception as e:
            st.error(f"Error in RNFLT processing: {e}")

    # B-scan block
    if bscan_file is not None:
        try:
            image_pil = Image.open(bscan_file).convert("L")
            st.markdown("<hr>", unsafe_allow_html=True)
            if b_model is not None and bscan_allow_predict:
                batch, proc = preprocess_bscan_pil(image_pil, size=(320,320))
                pred_raw = float(b_model.predict(batch, verbose=0)[0][0])
                label_b = "Glaucoma-like" if pred_raw > 0.5 else "Healthy-like"
                conf = pred_raw*100 if label_b == "Glaucoma-like" else (1 - pred_raw) * 100
                severity_overall = max(severity_overall, conf)

                m1, m2 = st.columns(2)
                m1.markdown(f"<div style='color:var(--muted); font-size:12px;'>CNN Prediction</div><div style='font-weight:900; font-size:22px; color:#fff;'>{'🚨' if 'Glaucoma' in label_b else '✅'} {label_b}</div>", unsafe_allow_html=True)
                m2.markdown(f"<div style='color:var(--muted); font-size:12px;'>Confidence</div><div style='font-weight:900; font-size:22px; color:#fff;'>{conf:.2f}%</div>", unsafe_allow_html=True)
                st.markdown(render_sev_html(conf), unsafe_allow_html=True)

                heat = gradcam_for_model(batch, b_model)
                if heat is not None:
                    hm_small = (cv2.resize(heat, (proc.shape[1], proc.shape[0])) * 255).astype(np.uint8)
                    hm_color = cv2.applyColorMap(hm_small, cv2.COLORMAP_JET)
                    overlay_small = (np.stack([proc]*3, axis=-1) * 255).astype(np.uint8)
                    overlay_small = cv2.addWeighted(overlay_small, 0.6, hm_color, 0.4, 0)
                    orig_w, orig_h = image_pil.size
                    overlay_up = cv2.resize(overlay_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                    overlay_pil = Image.fromarray(overlay_up)
                    st.image([image_pil.resize((display_width, int(display_width * orig_h / orig_w))), overlay_pil.resize((display_width, int(display_width * orig_h / orig_w)))],
                             caption=["Original B-Scan (preview)", "Grad-CAM Overlay (preview)"], width=display_width)
                else:
                    st.image(image_pil, caption="Original B-Scan (preview)", width=display_width)
            else:
                st.info("B-scan model unavailable or prediction not enabled. Showing preview only.")
                st.image(image_pil, caption="B-scan preview", width=display_width)
        except Exception as e:
            st.error(f"B-scan error: {e}")

    # Combined severity summary
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center'>Overall Severity Index</h3>", unsafe_allow_html=True)
    st.markdown(render_sev_html(severity_overall), unsafe_allow_html=True)

    # download options for plots
    if figs:
        pngb = fig_to_png_bytes(figs[0])
        pdfb = create_pdf_bytes(figs)
        st.markdown("<div style='text-align:center; margin-top:8px'>", unsafe_allow_html=True)
        st.download_button("📸 Download RNFLT PNG", data=pngb, file_name="oculaire_rnflt.png", mime="image/png")
        st.download_button("📄 Download Full Report (PDF)", data=pdfb, file_name="oculaire_report.pdf", mime="application/pdf")
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:var(--muted);padding:6px;'>OCULAIRE Neon Lab v6 — For research use only</div>", unsafe_allow_html=True)

# -----------------------
# Floating Streamlit expander chat (no JS navigation)
# -----------------------
st.markdown('<div class="floating-expander">', unsafe_allow_html=True)
with st.expander("💬 Ask AI assistant", expanded=False):
    st.markdown("<div style='font-weight:900; font-size:18px; margin-bottom:6px;'>🤖 OCULAIRE Assistant</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:var(--muted); margin-bottom:10px;'>Ask about glaucoma, OCT, RNFLT or interpretation of analysis.</div>", unsafe_allow_html=True)

    # show chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"<div class='chat-msg-user'><strong>You:</strong> {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-msg-assistant'><strong>OCULAIRE:</strong> {msg['content']}</div>", unsafe_allow_html=True)

    # input area
    user_question = st.text_input("Your question:", key="chat_input", placeholder="e.g., What is glaucoma? How does OCT detect it?", label_visibility="collapsed")
    col1, col2 = st.columns([4,1])
    with col1:
        send_btn = st.button("📤 Send", use_container_width=True)
    with col2:
        clear_btn = st.button("🗑️ Clear chat", use_container_width=True)

    # Robust send handler - no experimental_rerun
    if send_btn:
        if not user_question or user_question.strip() == "":
            st.warning("Please type a question first.")
        else:
            # Append user message immediately
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            # Temporary assistant placeholder so user sees immediate feedback
            st.session_state.chat_history.append({"role": "assistant", "content": "⏳ Thinking... contacting Gemini..."})
            # Ensure UI updates via state tick (Streamlit will rerun after this button click completes)
            st.session_state.ui_tick += 1

            # Do the remote call synchronously and replace the placeholder with the real reply
            try:
                reply_text = ask_glaucoma_assistant(user_question, st.session_state.chat_history, API_KEY)
            except Exception as e:
                reply_text = f"❌ Assistant error: {str(e)}"
            # store raw reply for sidebar/debug
            st.session_state.last_raw_reply = reply_text

            # remove the previous "Thinking..." assistant placeholder (search from end)
            for i in range(len(st.session_state.chat_history)-1, -1, -1):
                if st.session_state.chat_history[i]["role"] == "assistant" and "Thinking..." in st.session_state.chat_history[i]["content"]:
                    st.session_state.chat_history.pop(i)
                    break
            # append actual reply
            st.session_state.chat_history.append({"role": "assistant", "content": reply_text})
            # increment tick to force UI to reflect new messages (Streamlit reruns automatically after button event)
            st.session_state.ui_tick += 1

    if clear_btn:
        st.session_state.chat_history = []
        st.session_state.ui_tick += 1

st.markdown('</div>', unsafe_allow_html=True)
