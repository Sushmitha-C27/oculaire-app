# app.py — OCULAIRE Neon Lab v6 (full file)
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
# Session state init
# -----------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_processed_q" not in st.session_state:
    st.session_state.last_processed_q = None

# -----------------------
# Helpers: get API key
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
# Display config
# -----------------------
display_width = 640  # width for B-scan preview and overlay

# -----------------------
# Matplotlib neon style
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
# CSS — neon
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
/* Header */
.header { text-align:center; margin-top:10px; margin-bottom:10px; }
.header h1 { font-size:42px; font-weight:900; letter-spacing:3px;
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
  border-radius:12px; padding:16px;
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

/* Floating expander neon style (summary label changed later) */
.floating-expander details summary {
  background: linear-gradient(135deg, rgba(0,245,255,0.25), rgba(255,64,196,0.25)) !important;
  padding: 16px !important;
  border-radius: 14px !important;
  cursor: pointer !important;
  font-weight: 800 !important;
  font-size: 18px !important;
  color: #e6faff !important;
  display: flex !important;
  align-items: center !important;
  gap: 10px !important;
  transition: all 0.3s ease !important;
}
.floating-expander details summary::before { content: "💬"; font-size:24px; margin-right:6px; }

/* Hide default footer */
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Chat assistant function
# -----------------------
MODEL_NAME = "models/gemini-2.5-pro"

def ask_glaucoma_assistant(question, history, api_key):
    """Call Google Gemini API with glaucoma-specific context"""
    if not api_key or not api_key.strip():
        return "⚠️ Please configure your Google Gemini API key (see sidebar)."
    system_instruction = """You are a specialized medical AI assistant focused exclusively on glaucoma.
Answer only glaucoma/OCT/RNFLT related concise (<=200 words) educational info + brief disclaimer."""
    try:
        if USE_SDK:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(MODEL_NAME if MODEL_NAME else "gemini-1.5-flash")
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
                    "generationConfig": {"temperature": 0.7, "maxOutputTokens": 400}
                },
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            elif response.status_code == 403:
                return "🔑 API key invalid. Check key or restrictions."
            else:
                return f"❌ API Error ({response.status_code})"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# -----------------------
# Load models/resources (cached)
# -----------------------
@st.cache_resource
def load_models():
    b_model = None
    scaler = kmeans = avg_healthy = avg_glaucoma = thin_cluster = None
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
# Helper functions
# -----------------------
def process_npz_file(file_like):
    try:
        buf = io.BytesIO(file_like.getvalue())
        data = np.load(buf, allow_pickle=True)
        if "volume" in data:
            arr = data["volume"]
        else:
            # take first array in archive
            arr = data[data.files[0]]
        if arr.ndim == 3:
            arr = arr[0, :, :]
        vals = arr.flatten().astype(float)
        metrics = {"mean": np.nanmean(vals), "std": np.nanstd(vals), "min": np.nanmin(vals), "max": np.nanmax(vals)}
        return arr, metrics
    except Exception as e:
        st.error(f"NPZ read error: {e}")
        return None, None

def process_image_rnflt(file_like):
    try:
        image = Image.open(file_like).convert("L")  # single channel
        arr = np.array(image).astype(float)
        # normalize similar to npz processing
        vals = arr.flatten()
        metrics = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals)), "min": float(np.nanmin(vals)), "max": float(np.nanmax(vals))}
        return arr, metrics, image
    except Exception as e:
        st.error(f"RNFLT image read error: {e}")
        return None, None, None

def preprocess_bscan(image_pil, size=(320,320)):
    arr = np.array(image_pil.convert('L')).astype(np.float32)
    arr = np.clip(arr, 0, np.percentile(arr, 99))
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    arr_res = cv2.resize(arr, size, interpolation=cv2.INTER_LINEAR)
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
            # assume binary pred at index 0
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

def render_severity_html(pct):
    pct = max(0.0, min(100.0, float(pct)))
    html = f"""
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
    return html

# -----------------------
# Sidebar: RNFLT input type & converter UI
# -----------------------
st.sidebar.title("RNFLT Input & Tools")
rnflt_input_mode = st.sidebar.radio("RNFLT input type", ["NPZ (recommended)", "Image (single RNFLT image)"])

st.sidebar.markdown("---")
st.sidebar.subheader("Image → NPZ converter")
st.sidebar.markdown("If you only have RNFLT slice images, upload a sequence (PNG/JPG). I'll pack them into a `volume` and let you download `.npz`.")
conv_files = st.sidebar.file_uploader("Upload RNFLT slice images (ordered)", accept_multiple_files=True, type=["png","jpg","jpeg"])
if conv_files:
    if st.sidebar.button("Convert to .npz and download"):
        try:
            stacks = []
            for f in conv_files:
                im = Image.open(f).convert("L")
                arr = np.array(im).astype(np.float32)
                stacks.append(arr)
            vol = np.stack(stacks, axis=0)  # shape (N, H, W)
            buf = io.BytesIO()
            np.savez_compressed(buf, volume=vol)
            buf.seek(0)
            st.sidebar.success(f"Packed {len(stacks)} slices into volume with shape {vol.shape}")
            st.sidebar.download_button("⬇️ Download RNFLT volume (.npz)", data=buf.getvalue(), file_name="rnflt_volume.npz", mime="application/octet-stream")
        except Exception as e:
            st.sidebar.error(f"Conversion error: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("⚠️ Use NPZ if possible for full RNFLT maps. Images are supported but may lose metadata.")

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
# Input upload area
# -----------------------
colA, colB = st.columns(2)

with colA:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🩺 RNFLT Map Analysis")
    if rnflt_input_mode == "NPZ (recommended)":
        rnflt_file = st.file_uploader("Upload RNFLT file (.npz)", type=["npz"])
        rnflt_arr = None
        rnflt_metrics = None
        if rnflt_file:
            rnflt_arr, rnflt_metrics = process_npz_file(rnflt_file)
    else:
        rnflt_img_file = st.file_uploader("Upload RNFLT image (single grayscale RNFLT)", type=["png","jpg","jpeg"])
        rnflt_arr = None
        rnflt_metrics = None
        rnflt_pil = None
        if rnflt_img_file:
            rnflt_arr, rnflt_metrics, rnflt_pil = process_image_rnflt(rnflt_img_file)

    st.markdown("</div>", unsafe_allow_html=True)

with colB:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("👁️ B-Scan Slice Analysis (Image)")
    bscan_file = st.file_uploader("Upload B-Scan Image", type=["jpg","png","jpeg"])
    st.markdown("</div>", unsafe_allow_html=True)

threshold = st.slider("Thin-zone threshold (µm)", 5, 50, 10)

# -----------------------
# ANALYSIS pipeline
# -----------------------
if rnflt_arr is not None or bscan_file is not None:
    figs = []
    severity_overall = 0.0
    st.markdown("<hr>", unsafe_allow_html=True)

    # RNFLT processing
    if rnflt_arr is not None:
        try:
            metrics = rnflt_metrics
            X = np.array([[metrics["mean"], metrics["std"], metrics["min"], metrics["max"]]])
            label_r = "Unknown"
            if scaler is not None and kmeans is not None:
                Xs = scaler.transform(X)
                cluster = int(kmeans.predict(Xs)[0])
                label_r = "Glaucoma-like" if cluster == thin_cluster else "Healthy-like"
            else:
                cluster = "?"
                label_r = "Unknown (no clustering model)"
            # compute diff/risk using avg_healthy if available
            if avg_healthy is not None:
                # ensure shapes match
                healthy = avg_healthy
                if rnflt_arr.shape != healthy.shape:
                    healthy = cv2.resize(healthy, (rnflt_arr.shape[1], rnflt_arr.shape[0]))
                diff = rnflt_arr - healthy
                risk = np.where(diff < -threshold, diff, np.nan)
                total = np.isfinite(diff).sum()
                risky = np.isfinite(risk).sum()
                sev = (risky / total) * 100 if total else 0.0
            else:
                diff = rnflt_arr - np.nanmean(rnflt_arr)
                risk = np.where(diff < -threshold, diff, np.nan)
                sev = np.nanpercentile(np.nan_to_num(diff), 75)
            severity_overall = max(severity_overall, float(sev))
            # display metrics
            m1, m2, m3, m4 = st.columns([2,2,2,2])
            m1.markdown(f"<div style='color:var(--muted); font-size:12px;'>Status</div><div style='font-weight:800; font-size:22px; color:#fff; text-shadow:0 0 12px rgba(0,245,255,0.6);'>{'🚨' if 'Glaucoma' in label_r else '✅'} {label_r}</div>", unsafe_allow_html=True)
            m2.markdown(f"<div style='color:var(--muted); font-size:12px;'>Mean RNFLT</div><div style='font-weight:800; font-size:22px; color:#fff;'>{metrics['mean']:.2f}</div>", unsafe_allow_html=True)
            m3.markdown(f"<div style='color:var(--muted); font-size:12px;'>Std Dev</div><div style='font-weight:800; font-size:22px; color:#fff;'>{metrics['std']:.2f}</div>", unsafe_allow_html=True)
            m4.markdown(f"<div style='color:var(--muted); font-size:12px;'>Cluster</div><div style='font-weight:800; font-size:22px; color:#fff;'>{cluster}</div>", unsafe_allow_html=True)

            # show RNFLT map image (display bigger)
            try:
                rnflt_img_show = Image.fromarray(np.uint8(255 * (rnflt_arr - np.nanmin(rnflt_arr)) / (np.nanmax(rnflt_arr) - np.nanmin(rnflt_arr) + 1e-9)))
                st.image(rnflt_img_show, caption="RNFLT map (preview)", width=display_width)
            except Exception:
                pass

            # show severity
            st.markdown(render_severity_html(severity_overall), unsafe_allow_html=True)

            # optional plots
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
            st.error(f"Error in RNFLT section: {e}")

    # B-scan processing
    if bscan_file is not None and b_model is not None:
        try:
            image_pil = Image.open(bscan_file).convert("L")
            batch, proc = preprocess_bscan(image_pil, size=(320,320))
            pred_raw = float(b_model.predict(batch, verbose=0)[0][0])
            label_b = "Glaucoma-like" if pred_raw > 0.5 else "Healthy-like"
            conf = pred_raw*100 if label_b=="Glaucoma-like" else (1-pred_raw)*100
            severity_overall = max(severity_overall, conf)

            st.markdown("<hr>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.markdown(f"<div style='color:var(--muted); font-size:12px;'>CNN Prediction</div><div style='font-weight:800; font-size:22px; color:#fff;'>{'🚨' if 'Glaucoma' in label_b else '✅'} {label_b}</div>", unsafe_allow_html=True)
            m2.markdown(f"<div style='color:var(--muted); font-size:12px;'>Confidence</div><div style='font-weight:800; font-size:22px; color:#fff;'>{conf:.2f}%</div>", unsafe_allow_html=True)
            st.markdown(render_severity_html(conf), unsafe_allow_html=True)

            heat = gradcam(batch, b_model)
            if heat is not None:
                # create heatmap overlay and upsample to display size
                hm_small = (cv2.resize(heat, (proc.shape[1], proc.shape[0])) * 255).astype(np.uint8)
                hm_color = cv2.applyColorMap(hm_small, cv2.COLORMAP_JET)
                overlay_small = (np.stack([proc]*3, axis=-1) * 255).astype(np.uint8)
                overlay_small = cv2.addWeighted(overlay_small, 0.6, hm_color, 0.4, 0)
                # upscale to original B-scan image size for preview
                orig_w, orig_h = image_pil.size
                overlay_up = cv2.resize(overlay_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                overlay_pil = Image.fromarray(overlay_up)
                st.image([image_pil.resize((display_width, int(display_width * orig_h / orig_w))), overlay_pil.resize((display_width, int(display_width * orig_h / orig_w)))],
                         caption=["Original B-Scan (preview)", "Grad-CAM Overlay (preview)"], width=display_width)
            else:
                st.image(image_pil, caption="Original B-Scan", width=display_width)
        except Exception as e:
            st.error(f"B-scan error: {e}")
    elif bscan_file is not None and b_model is None:
        st.warning("B-scan model unavailable (bscan_cnn.h5 not found) — uploading still allowed but prediction/grad-cam disabled.")

    # Combined severity header
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center'>Overall Severity Index</h3>", unsafe_allow_html=True)
    st.markdown(render_severity_html(severity_overall), unsafe_allow_html=True)

    # Download report buttons
    if figs:
        png_bytes = fig_to_png(figs[0])
        pdf_bytes = create_pdf(figs)
        st.markdown("<div style='text-align:center; margin-top:10px'>", unsafe_allow_html=True)
        st.download_button("📸 Download RNFLT PNG", data=png_bytes, file_name="oculaire_rnflt.png", mime="image/png")
        st.download_button("📄 Download Full Report (PDF)", data=pdf_bytes, file_name="oculaire_report.pdf", mime="application/pdf")
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:var(--muted);padding:6px;'>OCULAIRE Neon Lab v6 — For research use only</div>", unsafe_allow_html=True)

# -----------------------
# Floating Expander Chat (Streamlit-only, no JS navigation)
# -----------------------
st.markdown('<div class="floating-expander">', unsafe_allow_html=True)
# change label to "💬 Ask AI assistant" with neon look (CSS above)
with st.expander("💬 Ask AI assistant", expanded=False):
    st.markdown("<div style='font-weight:900; font-size:18px; margin-bottom:6px;'>🤖 OCULAIRE Assistant</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:var(--muted); margin-bottom:10px;'>Ask about glaucoma, OCT, RNFLT or interpretation of analysis.</div>", unsafe_allow_html=True)

    # show history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"<div style='padding:10px; border-radius:8px; background:rgba(0,245,255,0.06); margin-bottom:8px;'><strong>You:</strong> {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='padding:10px; border-radius:8px; background:rgba(255,64,196,0.06); margin-bottom:8px;'><strong>OCULAIRE:</strong> {msg['content']}</div>", unsafe_allow_html=True)

    user_question = st.text_input("Your question:", key="chat_input", placeholder="e.g., What is glaucoma? How does OCT detect it?", label_visibility="collapsed")
    col1, col2 = st.columns([4,1])
    with col1:
        send_btn = st.button("📤 Send")
    with col2:
        clear_btn = st.button("🗑️ Clear chat")

    if send_btn:
        if not user_question or user_question.strip() == "":
            st.warning("Please type a question first.")
        else:
            if not API_KEY:
                st.error("❌ Gemini API key not configured. Add GEMINI_API_KEY to secrets or environment.")
            else:
                with st.spinner("🔍 Searching for answers..."):
                    # append user query, call assistant, append reply
                    st.session_state.chat_history.append({"role":"user","content":user_question})
                    reply = ask_glaucoma_assistant(user_question, st.session_state.chat_history, API_KEY)
                    st.session_state.chat_history.append({"role":"assistant","content":reply})
                    # re-render by rerunning (safe)
                    st.experimental_rerun()

    if clear_btn:
        st.session_state.chat_history = []
        # rerun so the expander re-renders
        st.experimental_rerun()

st.markdown('</div>', unsafe_allow_html=True)
s
