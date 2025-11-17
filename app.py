# app.py — OCULAIRE Neon Lab v6.2 (no experimental_rerun, sidebar B-scan options)
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

# Optional Google SDK
try:
    import google.generativeai as genai
    USE_SDK = True
except Exception:
    USE_SDK = False

# Page config
st.set_page_config(page_title="OCULAIRE: Neon Glaucoma Detection Dashboard",
                   layout="wide",
                   page_icon="👁️")

# ---- Session state init ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# keep a small toggle to trigger UI updates if needed
if "ui_tick" not in st.session_state:
    st.session_state.ui_tick = 0

# ---- Helpers: API key ----
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
MODEL_NAME = "models/gemini-2.5-pro"  # change if needed

# ---- Plot styling (neon) ----
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

# ---- CSS (neon, longer bar, beating expander) ----
st.markdown("""
<style>
:root {
  --bg:#020208;
  --panel:#0a0f25;
  --neonA:#00f5ff;
  --neonB:#ff40c4;
  --muted:#a4b1c9;
}
.stApp { background: radial-gradient(circle at 20% 20%, #091133, #020208 90%); color:#e6faff; font-family: Inter, system-ui; }

/* header */
.header { text-align:center; margin-top:10px; margin-bottom:10px; }
.header h1 { font-size:42px; font-weight:900; letter-spacing:3px;
  background: linear-gradient(90deg, var(--neonA), var(--neonB));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  text-shadow: 0 0 20px rgba(0,245,255,0.8), 0 0 35px rgba(255,64,196,0.5);
}
.header h3 { color:var(--muted); font-weight:400; font-size:15px; }

/* card */
.card { background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)); border:1px solid rgba(255,255,255,0.05); box-shadow:0 0 25px rgba(0,245,255,0.05); border-radius:12px; padding:16px; }

/* severity bar: longer / faster beat */
.sev-wrap { margin-top:16px; }
.sev-outer { height:18px; width:94%; margin: 8px auto; background: rgba(255,255,255,0.03); border-radius:14px; overflow:hidden; }
.sev-inner { height:100%; width:0%; background: linear-gradient(90deg,var(--neonA),var(--neonB)); border-radius:14px; box-shadow: 0 0 25px rgba(0,245,255,0.6), 0 0 25px rgba(255,64,196,0.5); transition: width 0.5s ease-in-out; }
.sev-chip { margin-top:10px; display:inline-block; padding:8px 14px; border-radius:18px; font-weight:800; font-size:14px; color:#021617; background: linear-gradient(90deg, rgba(0,245,255,0.95), rgba(255,64,196,0.95)); box-shadow: 0 0 30px rgba(0,245,255,0.25), 0 0 40px rgba(255,64,196,0.2); animation: pulse 0.9s infinite; }
@keyframes pulse { 0%{transform:scale(1);} 50%{transform:scale(1.07);} 100%{transform:scale(1);} }

/* floating expander neon style */
.floating-expander details summary {
  background: linear-gradient(135deg, rgba(0,245,255,0.25), rgba(255,64,196,0.25)) !important;
  padding: 14px !important; border-radius: 14px !important; cursor:pointer !important; font-weight:800 !important; font-size:16px !important; color:#e6faff !important; display:flex !important; align-items:center !important; gap:10px !important; transition:all 0.25s ease !important;
}
.floating-expander details summary::before { content: "💬"; font-size:22px; margin-right:6px; }

/* small adjustments */
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ---- Chat assistant (Gemini) ----
def ask_glaucoma_assistant(question, history, api_key):
    """Call Google Gemini API or REST fallback"""
    if not api_key or not api_key.strip():
        return "⚠️ Please configure your Google Gemini API key in secrets or environment."
    system_instruction = ("You are a specialized medical assistant focused on glaucoma, OCT and RNFLT. "
                          "Answer concisely (<=200 words), include a short educational disclaimer.")
    try:
        if USE_SDK:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(MODEL_NAME) if MODEL_NAME else genai.GenerativeModel("gemini-1.5-flash")
            chat_history = []
            for msg in history[-8:]:
                role = "user" if msg["role"] == "user" else "model"
                chat_history.append({"role": role, "parts": [msg["content"]]})
            chat = model.start_chat(history=chat_history)
            response = chat.send_message(f"{system_instruction}\n\nUser: {question}")
            return response.text
        else:
            conversation_context = ""
            for msg in history[-8:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                conversation_context += f"{role}: {msg['content']}\n\n"
            full_prompt = f"{system_instruction}\n\n{conversation_context}User: {question}\n\nAssistant:"
            model_path = MODEL_NAME if MODEL_NAME else "gemini-1.5-flash-latest"
            url = f"https://generativelanguage.googleapis.com/v1beta/{model_path}:generateContent?key={api_key}"
            resp = requests.post(url, headers={"Content-Type":"application/json"}, json={
                "contents":[{"parts":[{"text": full_prompt}]}],
                "generationConfig":{"temperature":0.7,"maxOutputTokens":400}
            }, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                try:
                    return data["candidates"][0]["content"]["parts"][0]["text"]
                except Exception:
                    return json.dumps(data)[:1000]
            elif resp.status_code == 403:
                return "🔑 API key invalid or restricted. Check your key at Google AI Studio."
            else:
                return f"❌ API Error ({resp.status_code})"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ---- Model/resource loader (defensive) ----
@st.cache_resource
def load_models():
    b_model = None
    scaler = None
    kmeans = None
    avg_healthy = None
    avg_glaucoma = None
    thin_cluster = None
    # b-scan CNN
    try:
        b_model = tf.keras.models.load_model("bscan_cnn.h5", compile=False)
    except Exception:
        b_model = None
    # RNFLT resources
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

# ---- Helpers for inputs ----
def process_npz_file(file_like):
    try:
        buf = io.BytesIO(file_like.getvalue())
        data = np.load(buf, allow_pickle=True)
        if "volume" in data:
            arr = data["volume"]
        else:
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
        image = Image.open(file_like).convert("L")
        arr = np.array(image).astype(float)
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
      setTimeout(function(){{
        var el=document.getElementById('sev_inner');
        if(el) el.style.width='{pct:.1f}%';
      }},80);
    </script>
    """
    return html

# ---- Sidebar: RNFLT & B-scan input & converters ----
st.sidebar.title("RNFLT & B-scan Input Tools")

# RNFLT input mode
rnflt_input_mode = st.sidebar.radio("RNFLT input type", ["NPZ (recommended)", "Image (single RNFLT image)"])
st.sidebar.markdown("---")
st.sidebar.subheader("RNFLT: Image → NPZ converter")
st.sidebar.markdown("If you only have RNFLT slice images, upload a sequence (PNG/JPG). I'll pack them into a `volume` `.npz`.")
rnflt_conv_files = st.sidebar.file_uploader("Upload RNFLT slice images (ordered)", accept_multiple_files=True, type=["png","jpg","jpeg"], key="rnflt_conv")
if rnflt_conv_files:
    if st.sidebar.button("Convert RNFLT images → .npz", key="conv_rnflt_btn"):
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
            st.sidebar.success(f"Packed {len(stacks)} slices into volume {vol.shape}")
            st.sidebar.download_button("⬇️ Download RNFLT volume (.npz)", data=buf.getvalue(), file_name="rnflt_volume.npz", mime="application/octet-stream", key="dl_rnflt_npz")
        except Exception as e:
            st.sidebar.error(f"Conversion error: {e}")

st.sidebar.markdown("---")
# B-scan input mode + converter
bscan_input_mode = st.sidebar.radio("B-scan input type", ["Single Image", "Volume NPZ (many)"])
st.sidebar.subheader("B-scan: Image → NPZ converter")
bscan_conv_files = st.sidebar.file_uploader("Upload B-scan slice images (ordered)", accept_multiple_files=True, type=["png","jpg","jpeg"], key="bscan_conv")
if bscan_conv_files:
    if st.sidebar.button("Convert B-scan images → .npz", key="conv_bscan_btn"):
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
            st.sidebar.success(f"Packed {len(stacks)} slices into volume {vol.shape}")
            st.sidebar.download_button("⬇️ Download B-scan volume (.npz)", data=buf.getvalue(), file_name="bscan_volume.npz", mime="application/octet-stream", key="dl_bscan_npz")
        except Exception as e:
            st.sidebar.error(f"Conversion error: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("⚠️ NPZ recommended for full maps. Images supported but may lose metadata.")

# ---- Header ----
st.markdown("""
<div class="header">
  <h1>👁️ OCULAIRE</h1>
  <h3>AI-Powered Glaucoma Detection Dashboard — Neon Lab v6.2</h3>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ---- Main inputs ----
colA, colB = st.columns(2)

with colA:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🩺 RNFLT Map Analysis")
    rnflt_arr = None
    rnflt_metrics = None
    rnflt_pil = None
    if rnflt_input_mode == "NPZ (recommended)":
        rnflt_file = st.file_uploader("Upload RNFLT (.npz)", type=["npz"], key="rnflt_file")
        if rnflt_file:
            rnflt_arr, rnflt_metrics = process_npz_file(rnflt_file)
    else:
        rnflt_img_file = st.file_uploader("Upload RNFLT image (single)", type=["png","jpg","jpeg"], key="rnflt_img")
        if rnflt_img_file:
            rnflt_arr, rnflt_metrics, rnflt_pil = process_image_rnflt(rnflt_img_file)
    st.markdown("</div>", unsafe_allow_html=True)

with colB:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("👁️ B-Scan Analysis")
    bscan_file = None
    bscan_volume = None
    if bscan_input_mode == "Single Image":
        bscan_file = st.file_uploader("Upload B-Scan image", type=["png","jpg","jpeg"], key="bscan_file")
    else:
        bscan_npz_file = st.file_uploader("Upload B-scan volume (.npz)", type=["npz"], key="bscan_npz_file")
        if bscan_npz_file:
            try:
                buf = io.BytesIO(bscan_npz_file.getvalue())
                data = np.load(buf, allow_pickle=True)
                bscan_volume = data["volume"] if "volume" in data else data[data.files[0]]
            except Exception as e:
                st.error(f"B-scan NPZ read error: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

threshold = st.slider("Thin-zone threshold (µm)", 5, 50, 10)

# ---- Analysis ----
if rnflt_arr is not None or bscan_file is not None or bscan_volume is not None:
    figs = []
    severity_overall = 0.0
    st.markdown("<hr>", unsafe_allow_html=True)

    # RNFLT processing
    if rnflt_arr is not None:
        try:
            metrics = rnflt_metrics
            label_r = "Unknown"
            cluster = "?"
            if scaler is not None and kmeans is not None:
                X = np.array([[metrics["mean"], metrics["std"], metrics["min"], metrics["max"]]])
                Xs = scaler.transform(X)
                cluster = int(kmeans.predict(Xs)[0])
                label_r = "Glaucoma-like" if cluster == thin_cluster else "Healthy-like"
            # compute diff/risk
            if avg_healthy is not None:
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
                sev = float(np.nanpercentile(np.nan_to_num(diff), 75))
            severity_overall = max(severity_overall, float(sev))
            # display metrics
            m1, m2, m3, m4 = st.columns([2,2,2,2])
            m1.markdown(f"<div style='color:var(--muted); font-size:12px;'>Status</div><div style='font-weight:800; font-size:22px;'>{'🚨' if 'Glaucoma' in label_r else '✅'} {label_r}</div>", unsafe_allow_html=True)
            m2.markdown(f"<div style='color:var(--muted); font-size:12px;'>Mean RNFLT</div><div style='font-weight:800; font-size:22px;'>{metrics['mean']:.2f}</div>", unsafe_allow_html=True)
            m3.markdown(f"<div style='color:var(--muted); font-size:12px;'>Std Dev</div><div style='font-weight:800; font-size:22px;'>{metrics['std']:.2f}</div>", unsafe_allow_html=True)
            m4.markdown(f"<div style='color:var(--muted); font-size:12px;'>Cluster</div><div style='font-weight:800; font-size:22px;'>{cluster}</div>", unsafe_allow_html=True)
            # preview RNFLT map
            try:
                img_preview = np.uint8(255 * (rnflt_arr - np.nanmin(rnflt_arr)) / (np.nanmax(rnflt_arr) - np.nanmin(rnflt_arr) + 1e-9))
                st.image(Image.fromarray(img_preview), caption="RNFLT map (preview)", width=640)
            except Exception:
                pass
            # severity widget
            st.markdown(render_severity_html(severity_overall), unsafe_allow_html=True)
            # plots
            fig, axes = plt.subplots(1,3,figsize=(18,5), constrained_layout=True)
            axes[0].imshow(rnflt_arr, cmap='turbo'); axes[0].axis('off'); axes[0].set_title("Uploaded RNFLT")
            axes[1].imshow(diff, cmap='bwr', vmin=-30, vmax=30); axes[1].axis('off'); axes[1].set_title("Difference (vs Healthy)")
            axes[2].imshow(risk, cmap='hot'); axes[2].axis('off'); axes[2].set_title("Risk Map")
            for ax in axes: ax.set_facecolor('#050612')
            fig.patch.set_facecolor("#050612")
            st.pyplot(fig)
            figs.append(fig)
        except Exception as e:
            st.error(f"Error in RNFLT processing: {e}")

    # B-scan single image processing
    if bscan_file is not None:
        try:
            image_pil = Image.open(bscan_file).convert("L")
            batch, proc = preprocess_bscan(image_pil, size=(320,320))
            if b_model is not None:
                pred_raw = float(b_model.predict(batch, verbose=0)[0][0])
                label_b = "Glaucoma-like" if pred_raw > 0.5 else "Healthy-like"
                conf = pred_raw*100 if label_b == "Glaucoma-like" else (1-pred_raw)*100
                severity_overall = max(severity_overall, conf)
                st.markdown("<hr>", unsafe_allow_html=True)
                m1, m2 = st.columns(2)
                m1.markdown(f"<div style='color:var(--muted); font-size:12px;'>CNN Prediction</div><div style='font-weight:800; font-size:22px;'>{'🚨' if 'Glaucoma' in label_b else '✅'} {label_b}</div>", unsafe_allow_html=True)
                m2.markdown(f"<div style='color:var(--muted); font-size:12px;'>Confidence</div><div style='font-weight:800; font-size:22px;'>{conf:.2f}%</div>", unsafe_allow_html=True)
                st.markdown(render_severity_html(conf), unsafe_allow_html=True)
            else:
                st.warning("B-scan model not found — preview only.")
                st.image(image_pil, width=640)
            heat = gradcam(batch, b_model) if b_model is not None else None
            if heat is not None:
                hm_small = (cv2.resize(heat, (proc.shape[1], proc.shape[0])) * 255).astype(np.uint8)
                hm_color = cv2.applyColorMap(hm_small, cv2.COLORMAP_JET)
                overlay_small = (np.stack([proc]*3, axis=-1)*255).astype(np.uint8)
                overlay_small = cv2.addWeighted(overlay_small, 0.6, hm_color, 0.4, 0)
                orig_w, orig_h = image_pil.size
                overlay_up = cv2.resize(overlay_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                overlay_pil = Image.fromarray(overlay_up)
                st.image([image_pil.resize((640, int(640 * orig_h / orig_w))), overlay_pil.resize((640, int(640 * orig_h / orig_w)))],
                         caption=["Original B-Scan (preview)", "Grad-CAM Overlay (preview)"], width=640)
        except Exception as e:
            st.error(f"B-scan processing error: {e}")

    # B-scan volume preview (first slice)
    if bscan_volume is not None:
        try:
            slice0 = bscan_volume[0]
            img_preview = np.uint8(255 * (slice0 - np.nanmin(slice0)) / (np.nanmax(slice0) - np.nanmin(slice0) + 1e-9))
            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader("B-scan volume preview (first slice)")
            st.image(Image.fromarray(img_preview), width=640)
            st.info(f"B-scan volume shape: {bscan_volume.shape}")
        except Exception as e:
            st.error(f"B-scan volume display error: {e}")

    # Combined severity summary
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center'>Overall Severity Index</h3>", unsafe_allow_html=True)
    st.markdown(render_severity_html(severity_overall), unsafe_allow_html=True)

    # downloads if we created plots
    if figs:
        png_bytes = fig_to_png(figs[0])
        pdf_bytes = create_pdf(figs)
        st.markdown("<div style='text-align:center; margin-top:10px'>", unsafe_allow_html=True)
        st.download_button("📸 Download RNFLT PNG", data=png_bytes, file_name="oculaire_rnflt.png", mime="image/png")
        st.download_button("📄 Download Full Report (PDF)", data=pdf_bytes, file_name="oculaire_report.pdf", mime="application/pdf")
        st.markdown("</div>", unsafe_allow_html=True)

# ---- footer ----
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:var(--muted);padding:6px;'>OCULAIRE Neon Lab v6.2 — For research use only</div>", unsafe_allow_html=True)

# ---- Floating expander chat (Streamlit widgets only) ----
st.markdown('<div class="floating-expander">', unsafe_allow_html=True)
with st.expander("💬 Ask AI assistant", expanded=False):
    st.markdown("<div style='font-weight:900; font-size:18px; margin-bottom:6px;'>🤖 OCULAIRE Assistant</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:var(--muted); margin-bottom:10px;'>Ask about glaucoma, OCT, RNFLT or interpretation of analysis.</div>", unsafe_allow_html=True)

    # history display
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"<div style='padding:10px;border-radius:8px;background:rgba(0,245,255,0.06);margin-bottom:8px;'><strong>You:</strong> {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='padding:10px;border-radius:8px;background:rgba(255,64,196,0.06);margin-bottom:8px;'><strong>OCULAIRE:</strong> {msg['content']}</div>", unsafe_allow_html=True)

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
                # append and call assistant — no experimental_rerun()
                st.session_state.chat_history.append({"role":"user","content":user_question})
                reply = ask_glaucoma_assistant(user_question, st.session_state.chat_history, API_KEY)
                st.session_state.chat_history.append({"role":"assistant","content":reply})
                # update a ui tick so re-render shows new messages (Streamlit re-runs automatically)
                st.session_state.ui_tick = st.session_state.ui_tick + 1

    if clear_btn:
        st.session_state.chat_history = []
        st.session_state.ui_tick = st.session_state.ui_tick + 1

st.markdown('</div>', unsafe_allow_html=True)
