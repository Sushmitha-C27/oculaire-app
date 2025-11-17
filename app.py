# app.py — OCULAIRE Neon Lab v5 (RNFLT image upload + Streamlit-only chat)
# Run: streamlit run app.py

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
if 'last_processed_q' not in st.session_state:
    st.session_state.last_processed_q = None

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
# Matplotlib / theme config
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
# CSS — Neon Theme + animations
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

/* Severity Bar — make it long and animated faster */
.sev-wrap { margin-top:16px; width:95%; margin-left:auto; margin-right:auto; }
.sev-outer { height:18px; width:100%; background: rgba(255,255,255,0.03); border-radius:14px; overflow:hidden; }
.sev-inner {
  height:100%; width:0%;
  background: linear-gradient(90deg,var(--neonA),var(--neonB));
  border-radius:14px;
  box-shadow: 0 0 25px rgba(0,245,255,0.6), 0 0 25px rgba(255,64,196,0.5);
  transition: width 0.8s ease-in-out;
  animation: sev-beat 1.2s infinite ease-in-out;
}
@keyframes sev-beat {
  0% { transform: scaleX(1); }
  50% { transform: scaleX(1.01); }
  100% { transform: scaleX(1); }
}
.sev-chip {
  margin-top:10px; display:inline-block;
  padding:6px 12px; border-radius:14px;
  font-weight:800; font-size:14px; color:#021617;
  background: linear-gradient(90deg, rgba(0,245,255,0.95), rgba(255,64,196,0.95));
  box-shadow: 0 0 20px rgba(0,245,255,0.4), 0 0 20px rgba(255,64,196,0.3);
}

/* Chat message styling */
.user-msg {
  background: linear-gradient(135deg, rgba(0,245,255,0.12), rgba(0,245,255,0.05));
  border-left: 3px solid var(--neonA);
  padding: 12px;
  border-radius: 8px;
  margin: 8px 0;
}
.assistant-msg {
  background: linear-gradient(135deg, rgba(255,64,196,0.12), rgba(255,64,196,0.05));
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
  margin-bottom: 12px;
  text-shadow: 0 0 20px rgba(0,245,255,0.3);
}

/* Floating Chat Expander at Bottom — neon beating */
.floating-expander {
  position: fixed !important;
  bottom: 20px !important;
  right: 20px !important;
  width: 460px !important;
  max-width: 92vw !important;
  z-index: 9999 !important;
  animation: float 3s ease-in-out infinite !important;
}
@keyframes float { 0%,100%{transform:translateY(0);}50%{transform:translateY(-6px);} }

.floating-expander details {
  background: linear-gradient(180deg, rgba(10,15,37,0.98), rgba(2,2,8,0.98)) !important;
  border: 2px solid rgba(0,245,255,0.35) !important;
  border-radius: 14px !important;
  animation: neon-pulse 1.8s ease-in-out infinite !important;
}
@keyframes neon-pulse {
  0% { box-shadow: 0 8px 30px rgba(0,245,255,0.08); }
  50% { box-shadow: 0 14px 50px rgba(0,245,255,0.16); }
  100% { box-shadow: 0 8px 30px rgba(0,245,255,0.08); }
}
.floating-expander details summary {
  background: linear-gradient(135deg, rgba(0,245,255,0.12), rgba(255,64,196,0.12)) !important;
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
.floating-expander details summary::before {
  content: "💬";
  font-size: 20px;
  display: inline-block;
  margin-right: 6px;
}
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Model name for REST (if using)
# -----------------------
MODEL_NAME = "models/gemini-2.5-pro"  # adjust as needed

# -----------------------
# Chatbot function
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

Important: Always remind users to consult healthcare professionals for medical decisions.
"""
    try:
        if USE_SDK:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(MODEL_NAME)
            chat_history = []
            for msg in history[-8:]:
                role = "user" if msg["role"] == "user" else "model"
                chat_history.append({"role": role, "parts": [msg["content"]]})
            chat = model.start_chat(history=chat_history)
            response = chat.send_message(f"{system_instruction}\n\nUser question: {question}")
            return response.text
        else:
            conversation_context = ""
            for msg in history[-8:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                conversation_context += f"{role}: {msg['content']}\n\n"
            full_prompt = f"{system_instruction}\n\n{conversation_context}User: {question}\n\nAssistant:"
            url = f"https://generativelanguage.googleapis.com/v1beta/{MODEL_NAME}:generateContent?key={api_key}"
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": full_prompt}]}],
                    "generationConfig": {"temperature": 0.6, "maxOutputTokens": 400}
                },
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                # Defensive access
                try:
                    return data["candidates"][0]["content"]["parts"][0]["text"]
                except Exception:
                    return json.dumps(data)[:1000]
            elif response.status_code == 403:
                return "🔑 API key invalid or restricted. Create an unrestricted key in Google AI Studio."
            else:
                return f"❌ Error ({response.status_code}): {response.text[:300]}"
    except Exception as e:
        return f"❌ Error calling Gemini: {e}"

# -----------------------
# Load Models & resources (cached)
# -----------------------
@st.cache_resource
def load_models_and_refs():
    # Load bscan model if available
    try:
        b_model = tf.keras.models.load_model("bscan_cnn.h5", compile=False)
    except Exception:
        b_model = None
    # Load scaler and kmeans if available (used previously with npz pipeline)
    try:
        scaler = joblib.load("rnflt_scaler.joblib")
        kmeans = joblib.load("rnflt_kmeans.joblib")
    except Exception:
        scaler = None
        kmeans = None
    # Load avg maps used for diff/risk (if present)
    try:
        avg_healthy = np.load("avg_map_healthy.npy")
        avg_glaucoma = np.load("avg_map_glaucoma.npy")
    except Exception:
        avg_healthy = None
        avg_glaucoma = None
    return b_model, scaler, kmeans, avg_healthy, avg_glaucoma

b_model, scaler, kmeans, avg_healthy, avg_glaucoma = load_models_and_refs()

# -----------------------
# Helpers: RNFLT image processor (new)
# -----------------------
def process_rnflt_image(f):
    """
    Accepts a Streamlit UploadedFile or file-like and returns normalized RNFLT numpy array and metrics.
    Strategy:
      - Load grayscale
      - Normalize to [0,1], then scale to µm-like range (we choose 0-110)
      - Return arr (float) and metrics dict
    """
    try:
        # read bytes (works for UploadFile)
        if hasattr(f, "getvalue"):
            content = f.getvalue()
            buf = io.BytesIO(content)
            img = Image.open(buf).convert("L")
        else:
            img = Image.open(f).convert("L")
        arr = np.array(img).astype(float)

        # mask central black optic disc: preserve but it's already dark; keep as-is
        # normalize robustly
        lo = np.nanpercentile(arr, 2)
        hi = np.nanpercentile(arr, 99)
        arr_clipped = np.clip(arr, lo, hi)
        arr_norm = (arr_clipped - lo) / (hi - lo + 1e-9)
        # scale to microns-like range (tweakable)
        arr_microns = arr_norm * 110.0  # 0 - 110 µm approximate scale
        metrics = {
            "mean": float(np.nanmean(arr_microns)),
            "std": float(np.nanstd(arr_microns)),
            "min": float(np.nanmin(arr_microns)),
            "max": float(np.nanmax(arr_microns)),
            "shape": arr_microns.shape
        }
        return arr_microns, metrics
    except Exception as e:
        st.error(f"Error reading RNFLT image: {e}")
        return None, None

def compute_risk_map(rnflt, healthy, threshold=-10):
    """
    rnflt: 2D float array
    healthy: reference healthy average map or None
    threshold: µm difference threshold (negative means thinner)
    """
    try:
        if healthy is None:
            # create dummy healthy: mean of rnflt as baseline
            healthy_resized = np.full_like(rnflt, np.nanmean(rnflt))
        else:
            healthy_resized = cv2.resize(healthy, (rnflt.shape[1], rnflt.shape[0]))
        diff = rnflt - healthy_resized
        risk = np.where(diff < threshold, diff, np.nan)
        total = np.isfinite(diff).sum()
        risky = np.isfinite(risk).sum()
        severity = (risky / total) * 100 if total else 0.0
        return diff, risk, severity
    except Exception as e:
        st.error(f"Error computing risk map: {e}")
        return None, None, 0.0

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
      }},120);
    </script>
    """
    return html

# -----------------------
# Sidebar — API status & instructions
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
    <strong>How to configure Gemini API key (Streamlit Cloud):</strong><br><br>
    1. App settings → Secrets<br>
    2. Add <code>GEMINI_API_KEY = "your-key"</code><br><br>
    For local testing: export GEMINI_API_KEY or create .streamlit/secrets.toml
    </div>
    """, unsafe_allow_html=True)

# -----------------------
# RNFLT / B-scan upload UI (now image-based RNFLT)
# -----------------------
colA, colB = st.columns(2)
with colA:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🩺 RNFLT Map Analysis (Image)")
    rnflt_file = st.file_uploader("Upload RNFLT Map (png/jpg/jpeg)", type=["png","jpg","jpeg"])
    st.markdown("</div>", unsafe_allow_html=True)

with colB:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("👁️ B-Scan Slice Analysis (Image)")
    bscan_file = st.file_uploader("Upload B-Scan Image", type=["jpg","png","jpeg"])
    st.markdown("</div>", unsafe_allow_html=True)

threshold = st.slider("Thin-zone threshold (µm)", 5, 50, 10)

# -----------------------
# ANALYSIS (RNFLT image processing + B-scan)
# -----------------------
if rnflt_file or bscan_file:
    figs = []
    severity_overall = 0.0
    st.markdown("<hr>", unsafe_allow_html=True)

    # RNFLT Image Processing
    if rnflt_file:
        rnflt_map, metrics = process_rnflt_image(rnflt_file)
        if rnflt_map is not None:
            # Attempt to label using existing scaler/kmeans if available; otherwise simple threshold heuristic
            if scaler is not None and kmeans is not None:
                try:
                    X = np.array([[metrics["mean"], metrics["std"], metrics["min"], metrics["max"]]])
                    Xs = scaler.transform(X)
                    cluster = int(kmeans.predict(Xs)[0])
                    # infer thin cluster from loaded avg maps if possible
                    thin_cluster_guess = 0
                    if avg_healthy is not None and avg_glaucoma is not None:
                        thin_cluster_guess = 0 if np.nanmean(avg_healthy) > np.nanmean(avg_glaucoma) else 1
                    label_r = "Glaucoma-like" if cluster == thin_cluster_guess else "Healthy-like"
                except Exception:
                    cluster = "-"
                    label_r = "Unknown"
            else:
                # simple heuristic using mean RNFLT (adjust threshold as appropriate)
                cluster = "-"
                label_r = "Glaucoma-like" if metrics["mean"] < 65 else "Healthy-like"

            # compute risk/diff maps — average healthy will be resized if present
            diff, risk, sev = compute_risk_map(rnflt_map, avg_healthy, -threshold)
            severity_overall = max(severity_overall, sev)

            # display metrics
            m1, m2, m3, m4 = st.columns([2,2,2,2])
            m1.markdown(f"<div class='metric-label'>Status</div><div class='large-metric'>{'🚨' if 'Glaucoma' in label_r else '✅'} {label_r}</div>", unsafe_allow_html=True)
            m2.markdown(f"<div class='metric-label'>Mean RNFLT (µm)</div><div class='large-metric'>{metrics['mean']:.2f}</div>", unsafe_allow_html=True)
            m3.markdown(f"<div class='metric-label'>Std Dev</div><div class='large-metric'>{metrics['std']:.2f}</div>", unsafe_allow_html=True)
            m4.markdown(f"<div class='metric-label'>Cluster</div><div class='large-metric'>{cluster}</div>", unsafe_allow_html=True)

            st.markdown(render_severity(sev), unsafe_allow_html=True)

            # Visualize RNFLT, diff, risk
            fig, axes = plt.subplots(1,3,figsize=(18,6),constrained_layout=True)
            ax0, ax1, ax2 = axes
            im0 = ax0.imshow(rnflt_map, cmap='gray')
            ax0.axis('off'); ax0.set_title("Uploaded RNFLT (µm)")
            plt.colorbar(im0, ax=ax0, shrink=0.8)

            if diff is not None:
                im1 = ax1.imshow(diff, cmap='bwr', vmin=-40, vmax=40)
                ax1.axis('off'); ax1.set_title("Difference (vs Healthy)")
                plt.colorbar(im1, ax=ax1, shrink=0.8)
            else:
                ax1.axis('off'); ax1.set_title("Difference (n/a)")

            if risk is not None:
                im2 = ax2.imshow(risk, cmap='hot')
                ax2.axis('off'); ax2.set_title("Risk Map (thin zones)")
                plt.colorbar(im2, ax=ax2, shrink=0.8)
            else:
                ax2.axis('off'); ax2.set_title("Risk Map (n/a)")

            fig.patch.set_facecolor("#050612")
            st.pyplot(fig)
            figs.append(fig)

    # B-Scan Processing
    if bscan_file and b_model is not None:
        try:
            image_pil = Image.open(bscan_file).convert("L")
            batch, proc = preprocess_bscan(image_pil)
            pred_raw = float(b_model.predict(batch, verbose=0)[0][0])
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
        except Exception as e:
            st.error(f"B-scan processing error: {e}")

    # Combined severity display + downloads
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align:center'>Overall Severity Index</h4>", unsafe_allow_html=True)
    st.markdown(render_severity(severity_overall), unsafe_allow_html=True)

    if figs:
        png_bytes = fig_to_png(figs[0])
        pdf_bytes = create_pdf(figs)
        st.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
        st.download_button("📸 Download RNFLT PNG", data=png_bytes, file_name="oculaire_rnflt.png", mime="image/png")
        st.download_button("📄 Download Full Report (PDF)", data=pdf_bytes, file_name="oculaire_report.pdf", mime="application/pdf")
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:var(--muted);padding:6px;'>OCULAIRE Neon Lab v5 — For research use only</div>", unsafe_allow_html=True)

# -----------------------
# Floating Expander Chat (Streamlit-only, no JS navigation)
# -----------------------
st.markdown('<div class="floating-expander">', unsafe_allow_html=True)
with st.expander("💬 Ask AI assistant", expanded=False):
    st.markdown("<div class='chat-header'>🤖 OCULAIRE Assistant</div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:var(--muted); font-size:13px; margin-bottom:8px;'>Ask me about glaucoma, OCT, RNFLT, or eye health.</p>", unsafe_allow_html=True)

    # display chat history (newest last)
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-msg'><strong>You:</strong> {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-msg'><strong>🤖:</strong> {msg['content']}</div>", unsafe_allow_html=True)

    user_question = st.text_input("Your question:", key="chat_input", placeholder="e.g., What is glaucoma? How does OCT detect it?", label_visibility="collapsed")
    col1, col2 = st.columns([4,1])
    with col1:
        send_btn = st.button("📤 Send", use_container_width=True)
    with col2:
        clear_btn = st.button("🗑️", use_container_width=True)

    if send_btn:
        q = (user_question or "").strip()
        if not q:
            st.warning("Please enter a question.")
        else:
            # Prevent duplicate processing across reruns quickly
            if q != st.session_state.get("last_processed_q"):
                st.session_state.last_processed_q = q
                st.session_state.chat_history.append({"role": "user", "content": q})
                # call assistant
                with st.spinner("🔍 Searching for answers..."):
                    reply = ask_glaucoma_assistant(q, st.session_state.chat_history, API_KEY)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                # rerun to show updated history and clear text input
                # To avoid immediate errors, use a tiny sleep and then rerun
                time.sleep(0.05)
                st.experimental_rerun()

    if clear_btn:
        st.session_state.chat_history = []
        st.experimental_rerun()

st.markdown('</div>', unsafe_allow_html=True)
