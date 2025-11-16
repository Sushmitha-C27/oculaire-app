# app.py — OCULAIRE Neon Lab v5 with Glaucoma Chatbot (query-param toggle)
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

# Try to import google.generativeai, fallback to requests
try:
    import google.generativeai as genai
    USE_SDK = True
except ImportError:
    import requests
    import json
    USE_SDK = False

# -----------------------
# Page Config
# -----------------------
st.set_page_config(page_title="OCULAIRE: Neon Glaucoma Detection Dashboard",
                   layout="wide",
                   page_icon="👁️")

# -----------------------
# Initialize Session State for Chat (prevent AttributeError)
# -----------------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chat_open' not in st.session_state:
    st.session_state.chat_open = False
if 'chat_input' not in st.session_state:
    st.session_state.chat_input = ""

# -----------------------
# Handle query param toggle (robust)
# If URL contains ?open_chat=1, open the chat and then clear params
# -----------------------
params = st.experimental_get_query_params()
if "open_chat" in params and params.get("open_chat", [""])[0] in ("1", "true", "True"):
    st.session_state.chat_open = True
    # clear the param so future clicks will still add it again
    st.experimental_set_query_params()

# Get API key from Streamlit secrets or environment variable
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
# CSS — Neon Theme + Animations (unchanged)
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

/* Floating Chat Bubble */
.chat-bubble {
  position: fixed;
  bottom: 30px;
  right: 30px;
  width: 70px;
  height: 70px;
  border-radius: 50%;
  background: linear-gradient(135deg, var(--neonA), var(--neonB));
  box-shadow: 0 0 30px rgba(0,245,255,0.6), 0 0 40px rgba(255,64,196,0.5);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 32px;
  z-index: 9999;
  animation: float 3s ease-in-out infinite, glow 2s ease-in-out infinite;
  transition: transform 0.3s ease;
}
.chat-bubble:hover {
  transform: scale(1.1);
  box-shadow: 0 0 40px rgba(0,245,255,0.8), 0 0 50px rgba(255,64,196,0.7);
}
@keyframes float {
  0%, 100% { transform: translateY(0px); }
  50% { transform: translateY(-10px); }
}
@keyframes glow {
  0%, 100% { box-shadow: 0 0 30px rgba(0,245,255,0.6), 0 0 40px rgba(255,64,196,0.5); }
  50% { box-shadow: 0 0 40px rgba(0,245,255,0.9), 0 0 60px rgba(255,64,196,0.8); }
}

/* Floating Chat Pill (text) */
.floating-chat-pill {
  position: fixed;
  bottom: 36px;
  right: 110px;
  z-index: 9999;
  background: linear-gradient(135deg, rgba(0,245,255,0.08), rgba(255,64,196,0.06));
  padding: 12px 18px;
  border-radius: 30px;
  color: #e6faff;
  font-weight: 800;
  cursor: pointer;
  box-shadow: 0 8px 40px rgba(0,0,0,0.4);
  transition: transform 0.12s ease;
}
.floating-chat-pill:hover { transform: translateY(-4px); }

footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Chatbot Function (unchanged)
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
# Header + model loading + helpers (same as you had)...
# For brevity I'm including the same load_models and helper functions:
# load_models(), process_npz(), compute_risk_map(), preprocess_bscan(), gradcam(), fig_to_png(), create_pdf(), render_severity()
# (Paste exactly the helper code from your previous file into these functions)
# -----------------------

# -- Insert your load_models and helper functions here (copy from your existing app) --
# To keep this example concise, I'll re-add them quickly:

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
# Main UI layout (uploads, analysis, etc.) — keep as you had
# -----------------------
st.markdown("""
<div class="header">
  <h1>👁️ OCULAIRE</h1>
  <h3>AI-Powered Glaucoma Detection Dashboard — Neon Lab v5</h3>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

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

# ANALYSIS (unchanged)
if rnflt_file or bscan_file:
    figs = []
    severity_overall = 0
    st.markdown("<hr>", unsafe_allow_html=True)

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

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align:center'>Overall Severity Index</h4>", unsafe_allow_html=True)
    st.markdown(render_severity(severity_overall), unsafe_allow_html=True)

    if figs:
        png_bytes = fig_to_png(figs[0])
        pdf_bytes = create_pdf(figs)
        st.markdown("<div class='download-btns'>", unsafe_allow_html=True)
        st.download_button("📸 Download RNFLT PNG", data=png_bytes, file_name="oculaire_rnflt.png", mime="image/png")
        st.download_button("📄 Download Full Report (PDF)", data=pdf_bytes, file_name="oculaire_report.pdf", mime="application/pdf")
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:var(--muted);padding:6px;'>OCULAIRE Neon Lab v5 — For research use only</div>", unsafe_allow_html=True)

# -----------------------
# Floating bubble JS: modify URL to add ?open_chat=1 (works in iframe + direct)
# -----------------------
st.markdown("""
<script>
(function(){
  function addOpenChatParam() {
    try {
      // Build new URL preserving existing params but setting open_chat=1
      const url = new URL(window.location.href);
      url.searchParams.set('open_chat', '1');
      // use location.replace to avoid creating history entries if you prefer
      window.location.href = url.toString();
    } catch (e) {
      console.error("Failed to set open_chat param", e);
    }
  }
  const bubble = document.getElementById('chatBubble');
  const pill = document.getElementById('chatPill');
  [bubble, pill].forEach(el => {
    if (!el) return;
    el.style.cursor = 'pointer';
    el.addEventListener('click', function(e){
      e.preventDefault();
      // small visual feedback
      el.style.transform = 'scale(0.98)';
      setTimeout(()=> el.style.transform = '', 120);
      addOpenChatParam();
    });
  });
})();
</script>
""", unsafe_allow_html=True)

# -----------------------
# When chat is open, show in sidebar (same as before)
# -----------------------
if st.session_state.chat_open:
    with st.sidebar:
        st.markdown("---")
        st.markdown("<div class='chat-header'>🤖 Glaucoma Assistant</div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:var(--muted); font-size:13px; margin-bottom:15px;'>Ask me anything about glaucoma!</p>", unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"<div class='user-msg'><strong>You:</strong> {msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='assistant-msg'><strong>🤖:</strong> {msg['content']}</div>", unsafe_allow_html=True)
        user_question = st.text_input("Your question:", key="chat_input", placeholder="What is glaucoma?")
        col1, col2, col3 = st.columns([3,1,1])
        with col1:
            if st.button("📤 Send", use_container_width=True):
                if user_question and API_KEY:
                    with st.spinner("🔍 Thinking..."):
                        response = ask_glaucoma_assistant(user_question, st.session_state.chat_history, API_KEY)
                        st.session_state.chat_history.append({"role": "user", "content": user_question})
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                        st.session_state.chat_input = ""
                        st.experimental_rerun()
                elif not API_KEY:
                    st.error("❌ No API key")
        with col2:
            if st.button("🗑️", use_container_width=True):
                st.session_state.chat_history = []
                st.experimental_rerun()
        with col3:
            if st.button("✖️", use_container_width=True):
                st.session_state.chat_open = False
                st.experimental_rerun()
