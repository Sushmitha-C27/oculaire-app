# app.py — OCULAIRE Neon Lab v5 with Floating Glaucoma Chatbot
import os
import io
import joblib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st
from PIL import Image
import cv2
from matplotlib.backends.backend_pdf import PdfPages

# Optional Gemini SDK
try:
    import google.generativeai as genai
    USE_SDK = True
except Exception:
    import requests
    USE_SDK = False

# -----------------------
# Configuration
# -----------------------
st.set_page_config(page_title="OCULAIRE: Neon Glaucoma Detection Dashboard",
                   layout="wide", page_icon="👁️")

# Model name for Gemini calls (change if you prefer another)
MODEL_NAME = "models/gemini-2.5-pro"

# -----------------------
# Session state defaults (avoid AttributeError)
# -----------------------
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [{"role": "assistant", "content": "Hello — I'm the Glaucoma Assistant. Ask me glaucoma-specific questions."}]
if "chat_open" not in st.session_state:
    st.session_state["chat_open"] = False
if "chat_input" not in st.session_state:
    st.session_state["chat_input"] = ""

# -----------------------
# Get API key (secrets preferred)
# -----------------------
def get_gemini_key():
    # Streamlit secrets: add to .streamlit/secrets.toml => GEMINI_API_KEY = "..."
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    # fallback environment var
    return os.environ.get("GEMINI_API_KEY", None)

GEMINI_API_KEY = get_gemini_key()

# Configure plotting for dark/neon
plt.style.use("dark_background")
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
# CSS: neon theme + bubble
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
    .header { text-align:center; margin-top:8px; margin-bottom:6px; }
    .header h1 {
      font-size:42px; font-weight:900; letter-spacing:3px;
      background: linear-gradient(90deg, var(--neonA), var(--neonB));
      -webkit-background-clip:text; -webkit-text-fill-color:transparent;
      text-shadow: 0 0 20px rgba(0,245,255,0.7), 0 0 35px rgba(255,64,196,0.4);
    }
    .header h3 { color:var(--muted); font-weight:400; font-size:14px; }
    .card { background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
      border:1px solid rgba(255,255,255,0.04); box-shadow: 0 0 25px rgba(0,245,255,0.04);
      border-radius:12px; padding:14px; }
    .metric-label { color:var(--muted); font-size:12px; }
    .large-metric { font-weight:800; font-size:22px; color:#fff; text-shadow:0 0 15px rgba(0,245,255,0.35); }
    .sev-wrap { margin-top:12px; }
    .sev-outer { height:18px; width:100%; background: rgba(255,255,255,0.04); border-radius:14px; overflow:hidden; }
    .sev-inner { height:100%; width:0%; background: linear-gradient(90deg,var(--neonA),var(--neonB));
      border-radius:14px; box-shadow: 0 0 25px rgba(0,245,255,0.55); transition: width 1s ease-in-out; }
    .sev-chip { margin-top:8px; display:inline-block; padding:6px 12px; border-radius:12px; font-weight:800; font-size:14px; color:#021617;
      background: linear-gradient(90deg, rgba(0,245,255,0.95), rgba(255,64,196,0.95)); box-shadow: 0 0 20px rgba(0,245,255,0.3); }
    .download-btns { margin-top:10px; display:flex; gap:10px; justify-content:flex-start; }
    .chat-header { text-align:center; color:var(--neonA); font-weight:700; padding:8px; }
    .user-msg { background: linear-gradient(135deg, rgba(0,245,255,0.12), rgba(0,245,255,0.03)); border-left: 3px solid var(--neonA); padding:12px; border-radius:8px; margin:8px 0; }
    .assistant-msg { background: linear-gradient(135deg, rgba(255,64,196,0.12), rgba(255,64,196,0.03)); border-left: 3px solid var(--neonB); padding:12px; border-radius:8px; margin:8px 0; }
    /* Floating chat bubble/pill */
    #oculaireChatBubble { position: fixed; right: 30px; bottom: 30px; z-index: 9999; display:flex; align-items:center; gap:12px; }
    #oculaireBubble { width:72px; height:72px; border-radius:999px; background:linear-gradient(135deg,#00f5ff,#ff40c4); display:flex; align-items:center; justify-content:center; font-size:34px; cursor:pointer; box-shadow:0 0 40px rgba(0,245,255,0.35); }
    #oculairePill { min-width:140px; padding:12px 18px; border-radius:40px; background: linear-gradient(135deg, rgba(0,245,255,0.07), rgba(255,64,196,0.06)); color:#e6faff; font-weight:800; cursor:pointer; }
    /* Hide the toggle button visually but keep it in DOM for JS click */
    .__oculaire_hidden_button_wrapper { position: absolute !important; left: -9999px !important; top: -9999px !important; opacity: 0 !important; height: 1px !important; width: 1px !important; overflow: hidden !important; }
    footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# Utility helpers (images/pdf)
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

def render_severity_html(pct):
    pct = float(max(0.0, min(100.0, pct)))
    html = f"""
    <div class='sev-wrap'>
      <div class='sev-outer'><div id='sev_inner' class='sev-inner' style='width:0%'></div></div>
      <div style='text-align:center'><div class='sev-chip'>{pct:.1f}%</div></div>
    </div>
    <script>
      setTimeout(function(){{ var el=document.getElementById('sev_inner'); if(el) el.style.width='{pct:.1f}%'; }}, 120);
    </script>
    """
    return html

# -----------------------
# Chatbot logic
# -----------------------
SYSTEM_INSTRUCTION = (
    "You are an expert assistant specialized in glaucoma. "
    "Only answer glaucoma / OCT / RNFLT / optic nerve / intraocular pressure related questions. "
    "If a user asks outside the domain, politely state you can only answer glaucoma-specific questions. "
    "Keep answers concise and include a short educational disclaimer."
)

def ask_glaucoma_assistant(question, history, api_key):
    """Call Gemini SDK or REST, fallback to a simple rule-based reply if unavailable."""
    # Basic input guard
    if not question or not question.strip():
        return "Please enter a question about glaucoma."

    # If SDK installed and key provided, try SDK call
    if USE_SDK and api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(MODEL_NAME)
            # Create small history (last few)
            conv = []
            for m in history[-6:]:
                role = "user" if m["role"] == "user" else "model"
                conv.append({"role": role, "parts": [m["content"]]})
            chat = model.start_chat(history=conv)
            prompt = SYSTEM_INSTRUCTION + "\n\nUser: " + question
            resp = chat.send_message(prompt)
            return getattr(resp, "text", str(resp))
        except Exception as e:
            # Fall back to REST approach or rule-based if SDK call fails
            # but surface a friendly error message
            return f"⚠️ AI service error (SDK): {str(e)} — showing fallback answer.\n\n" + fallback_reply(question)

    # If no SDK or SDK failed, attempt REST call if key present
    if api_key and not USE_SDK:
        try:
            # Build simple prompt
            conversation_context = ""
            for m in history[-6:]:
                role = "User" if m["role"] == "user" else "Assistant"
                conversation_context += f"{role}: {m['content']}\n\n"
            full_prompt = SYSTEM_INSTRUCTION + "\n\n" + conversation_context + "User: " + question + "\n\nAssistant:"
            url = f"https://generativelanguage.googleapis.com/v1beta/{MODEL_NAME}:generateContent?key={api_key}"
            payload = {
                "contents": [{"parts": [{"text": full_prompt}]}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 400}
            }
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            elif resp.status_code == 403:
                return "🔑 API key invalid or restricted. Check your API key and permissions."
            elif resp.status_code == 404:
                return "❌ Model not found or API endpoint inaccessible. Verify MODEL_NAME & API key."
            else:
                return f"❌ API Error {resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            return f"⚠️ REST call failed: {str(e)} — showing fallback answer.\n\n" + fallback_reply(question)

    # No API key or SDK: provide local rule-based fallback
    return fallback_reply(question)

def fallback_reply(q):
    ql = q.lower()
    if "what" in ql and "glaucoma" in ql:
        return ("Glaucoma is a group of eye diseases that damage the optic nerve, often associated with "
                "raised intraocular pressure, and can lead to vision loss if untreated. (Educational only.)")
    if "rnflt" in ql or "retinal nerve" in ql:
        return ("RNFLT = Retinal Nerve Fiber Layer Thickness; measured by OCT and used to detect "
                "thinning associated with glaucoma. (Educational only.)")
    if "b-scan" in ql or "bscan" in ql or "grad-cam" in ql:
        return ("B-scan refers to cross-sectional OCT slices; Grad-CAM highlights regions that influenced a CNN prediction.")
    return ("Sorry — I can only answer glaucoma-specific questions in this demo. Try asking about RNFLT, B-scan, or glaucoma basics.")

# -----------------------
# Sidebar: API status + instructions
# -----------------------
with st.sidebar:
    st.markdown("<div class='chat-header'>🔑 Gemini API Status</div>", unsafe_allow_html=True)
    if GEMINI_API_KEY:
        st.success("✅ Gemini API key found")
        st.info("Key loaded from Streamlit secrets or env var")
    else:
        st.error("❌ No Gemini API key configured")
        st.write("Chatbot will use a safe local fallback if no key is present.")
    st.markdown("---")
    st.write("To enable Gemini: add `GEMINI_API_KEY` to Streamlit secrets or the environment.\nGet a key at https://aistudio.google.com/apikey")

# -----------------------
# Load model artifacts (cached)
# -----------------------
@st.cache_resource
def load_models_and_artifacts():
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

b_model, scaler, kmeans, avg_healthy, avg_glaucoma, thin_cluster = load_models_and_artifacts()

# -----------------------
# Helpers for RNFLT / B-scan
# -----------------------
def process_npz_file(f):
    try:
        buf = io.BytesIO(f.getvalue())
        arrs = np.load(buf, allow_pickle=True)
        key = "volume" if "volume" in arrs else arrs.files[0]
        arr = arrs[key]
        if arr.ndim == 3:
            arr = arr[0, :, :]
        vals = arr.flatten().astype(float)
        metrics = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals)),
                   "min": float(np.nanmin(vals)), "max": float(np.nanmax(vals))}
        return arr, metrics
    except Exception as e:
        st.error(f"Could not read .npz: {e}")
        return None, None

def compute_risk_map_local(rnflt_map, avg_map, threshold=-10):
    if rnflt_map.shape != avg_map.shape:
        avg_map = cv2.resize(avg_map, (rnflt_map.shape[1], rnflt_map.shape[0]))
    diff = rnflt_map - avg_map
    risk = np.where(diff < threshold, diff, np.nan)
    total = np.isfinite(diff).sum()
    risky = np.isfinite(risk).sum()
    severity = (risky / total) * 100 if total > 0 else 0.0
    return diff, risk, severity

def preprocess_bscan_image(image_pil, size=(224,224)):
    arr = np.array(image_pil.convert('L'))
    arr = np.clip(arr, 0, np.percentile(arr, 99))
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    arr_res = cv2.resize(arr, size, interpolation=cv2.INTER_NEAREST)
    arr_rgb = np.repeat(arr_res[..., None], 3, axis=-1)
    batch = np.expand_dims(arr_rgb, axis=0).astype(np.float32)
    return batch, arr_res

def gradcam_local(batch, model):
    try:
        last_conv = None
        for layer in reversed(model.layers):
            from tensorflow.keras.layers import Conv2D, DepthwiseConv2D
            if isinstance(layer, (Conv2D, DepthwiseConv2D)):
                last_conv = layer.name
                break
        if last_conv is None:
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
# Header + upload UI
# -----------------------
st.markdown("""
<div class="header">
  <h1>👁️ OCULAIRE</h1>
  <h3>AI-Powered Glaucoma Detection Dashboard — Neon Lab v5</h3>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

left_col, right_col = st.columns([3,1])
with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🩺 RNFLT Map Analysis (.npz)")
    rnflt_file = st.file_uploader("Upload RNFLT .npz", type=["npz"], label_visibility="visible")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card" style="margin-top:12px">', unsafe_allow_html=True)
    st.subheader("👁️ B-scan Slice Analysis (image)")
    bscan_file = st.file_uploader("Upload B-scan image (jpg/png)", type=["jpg","png","jpeg"], label_visibility="visible")
    st.markdown("</div>", unsafe_allow_html=True)
    threshold = st.slider("Thin-zone threshold (µm)", min_value=5, max_value=50, value=10)

with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<div class='metric-label'>Overview</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if scaler is None:
        st.markdown("<div class='metric-label'>RNFLT artifacts: <b style='color:#ff8a8a'>missing</b></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='metric-label'>RNFLT artifacts: <b style='color:#8affd6'>loaded</b></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# Analysis logic (RNFLT + B-scan + downloads)
# -----------------------
figs_for_report = []
severity_overall = 0.0

if rnflt_file is not None:
    rnflt_map, metrics = process_npz_file(rnflt_file)
    if rnflt_map is not None and avg_healthy is not None and scaler is not None:
        X = np.array([[metrics["mean"], metrics["std"], metrics["min"], metrics["max"]]])
        Xs = scaler.transform(X)
        cluster = int(kmeans.predict(Xs)[0]) if kmeans is not None else -1
        label_r = "Glaucoma-like" if cluster == thin_cluster else "Healthy-like"
        diff, risk, sev = compute_risk_map_local(rnflt_map, avg_healthy, threshold=-threshold)
        severity_overall = max(severity_overall, sev)

        c1, c2, c3, c4 = st.columns([2,2,2,1])
        c1.markdown(f"<div class='metric-label'>Status</div><div class='large-metric'>{'🚨 ' if 'Glaucoma' in label_r else '✅ '}{label_r}</div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-label'>Mean RNFLT (µm)</div><div class='large-metric'>{metrics['mean']:.2f}</div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-label'>Std Dev</div><div class='large-metric'>{metrics['std']:.2f}</div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-label'>Cluster</div><div class='large-metric'>{cluster}</div>", unsafe_allow_html=True)

        st.markdown(render_severity_html(sev), unsafe_allow_html=True)

        fig, axes = plt.subplots(1,3,figsize=(15,5), constrained_layout=True)
        im0 = axes[0].imshow(rnflt_map, cmap='turbo'); axes[0].set_title("Uploaded RNFLT"); axes[0].axis('off')
        im1 = axes[1].imshow(diff, cmap='bwr', vmin=-30, vmax=30); axes[1].set_title("Difference (vs Healthy)"); axes[1].axis('off')
        im2 = axes[2].imshow(risk, cmap='hot'); axes[2].set_title("Risk Map (thinner zones)"); axes[2].axis('off')
        for ax in axes: ax.set_facecolor("#050612")
        fig.patch.set_facecolor("#050612")
        st.pyplot(fig)
        figs_for_report.append(fig)

if bscan_file is not None and b_model is not None:
    image_pil = Image.open(bscan_file).convert("L")
    batch, proc = preprocess_bscan_image(image_pil)
    try:
        pred_raw = float(b_model.predict(batch, verbose=0)[0][0])
    except Exception:
        pred_raw = 0.0
    label_b = "Glaucoma-like" if pred_raw > 0.5 else "Healthy-like"
    conf = pred_raw*100 if label_b == "Glaucoma-like" else (1 - pred_raw)*100
    severity_overall = max(severity_overall, conf)

    st.markdown("<hr>", unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    m1.markdown(f"<div class='metric-label'>CNN Prediction</div><div class='large-metric'>{'🚨' if 'Glaucoma' in label_b else '✅'} {label_b}</div>", unsafe_allow_html=True)
    m2.markdown(f"<div class='metric-label'>Confidence</div><div class='large-metric'>{conf:.2f}%</div>", unsafe_allow_html=True)
    st.markdown(render_severity_html(conf), unsafe_allow_html=True)

    heat = gradcam_local(batch, b_model)
    if heat is not None:
        heat_r = cv2.resize(heat, (224,224))
        hm = (heat_r * 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        overlay = (np.stack([proc]*3, axis=-1) * 255).astype(np.uint8)
        overlay = cv2.addWeighted(overlay, 0.6, hm_color, 0.4, 0)
        st.image([image_pil, overlay], caption=["Original B-scan", "Grad-CAM Overlay"], use_column_width=True)
        fig2, ax2 = plt.subplots(1,2,figsize=(8,4)); ax2[0].imshow(image_pil, cmap='gray'); ax2[0].axis('off'); ax2[0].set_title("Original")
        ax2[1].imshow(overlay); ax2[1].axis('off'); ax2[1].set_title("Grad-CAM Overlay")
        fig2.patch.set_facecolor("#050612")
        figs_for_report.append(fig2)

if (rnflt_file is not None) or (bscan_file is not None):
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center'>Overall Severity Index</h4>", unsafe_allow_html=True)
    st.markdown(render_severity_html(severity_overall), unsafe_allow_html=True)
    if figs_for_report:
        png_bytes = fig_to_png_bytes(figs_for_report[0])
        pdf_bytes = create_pdf_bytes(figs_for_report)
        st.markdown('<div class="download-btns">', unsafe_allow_html=True)
        st.download_button("📸 Download PNG", data=png_bytes, file_name="oculaire_preview.png", mime="image/png")
        st.download_button("📄 Download Report (PDF)", data=pdf_bytes, file_name="oculaire_report.pdf", mime="application/pdf")
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:var(--muted);padding:8px;'>OCULAIRE Neon Lab v5 — For research/demo use only</div>", unsafe_allow_html=True)

# -----------------------
# Floating chat bubble + hidden toggle button (robust)
# -----------------------
# Unique label for the hidden toggle button (JS will search for this exact text)
_TOGGLE_BUTTON_LABEL = "__STREAMLIT_TOGGLE_CHAT__"

# Hidden wrapper (keeps the Streamlit button in DOM but visually hidden)
st.markdown('<div class="__oculaire_hidden_button_wrapper">', unsafe_allow_html=True)
toggle_pressed = st.button(_TOGGLE_BUTTON_LABEL, key="__oculaire_toggle_btn__")
st.markdown('</div>', unsafe_allow_html=True)

# When that button is clicked (by JS), toggle the server-side state
if toggle_pressed:
    st.session_state["chat_open"] = not st.session_state["chat_open"]
    # immediately rerun so the chat sidebar/expander updates
    st.experimental_rerun()

# Render the visible floating bubble/pill
st.markdown(
    """
    <div id="oculaireChatBubble">
      <div id="oculaireBubble">🤖</div>
      <div id="oculairePill">Ask Assistant</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# JavaScript: find the unique Streamlit button and click it when bubble/pill clicked
st.markdown(
    f"""
    <script>
    (function() {{
      function clickToggle() {{
        const target = "{_TOGGLE_BUTTON_LABEL}";
        // Search current document buttons
        const btns = Array.from(document.querySelectorAll('button'));
        for (let b of btns) {{
          if ((b.innerText || "").trim() === target) {{
            b.click();
            return true;
          }}
        }}
        // If not found try parent (iframe cases)
        try {{
          if (window.parent && window.parent.document) {{
            const pbtns = Array.from(window.parent.document.querySelectorAll('button'));
            for (let b of pbtns) {{
              if ((b.innerText || "").trim() === target) {{
                b.click();
                return true;
              }}
            }}
          }}
        }} catch(e) {{ /* ignore cross-origin */ }}
        console.warn("Toggle button not found for label:", target);
        return false;
      }}

      const bubble = document.getElementById('oculaireBubble');
      const pill = document.getElementById('oculairePill');
      [bubble, pill].forEach(el => {{
        if (!el) return;
        el.addEventListener('click', function(e) {{
          e.preventDefault();
          el.style.transform = 'scale(0.98)';
          setTimeout(()=> el.style.transform = '', 120);
          clickToggle();
        }});
      }});
    }})();
    </script>
    """,
    unsafe_allow_html=True
)

# -----------------------
# Render chat UI (sidebar) when open
# -----------------------
if st.session_state.get("chat_open", False):
    with st.sidebar:
        st.markdown("---")
        st.markdown("<div class='chat-header'>🤖 Glaucoma Assistant</div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;color:var(--muted);margin-bottom:8px;'>Ask about glaucoma, OCT, RNFLT or B-scan.</p>", unsafe_allow_html=True)

        # history area
        for msg in st.session_state["chat_history"][-40:]:
            if msg["role"] == "user":
                st.markdown(f"<div class='user-msg'><strong>You:</strong> {msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='assistant-msg'><strong>🤖 Assistant:</strong> {msg['content']}</div>", unsafe_allow_html=True)

        st.text_input("Ask the assistant (glaucoma-only)", key="chat_input", placeholder="What is RNFLT?")

        col1, col2, col3 = st.columns([3,1,1])
        with col1:
            if st.button("📤 Send", use_container_width=True):
                q = st.session_state.get("chat_input", "").strip()
                if not q:
                    st.warning("Please enter a question.")
                else:
                    # call assistant
                    reply = ask_glaucoma_assistant(q, st.session_state["chat_history"], GEMINI_API_KEY)
                    st.session_state["chat_history"].append({"role": "user", "content": q})
                    st.session_state["chat_history"].append({"role": "assistant", "content": reply})
                    # clear input
                    st.session_state["chat_input"] = ""
                    st.experimental_rerun()
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state["chat_history"] = []
                st.experimental_rerun()
        with col3:
            if st.button("✖", use_container_width=True):
                st.session_state["chat_open"] = False
                st.experimental_rerun()

# End of app
