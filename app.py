# app.py - OCULAIRE Neon Lab v5 (complete, safe chatbot fallback, downloads, neon UI)
import os
import io
import time
import base64
import joblib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import cv2
from matplotlib.backends.backend_pdf import PdfPages
import streamlit as st

# -----------------------
# Safe optional import for Gemini
# -----------------------
genai = None
# --- SECURE METHOD: Prioritize Streamlit Secrets ---
# Use st.secrets to securely load the key from the [gemini] section
try:
    if 'gemini' in st.secrets and 'api_key' in st.secrets['gemini']:
        GEMINI_API_KEY = st.secrets['gemini']['api_key']
    else:
        # Fallback to environment variables if secrets are not set
        GEMINI_API_KEY = os.environ.get("GENAI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or None
except Exception:
    # Final safe fallback if st.secrets itself fails to load (e.g., local development without secrets.toml)
    GEMINI_API_KEY = os.environ.get("GENAI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or None

try:
    import google.generativeai as genai  # optional
except Exception:
    genai = None

# -----------------------
# Page config & plotting
# -----------------------
st.set_page_config(page_title="OCULAIRE: Neon Glaucoma Detection Dashboard",
                   layout="wide", page_icon="👁️")

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
# CSS: neon theme + floating chatbot bubble
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
.stApp { background: radial-gradient(circle at 20% 20%, #091133, #020208 90%); color: #e6faff; font-family: 'Plus Jakarta Sans', Inter, system-ui; }

/* Header */
.header { text-align:center; margin-top:8px; margin-bottom:6px; }
.header h1 { font-size:42px; font-weight:900; letter-spacing:3px;
  background: linear-gradient(90deg, var(--neonA), var(--neonB));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  text-shadow: 0 0 20px rgba(0,245,255,0.7), 0 0 35px rgba(255,64,196,0.4);
}
.header h3 { color:var(--muted); font-weight:400; font-size:14px; }

/* Cards */
.card { background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
  border:1px solid rgba(255,255,255,0.04); box-shadow: 0 0 25px rgba(0,245,255,0.04);
  border-radius:12px; padding:14px; }
.uploader-card { background:#0d1720; padding:12px; border-radius:8px; border:1px solid rgba(255,255,255,0.03); }

.metric-label { color:var(--muted); font-size:12px; }
.large-metric { font-weight:800; font-size:22px; color:#fff; text-shadow:0 0 15px rgba(0,245,255,0.35); }

/* Severity bar */
.sev-wrap { margin-top:12px; }
.sev-outer { height:18px; width:100%; background: rgba(255,255,255,0.04); border-radius:14px; overflow:hidden; }
.sev-inner { height:100%; width:0%; background: linear-gradient(90deg,var(--neonA),var(--neonB));
  border-radius:14px; box-shadow: 0 0 25px rgba(0,245,255,0.55); transition: width 1s cubic-bezier(.2,.9,.2,1); }
.sev-chip { margin-top:8px; display:inline-block; padding:6px 12px; border-radius:12px; font-weight:800; font-size:14px; color:#021617;
  background: linear-gradient(90deg, rgba(0,245,255,0.95), rgba(255,64,196,0.95)); box-shadow: 0 0 20px rgba(0,245,255,0.3); animation: pulse 1.6s infinite; }
@keyframes pulse { 0%{transform:scale(1);} 50%{transform:scale(1.05);} 100%{transform:scale(1);} }

/* download buttons */
.download-btns { margin-top:10px; display:flex; gap:10px; justify-content:flex-start; }

/* Floating Chatbot */
.chat-bubble-wrapper { position: fixed; bottom: 20px; left: 20px; z-index:999; display:flex; flex-direction:column; align-items:flex-start; gap:8px; }
.chat-bubble { width:64px; height:64px; border-radius:50%; display:flex; align-items:center; justify-content:center; cursor:pointer;
  background: linear-gradient(135deg, var(--neonA), var(--neonB)); color:#021617; box-shadow: 0 8px 30px rgba(0,245,255,0.18); font-size:28px; border: 2px solid rgba(255,255,255,0.06);}
.chat-window { width:360px; height:460px; background:var(--panel); border-radius:12px; padding:0; box-shadow:0 12px 40px rgba(0,0,0,0.7);}
.chat-header { padding:10px; background: linear-gradient(90deg,#091133,var(--panel)); color:var(--neonA); font-weight:700; border-radius:12px 12px 0 0; display:flex; justify-content:space-between; align-items:center; }
.chat-history { padding:10px; height:340px; overflow-y:auto; background:rgba(0,0,0,0.02); }
.user-message { text-align:right; margin:6px 0; padding:8px 10px; background: rgba(0,245,255,0.08); border-radius:10px; color:#e6faff; display:inline-block; max-width:85%; }
.ai-message { text-align:left; margin:6px 0; padding:8px 10px; background: rgba(255,64,196,0.06); border-radius:10px; color:#e6faff; display:inline-block; max-width:85%; }
.chat-input-area { padding:8px; border-top:1px solid rgba(255,255,255,0.03); display:flex; gap:6px; align-items:center; }
.chat-input { flex:1; padding:8px 10px; border-radius:8px; background:#06090f; color:#e6faff; border:1px solid rgba(255,255,255,0.03); }

footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Header UI
# -----------------------
st.markdown("""
<div class="header">
  <h1>👁️ OCULAIRE</h1>
  <h3>AI-Powered Glaucoma Detection Dashboard — Neon Lab v5</h3>
</div>
""", unsafe_allow_html=True)

# -----------------------
# Load models & artifacts (cached)
# -----------------------
@st.cache_resource
def load_models_and_artifacts():
    # B-scan CNN (optional)
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
# Helper functions
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
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
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

def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.getvalue()

def create_pdf_bytes(figs):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        for f in figs:
            pdf.savefig(f, bbox_inches='tight', facecolor=f.get_facecolor())
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
      setTimeout(function(){{
        var el = document.getElementById('sev_inner');
        if(el) el.style.width = '{pct:.1f}%';
      }}, 120);
    </script>
    """
    return html

# -----------------------
# Chatbot safe initialization & utilities
# -----------------------
DOMAIN = "Glaucoma"
SYSTEM_INSTRUCTION_PROMPT = (
    f"You are an expert assistant specialized in {DOMAIN}. "
    "Only answer questions related to this domain. If a question is outside the domain, politely decline."
)

if "chat_visible" not in st.session_state:
    st.session_state.chat_visible = False
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [{"role":"ai","content":f"Hello — I'm the {DOMAIN} Assistant. Ask me glaucoma-specific questions."}]
if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

# Configure genai if available and key present (do not error if not)
if genai is not None and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # create a model handle (but do not start chat now)
        st.session_state.genai_model = genai.GenerativeModel("models/gemini-2.5-pro")
        st.session_state.genai_available = True
    except Exception:
        st.session_state.genai_available = False
else:
    st.session_state.genai_available = False

def toggle_chat():
    st.session_state.chat_visible = not st.session_state.chat_visible

def submit_chat():
    user_q = st.session_state.chat_input.strip()
    if not user_q:
        return
    # append user message
    st.session_state.chat_messages.append({"role":"user","content":user_q})
    # local fallback responder
    if not st.session_state.get("genai_available", False):
        # small rule-based / safe reply
        reply = ""
        low = user_q.lower()
        if "what" in low and "glaucoma" in low:
            reply = "Glaucoma is a group of eye diseases that damage the optic nerve — often associated with raised intraocular pressure — and can lead to vision loss if untreated."
        elif "rnflt" in low or "retinal nerve" in low:
            reply = "RNFLT stands for Retinal Nerve Fiber Layer Thickness, often measured by OCT and used to detect thinning associated with glaucoma."
        elif "b-scan" in low or "bscan" in low:
            reply = "B-scan here refers to OCT cross-sectional images (B-scans) used in CNN models for classification and Grad-CAM interpretability."
        else:
            reply = "Sorry — I can only answer glaucoma-specific questions in this demo. Try asking about RNFLT, B-scan, or glaucoma basics."
        st.session_state.chat_messages.append({"role":"ai","content":reply})
        st.session_state.chat_input = ""
        return
    # If genai available, do a safe call
    try:
        model = st.session_state.genai_model
        # Use the model's simple generate call (API details may vary by version)
        # Create content with system instruction + user
        prompt = SYSTEM_INSTRUCTION_PROMPT + "\n\nUser: " + user_q
        resp = model.generate_content(prompt)
        ai_text = getattr(resp, "text", str(resp))
    except Exception as e:
        ai_text = "⚠️ AI service error or rate limit. Showing fallback message: I can only answer glaucoma-specific queries."
    st.session_state.chat_messages.append({"role":"ai","content":ai_text})
    st.session_state.chat_input = ""

# -----------------------
# Upload UI (main)
# -----------------------
left_col, right_col = st.columns([3, 1])
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
# Analysis logic
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

        # Metrics row
        c1, c2, c3, c4 = st.columns([2,2,2,1])
        c1.markdown(f"<div class='metric-label'>Status</div><div class='large-metric'>{'🚨 ' if 'Glaucoma' in label_r else '✅ '}{label_r}</div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-label'>Mean RNFLT (µm)</div><div class='large-metric'>{metrics['mean']:.2f}</div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-label'>Std Dev</div><div class='large-metric'>{metrics['std']:.2f}</div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-label'>Cluster</div><div class='large-metric'>{cluster}</div>", unsafe_allow_html=True)

        # Severity bar
        st.markdown(render_severity_html(sev), unsafe_allow_html=True)

        # Single figure (1x3) — shown once
        fig, axes = plt.subplots(1,3,figsize=(15,5), constrained_layout=True)
        im0 = axes[0].imshow(rnflt_map, cmap='turbo'); axes[0].set_title("Uploaded RNFLT")
        axes[0].axis('off')
        c0 = fig.colorbar(im0, ax=axes[0], fraction=0.05)
        im1 = axes[1].imshow(diff, cmap='bwr', vmin=-30, vmax=30); axes[1].set_title("Difference (vs Healthy)"); axes[1].axis('off')
        c1 = fig.colorbar(im1, ax=axes[1], fraction=0.05)
        im2 = axes[2].imshow(risk, cmap='hot'); axes[2].set_title("Risk Map (thinner zones)"); axes[2].axis('off')
        c2 = fig.colorbar(im2, ax=axes[2], fraction=0.05)
        fig.patch.set_facecolor("#050612")
        for ax in axes: ax.set_facecolor("#050612")
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
        # create a small figure for report
        fig2, ax2 = plt.subplots(1,2,figsize=(8,4)); ax2[0].imshow(image_pil, cmap='gray'); ax2[0].axis('off'); ax2[0].set_title("Original")
        ax2[1].imshow(overlay); ax2[1].axis('off'); ax2[1].set_title("Grad-CAM Overlay")
        fig2.patch.set_facecolor("#050612")
        figs_for_report.append(fig2)

# Combined severity summary + downloads
if (rnflt_file is not None) or (bscan_file is not None):
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center'>Overall Severity Index</h4>", unsafe_allow_html=True)
    st.markdown(render_severity_html(severity_overall), unsafe_allow_html=True)
    # downloads
    if figs_for_report:
        png_bytes = fig_to_png_bytes(figs_for_report[0])
        pdf_bytes = create_pdf_bytes(figs_for_report)
        st.markdown('<div class="download-btns">', unsafe_allow_html=True)
        st.download_button("📸 Download RNFLT PNG", data=png_bytes, file_name="oculaire_rnflt.png", mime="image/png")
        st.download_button("📄 Download Full Report (PDF)", data=pdf_name, mime="application/pdf")
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:var(--muted);padding:6px;'>OCULAIRE Neon Lab v5 — For research/demo use only</div>", unsafe_allow_html=True)

# -----------------------
# Floating Chatbot UI (uses streamlit button to toggle)
# -----------------------
# Place markup for bubble + a Streamlit button (styled via CSS above)
st.markdown("""
<div class="chat-bubble-wrapper">
  <div style="display:flex;gap:8px;align-items:flex-end">
    <div class="chat-bubble" id="bubble" onclick="document.getElementById('toggle_chat_btn').click();">🤖</div>
  </div>
</div>
""", unsafe_allow_html=True)

# A standard Streamlit button (visually hidden by CSS but present in DOM) toggles chat state.
# We avoid writing session_state outside callbacks by using on_click.
st.button("Toggle Chat (hidden)", key="toggle_chat_btn", on_click=toggle_chat)

# If visible, render chat window (use container anchored bottom-left by CSS)
if st.session_state.chat_visible:
    st.markdown("""
    <div class="chat-bubble-wrapper">
      <div class="chat-window">
        <div class="chat-header">Glaucoma Assistant
          <button style="background:none;border:none;color:var(--neonA);font-weight:bold;cursor:pointer" onclick="document.getElementById('toggle_chat_btn').click();">✖</button>
        </div>
        <div class="chat-history" id="chat_history_div">
    """, unsafe_allow_html=True)

    # Render chat messages
    for msg in st.session_state.chat_messages[-40:]:
        if msg["role"] == "user":
            # NOTE: Using st.markdown.__wrapped__ and '' is a common Streamlit trick
            # to render a string-formatted HTML block reliably.
            st.markdown(f"<div class='user-message'>{st.markdown.__wrapped__ and ''}{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='ai-message'>{msg['content']}</div>", unsafe_allow_html=True)

    # Input row; use on_change callback to submit (safe)
    st.text_input("Ask the assistant (glaucoma-only)", key="chat_input", on_change=submit_chat, placeholder="Ask about glaucoma, RNFLT or B-scan")
    st.markdown("</div></div></div>", unsafe_allow_html=True)

# End of app
