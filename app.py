# app.py — OCULAIRE Neon Lab v5 with Floating Glaucoma Chat Assistant
import os
import io
import joblib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import cv2
from matplotlib.backends.backend_pdf import PdfPages
import streamlit as st

# Optional SDK import (safe)
try:
    import google.generativeai as genai
    HAVE_GENAI_SDK = True
except Exception:
    genai = None
    HAVE_GENAI_SDK = False

# -----------------------
# Settings - change MODEL_NAME if you want a different model
# -----------------------
MODEL_NAME = "models/gemini-2.5-pro"   # <--- change here if needed
# API key is read securely: st.secrets['GEMINI_API_KEY'] or environment var GEMINI_API_KEY
def get_gemini_api_key():
    try:
        return st.secrets.get("GEMINI_API_KEY") or st.secrets.get("gemini", {}).get("api_key")
    except Exception:
        return os.getenv("GEMINI_API_KEY") or os.getenv("GENAI_API_KEY") or None

GEMINI_API_KEY = get_gemini_api_key()

# -----------------------
# Page config & plotting defaults
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
# CSS: neon theme + floating chat bubble
# -----------------------
st.markdown("""
<style>
:root { --bg:#020208; --panel:#0a0f25; --neonA:#00f5ff; --neonB:#ff40c4; --muted:#a4b1c9; }
.stApp { background: radial-gradient(circle at 20% 20%, #091133, #020208 90%); color: #e6faff; font-family: Inter, system-ui; }

/* header */
.header { text-align:center; margin-top:8px; margin-bottom:6px; }
.header h1 { font-size:42px; font-weight:900; letter-spacing:3px;
  background: linear-gradient(90deg, var(--neonA), var(--neonB));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; text-shadow: 0 0 20px rgba(0,245,255,0.7), 0 0 35px rgba(255,64,196,0.4); }
.header h3 { color:var(--muted); font-weight:400; font-size:14px; }

/* cards + metrics */
.card { background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)); border:1px solid rgba(255,255,255,0.04);
  box-shadow: 0 0 25px rgba(0,245,255,0.04); border-radius:12px; padding:14px; }
.metric-label { color:var(--muted); font-size:12px; }
.large-metric { font-weight:800; font-size:22px; color:#fff; text-shadow:0 0 15px rgba(0,245,255,0.35); }

/* severity bar */
.sev-wrap { margin-top:12px; }
.sev-outer { height:18px; width:100%; background: rgba(255,255,255,0.04); border-radius:14px; overflow:hidden; }
.sev-inner { height:100%; width:0%; background: linear-gradient(90deg,var(--neonA),var(--neonB)); border-radius:14px;
  box-shadow: 0 0 25px rgba(0,245,255,0.55); transition: width 1s cubic-bezier(.2,.9,.2,1); }
.sev-chip { margin-top:8px; display:inline-block; padding:6px 12px; border-radius:12px; font-weight:800; font-size:14px; color:#021617;
  background: linear-gradient(90deg, rgba(0,245,255,0.95), rgba(255,64,196,0.95)); box-shadow: 0 0 20px rgba(0,245,255,0.3); animation: pulse 1.6s infinite; }
@keyframes pulse { 0%{transform:scale(1);} 50%{transform:scale(1.05);} 100%{transform:scale(1);} }

/* downloads */
.download-btns { margin-top:10px; display:flex; gap:10px; justify-content:flex-start; }

/* Chat bubble (floating bottom-right) */
.chat-container { position: fixed; right: 22px; bottom: 22px; z-index: 9999; display:flex; flex-direction:column; align-items:flex-end; gap:10px; }
.chat-bubble { width:66px; height:66px; border-radius:50%; display:flex; align-items:center; justify-content:center; cursor:pointer;
  background: linear-gradient(135deg, var(--neonA), var(--neonB)); color:#021617; font-size:30px; border:2px solid rgba(255,255,255,0.06);
  box-shadow: 0 12px 40px rgba(0,245,255,0.12); }
.chat-window { width:380px; height:480px; background:var(--panel); border-radius:12px; box-shadow: 0 18px 50px rgba(0,0,0,0.7); overflow:hidden; display:flex; flex-direction:column; }
.chat-header { padding:10px 12px; background: linear-gradient(90deg,#091133,var(--panel)); color:var(--neonA); font-weight:700; display:flex; justify-content:space-between; align-items:center; }
.chat-history { padding:10px; flex:1; overflow-y:auto; background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(0,0,0,0.01)); }
.user-msg { text-align:right; margin:8px 0; padding:10px; background: rgba(0,245,255,0.07); border-radius:10px; display:inline-block; max-width:86%; }
.ai-msg { text-align:left; margin:8px 0; padding:10px; background: rgba(255,64,196,0.06); border-radius:10px; display:inline-block; max-width:86%; }
.chat-input-row { padding:10px; border-top:1px solid rgba(255,255,255,0.03); display:flex; gap:8px; align-items:center; }
.input-field { flex:1; padding:8px 10px; border-radius:8px; background:#06090f; color:#e6faff; border:1px solid rgba(255,255,255,0.03); }
.send-btn { background: linear-gradient(90deg,var(--neonA),var(--neonB)); border:none; padding:8px 12px; border-radius:8px; cursor:pointer; color:#021617; font-weight:700; }

footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Header
# -----------------------
st.markdown("""
<div class="header">
  <h1>👁️ OCULAIRE</h1>
  <h3>AI-Powered Glaucoma Detection Dashboard — Neon Lab v5</h3>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# -----------------------
# helper functions — plots, files, severity UI
# -----------------------
def render_severity_html(pct: float) -> str:
    pct = float(max(0.0, min(100.0, pct)))
    html = f"""
    <div class='sev-wrap'>
      <div class='sev-outer'><div class='sev-inner' id='sev_inner' style='width:0%'></div></div>
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

def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.getvalue()

def create_pdf_bytes(figs):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        for f in figs:
            pdf.savefig(f, bbox_inches='tight', facecolor=f.get_facecolor())
    buf.seek(0)
    return buf.getvalue()

# -----------------------
# Models & artifacts loader (cached)
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
# Simple RNFLT / B-scan helpers
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

# -----------------------
# Chat logic: safe fallback + SDK usage (wrapped)
# -----------------------
if "chat_visible" not in st.session_state:
    st.session_state.chat_visible = False
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [{"role":"assistant","content":"Hello — I'm the Glaucoma Assistant. Ask me glaucoma-specific questions."}]
if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

def toggle_chat():
    st.session_state.chat_visible = not st.session_state.chat_visible

def local_rule_based_reply(user_q: str) -> str:
    low = user_q.lower()
    if "what" in low and "glaucoma" in low:
        return ("Glaucoma is a group of eye conditions that damage the optic nerve, "
                "often associated with raised intraocular pressure. This can lead to "
                "progressive vision loss if untreated. Consult an ophthalmologist for diagnosis and care.")
    if "rnflt" in low or "retinal nerve" in low:
        return ("RNFLT = Retinal Nerve Fiber Layer Thickness — measured by OCT. Thinning of RNFLT indicates "
                "possible glaucomatous damage.")
    if "b-scan" in low or "bscan" in low:
        return ("B-scan here refers to OCT cross-sectional images (B-scans). CNNs can classify slices and Grad-CAM "
                "helps visualize which regions influenced the decision.")
    return ("Sorry — in this demo I focus on glaucoma topics (RNFLT, OCT, B-scan). "
            "You can ask about those or consult a clinician for other health questions.")

def ask_genie_via_sdk(prompt: str) -> str:
    """Call SDK; wrapped safely. Returns text or error message."""
    try:
        if not HAVE_GENAI_SDK:
            return "⚠️ SDK not installed; falling back to local responder."
        if not GEMINI_API_KEY:
            return "⚠️ No Gemini API key configured; falling back to local responder."
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)
        # simple generate_content usage; some SDKs require different call forms — handle errors
        resp = model.generate_content(prompt)
        # resp may have attribute text or content -> try common options
        text = getattr(resp, "text", None)
        if text:
            return text
        # fallback to structured json content
        try:
            return resp.output[0].content[0].text
        except Exception:
            return str(resp)
    except Exception as e:
        return f"⚠️ Gemini SDK call error: {e}"

def ask_chatbot(user_q: str):
    st.session_state.chat_messages.append({"role":"user","content":user_q})
    # Compose instruction
    system_instruction = (
        "You are a helpful and concise expert assistant specialized ONLY in glaucoma, OCT imaging, RNFLT, B-scan interpretation, and related ophthalmic basics. "
        "If the user asks about unrelated topics, politely state you only answer glaucoma-specific questions. Keep answers short (<200 words) and include a short disclaimer."
    )
    prompt = system_instruction + "\n\nUser: " + user_q + "\n\nAssistant:"
    # prefer SDK if present
    if HAVE_GENAI_SDK and GEMINI_API_KEY:
        reply = ask_genie_via_sdk(prompt)
        if reply and not reply.startswith("⚠️"):
            st.session_state.chat_messages.append({"role":"assistant","content":reply + "\n\n_Disclaimer: educational only — not medical advice._"})
            return
    # fallback local
    reply = local_rule_based_reply(user_q)
    st.session_state.chat_messages.append({"role":"assistant","content":reply + "\n\n_Disclaimer: educational only — not medical advice._"})

# -----------------------
# Sidebar: API status & quick help
# -----------------------
with st.sidebar:
    st.markdown("<div style='font-weight:700'>🔑 Gemini API</div>", unsafe_allow_html=True)
    if GEMINI_API_KEY:
        st.success("API key configured")
        st.markdown(f"<div style='font-size:12px;color:var(--muted)'>Using key from secrets/environment. Model: <b>{MODEL_NAME}</b></div>", unsafe_allow_html=True)
    else:
        st.error("No Gemini API key found")
        st.markdown("<div style='font-size:12px;color:var(--muted)'>Chat will use local fallback replies. Set GEMINI_API_KEY as a secret or environment variable to enable cloud responses.</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='font-size:12px;color:var(--muted)'>Put your deployment artifacts (models / joblib / npy) next to app.py. Example: bscan_cnn.h5, rnflt_scaler.joblib, avg_map_healthy.npy</div>", unsafe_allow_html=True)

# -----------------------
# Main upload UI (left / right)
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
    if scaler is None:
        st.markdown("<div class='metric-label'>RNFLT artifacts: <b style='color:#ff8a8a'>missing</b></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='metric-label'>RNFLT artifacts: <b style='color:#8affd6'>loaded</b></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# Analysis and plotting
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
        fig.colorbar(im0, ax=axes[0], fraction=0.05)
        im1 = axes[1].imshow(diff, cmap='bwr', vmin=-30, vmax=30); axes[1].set_title("Difference (vs Healthy)"); axes[1].axis('off')
        fig.colorbar(im1, ax=axes[1], fraction=0.05)
        im2 = axes[2].imshow(risk, cmap='hot'); axes[2].set_title("Risk Map (thinner zones)"); axes[2].axis('off')
        fig.colorbar(im2, ax=axes[2], fraction=0.05)
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
        fig2, ax2 = plt.subplots(1,2,figsize=(8,4)); ax2[0].imshow(image_pil, cmap='gray'); ax2[0].axis('off'); ax2[0].set_title("Original"); ax2[1].imshow(overlay); ax2[1].axis('off'); ax2[1].set_title("Grad-CAM Overlay")
        fig2.patch.set_facecolor("#050612")
        figs_for_report.append(fig2)

# Combined severity + downloads
if rnflt_file or bscan_file:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center'>Overall Severity Index</h4>", unsafe_allow_html=True)
    st.markdown(render_severity_html(severity_overall), unsafe_allow_html=True)
    if figs_for_report:
        png = fig_to_png_bytes(figs_for_report[0])
        pdfb = create_pdf_bytes(figs_for_report)
        st.markdown('<div class="download-btns">', unsafe_allow_html=True)
        st.download_button("📸 Download PNG", data=png, file_name="oculaire_rnflt.png", mime="image/png")
        st.download_button("📄 Download Full Report (PDF)", data=pdfb, file_name="oculaire_report.pdf", mime="application/pdf")
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:var(--muted);padding:6px;'>OCULAIRE Neon Lab v5 — For research/demo use only</div>", unsafe_allow_html=True)

# -----------------------
# Floating Chat bubble HTML + toggle button (bottom-right)
# -----------------------
# Hidden Streamlit button toggles state reliably.
st.markdown("""
<div class="chat-container">
  <div style="display:flex;flex-direction:column;align-items:flex-end;gap:8px;">
    <div class="chat-bubble" onclick="document.getElementById('toggle_chat_btn').click();">🤖</div>
  </div>
</div>
""", unsafe_allow_html=True)
# Visible Streamlit button but we hide with CSS visually (so toggle works)
st.button("toggle_chat", key="toggle_chat_btn", on_click=toggle_chat, help="Toggle chat window (hidden)")

# If chat visible, render chat window anchored in same corner
if st.session_state.chat_visible:
    # Render the chat window HTML
    st.markdown("""
    <div class="chat-container" style="pointer-events:auto;">
      <div class="chat-window">
        <div class="chat-header">
          Glaucoma Assistant
          <button style="background:none;border:none;color:var(--neonA);font-weight:bold;cursor:pointer" onclick="document.getElementById('toggle_chat_btn').click();">✖</button>
        </div>
        <div class="chat-history" id="chat_history_div">
    """, unsafe_allow_html=True)

    # Show messages
    for msg in st.session_state.chat_messages[-40:]:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-msg'><strong>You:</strong> {st.experimental_escape_dialogue if False else msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='ai-msg'><strong>Assistant:</strong> {msg['content']}</div>", unsafe_allow_html=True)

    # Input row
    # Use form to avoid immediate rerun; user types and hits Send
    with st.form(key="chat_form", clear_on_submit=False):
        user_input = st.text_input("Type your question (glaucoma-only)", key="chat_input_field", placeholder="e.g., What is RNFLT? How is OCT used?")
        submitted = st.form_submit_button("Send")
        if submitted and user_input:
            ask_chatbot(user_input)
            # re-render: we don't call st.experimental_rerun to avoid losing scroll, but this block will re-run on next interaction.

    st.markdown("</div></div></div>", unsafe_allow_html=True)

# End of file
