# app.py — OCULAIRE Neon Lab v5 with Glaucoma Chatbot (robust bubble + pill + fallback)
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
except Exception:
    import requests
    import json
    USE_SDK = False

# -----------------------
# Page Config
# -----------------------
st.set_page_config(page_title="OCULAIRE: Neon Glaucoma Detection Dashboard",
                   layout="wide", page_icon="👁️")

# -----------------------
# Session state defaults
# -----------------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chat_open' not in st.session_state:
    st.session_state.chat_open = False
if 'chat_input' not in st.session_state:
    st.session_state.chat_input = ""

# -----------------------
# Query param fallback: if ?oculaire_toggle=1 present toggle the chat
# This provides a robust fallback (works even if JS is restricted)
params = st.experimental_get_query_params()
if 'oculaire_toggle' in params:
    # toggle and remove param
    st.session_state.chat_open = not st.session_state.chat_open
    # clear params
    st.experimental_set_query_params()

# -----------------------
# Get API key from Streamlit secrets or environment variable
def get_api_key():
    try:
        return st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", None)

API_KEY = get_api_key()

# -----------------------
# Neon Matplotlib Config
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
# CSS — Neon Theme + Floating Bubble + Pill
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

/* Floating bubble */
.chat-bubble {
  position: fixed;
  bottom: 30px;
  right: 30px;
  width: 72px;
  height: 72px;
  border-radius: 50%;
  background: linear-gradient(135deg, var(--neonA), var(--neonB));
  box-shadow: 0 8px 40px rgba(0,245,255,0.18), 0 0 40px rgba(255,64,196,0.18);
  display:flex; align-items:center; justify-content:center; font-size:34px; color:#021617;
  cursor:pointer; z-index:9999;
  transition: transform .12s ease;
}
.chat-bubble:hover { transform: scale(1.06); }

/* Floating pill (text next to bubble) */
.chat-pill {
  position: fixed;
  bottom: 38px;
  right: 115px;
  background: linear-gradient(90deg, rgba(0,245,255,0.08), rgba(255,64,196,0.06));
  padding: 12px 18px;
  border-radius: 28px;
  color: #e6faff;
  font-weight: 800;
  z-index:9999;
  cursor:pointer;
  box-shadow: 0 8px 40px rgba(0,0,0,0.4);
}
.chat-pill:hover { transform: translateY(-6px); }

/* tiny visible fallback link area (useful if JS can't click hidden button) */
.chat-fallback {
  position: fixed;
  bottom: 100px;
  right: 30px;
  z-index: 9999;
  font-size:12px;
  color: #a4b1c9;
  opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)

# -----------------------
# Chat assistant logic (safe fallbacks)
MODEL_NAME = "models/gemini-2.5-pro"

def ask_glaucoma_assistant(question, history, api_key):
    """Call Google Gemini API with glaucoma-specific context or fallback to simple rule replies."""
    if not api_key:
        return "⚠️ Please configure your Gemini API key (set GEMINI_API_KEY in Streamlit secrets or environment)."

    system_instruction = (
        "You are a specialized assistant for glaucoma. Answer only glaucoma-related questions, "
        "explain terms, be concise, and include a brief educational-only disclaimer."
    )

    try:
        if USE_SDK:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(MODEL_NAME)
            chat_history = []
            for msg in history[-6:]:
                role = "user" if msg.get("role") == "user" else "model"
                chat_history.append({"role": role, "parts": [msg.get("content","")]})
            chat = model.start_chat(history=chat_history)
            resp = chat.send_message(f"{system_instruction}\n\nUser question: {question}")
            return getattr(resp, "text", str(resp))
        else:
            # fallback REST call
            conversation_context = ""
            for msg in history[-6:]:
                role = "User" if msg.get("role") == "user" else "Assistant"
                conversation_context += f"{role}: {msg.get('content','')}\n\n"
            full_prompt = f"{system_instruction}\n\n{conversation_context}User: {question}\n\nAssistant:"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
            resp = requests.post(url, headers={"Content-Type":"application/json"}, json={
                "contents":[{"parts":[{"text": full_prompt}]}],
                "generationConfig":{"temperature":0.2,"maxOutputTokens":400}
            }, timeout=30)
            if resp.status_code == 200:
                d = resp.json()
                return d["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return f"⚠️ API Error: {resp.status_code} - {resp.text[:200]}"
    except Exception as e:
        # safe local fallback reply generator
        low = question.lower()
        if "rnflt" in low or "retinal" in low:
            return "RNFLT = Retinal Nerve Fiber Layer Thickness — measured by OCT and useful to detect thinning in glaucoma. (Educational only — not medical advice.)"
        if "what is glaucoma" in low or "what is a glaucoma" in low:
            return "Glaucoma is a group of eye diseases that can damage the optic nerve, often associated with raised intraocular pressure. Early detection is important. (Educational only.)"
        return f"⚠️ Assistant error: {str(e)} — fallback reply: I can answer glaucoma-focused questions. (Educational only.)"

# -----------------------
# Hidden toggle button strategy (primary mechanism)
# We'll create a hidden Streamlit button with a unique label and key.
_TOGGLE_LABEL = "__OCULAIRE_TOGGLE_CHAT__"
_toggle_clicked = st.button(_TOGGLE_LABEL, key="__oculaire_hidden_toggle__", help="hidden toggle (do not click)")

if _toggle_clicked:
    st.session_state.chat_open = not st.session_state.chat_open
    # immediate rerun to reflect new sidebar state
    st.experimental_rerun()

# -----------------------
# Floating bubble + pill HTML (user clicks these)
st.markdown(f"""
<div class="chat-bubble" id="oculaireBubble">🤖</div>
<div class="chat-pill" id="oculairePill"><a id="oculairePillLink" href="?oculaire_toggle=1" style="color:inherit;text-decoration:none">Ask Assistant</a></div>

<!-- small hint for fallback (visually subtle) -->
<div class="chat-fallback">If the bubble doesn't open, click the pill text or refresh.</div>

<script>
(function(){
  const label = "{_TOGGLE_LABEL}";
  // This function attempts multiple heuristics to find the Streamlit button and click it
  function tryClickButton() {
    // 1) Find by exact innerText match among all buttons
    let btns = Array.from(document.querySelectorAll('button'));
    for (let b of btns) {
      try {
        if ((b.innerText || "").trim() === label) { b.click(); return true; }
        // sometimes buttons have nested spans - compare normalized text
        if ((b.textContent || "").trim() === label) { b.click(); return true; }
      } catch(e){}
    }
    // 2) Try searching parent document (iframe parent scenarios)
    try {
      if (window.parent && window.parent.document) {
        let pbtns = Array.from(window.parent.document.querySelectorAll('button'));
        for (let b of pbtns) {
          try {
            if ((b.innerText || "").trim() === label) { b.click(); return true; }
            if ((b.textContent || "").trim() === label) { b.click(); return true; }
          } catch(e){}
        }
      }
    } catch(e){}
    // 3) Try clicking any button whose aria-label or value contains our token
    for (let b of btns) {
      try {
        if ((b.getAttribute('aria-label')||'').includes('__oculaire') || (b.value||'').includes('__oculaire')) { b.click(); return true; }
      } catch(e){}
    }
    // 4) Last resort: dispatch a synthetic click event on the first button (avoid if no match)
    if (btns.length>0) {
      try { btns[0].dispatchEvent(new MouseEvent('click')); return true; } catch(e){}
    }
    return false;
  }

  const bubble = document.getElementById('oculaireBubble');
  const pill = document.getElementById('oculairePill');
  [bubble, pill].forEach(el=>{
    if(!el) return;
    el.style.cursor='pointer';
    el.addEventListener('click', function(e){
      e.preventDefault();
      // slight visual feedback
      el.style.transform='scale(0.98)';
      setTimeout(()=> el.style.transform='', 120);
      // try JS click hidden Streamlit button
      const ok = tryClickButton();
      // if not ok, the pill link already navigates to ?oculaire_toggle=1 which the app reads as fallback
      if(!ok) {
        // optional: navigate to query param fallback (same effect)
        const href = document.getElementById('oculairePillLink')?.getAttribute('href');
        if(href) { window.location.href = href; }
      }
    });
  });
})();
</script>
""", unsafe_allow_html=True)

# -----------------------
# Header UI
st.markdown("""
<div class="header">
  <h1>👁️ OCULAIRE</h1>
  <h3>AI-Powered Glaucoma Detection Dashboard — Neon Lab v5</h3>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# -----------------------
# Load Models & Artifacts (cached)
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
# Helper functions (same as your prior helpers)
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
# Upload UI
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
# Analysis & plotting (kept concise)
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

# Combined severity summary + downloads
if (rnflt_file is not None) or (bscan_file is not None):
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center'>Overall Severity Index</h4>", unsafe_allow_html=True)
    st.markdown(render_severity_html(severity_overall), unsafe_allow_html=True)
    if figs_for_report:
        png_bytes = fig_to_png_bytes(figs_for_report[0])
        pdf_bytes = create_pdf_bytes(figs_for_report)
        st.markdown('<div style="display:flex;gap:12px;margin-top:10px">', unsafe_allow_html=True)
        st.download_button("📸 Download RNFLT PNG", data=png_bytes, file_name="oculaire_rnflt.png", mime="image/png")
        st.download_button("📄 Download Full Report (PDF)", data=pdf_bytes, file_name="oculaire_report.pdf", mime="application/pdf")
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:#a4b1c9;padding:6px;'>OCULAIRE Neon Lab v5 — For research/demo use only</div>", unsafe_allow_html=True)

# -----------------------
# Sidebar chat UI when chat_open is True
if st.session_state.chat_open:
    with st.sidebar:
        st.markdown("---")
        st.markdown("<div style='font-weight:800;font-size:18px;color: #00f5ff;'>🤖 Glaucoma Assistant</div>", unsafe_allow_html=True)
        st.markdown("<p style='color:#a4b1c9;font-size:13px;margin-bottom:10px;'>Ask me about RNFLT, OCT, B-scan or glaucoma basics. (Demo only)</p>", unsafe_allow_html=True)

        # show history
        for msg in st.session_state.chat_history[-40:]:
            if msg.get("role") == "user":
                st.markdown(f"<div style='background: rgba(0,245,255,0.06); padding:8px; border-radius:8px; margin-bottom:6px;'><strong>You:</strong> {msg.get('content')}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background: rgba(255,64,196,0.04); padding:8px; border-radius:8px; margin-bottom:6px;'><strong>Assistant:</strong> {msg.get('content')}</div>", unsafe_allow_html=True)

        # input bound to session_state.chat_input
        user_q = st.text_input("Your question:", key="chat_input", placeholder="e.g., What is RNFLT?")

        c1, c2, c3 = st.columns([3,1,1])
        with c1:
            if st.button("📤 Send", use_container_width=True):
                if user_q:
                    # call assistant
                    with st.spinner("Thinking..."):
                        resp = ask_glaucoma_assistant(user_q, st.session_state.chat_history, API_KEY)
                    st.session_state.chat_history.append({"role":"user","content":user_q})
                    st.session_state.chat_history.append({"role":"assistant","content":resp})
                    st.session_state.chat_input = ""
                    st.experimental_rerun()
        with c2:
            if st.button("🗑️", use_container_width=True):
                st.session_state.chat_history = []
                st.experimental_rerun()
        with c3:
            if st.button("✖️", use_container_width=True):
                st.session_state.chat_open = False
                st.experimental_rerun()

# End of file
