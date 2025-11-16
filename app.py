# app.py — OCULAIRE Neon Lab v5 with Glaucoma Chatbot (FINAL FIXED SUBMISSION)
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
# Initialize Session State for Chat
# -----------------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chat_open' not in st.session_state:
    st.session_state.chat_open = False
if 'chat_input' not in st.session_state: # Use 'chat_input' consistently for the text box key
    st.session_state.chat_input = ""

# Get API key (omitted for brevity)
def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except:
        return os.getenv("GEMINI_API_KEY")

API_KEY = get_api_key()

# -----------------------
# Neon Matplotlib Config / CSS (omitted for brevity)
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

# CSS... (omitted)

st.markdown("""
<style>
/* ... (Your existing CSS here) ... */
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
/* ... (rest of your existing CSS) ... */
</style>
""", unsafe_allow_html=True)


# -----------------------
# Chatbot Function
# -----------------------
def ask_glaucoma_assistant(question, history, api_key):
    """Call Google Gemini API with glaucoma-specific context"""
    
    if not api_key or not api_key.strip():
        return "⚠️ Please configure your Google Gemini API key (see sidebar)."
    
    # System prompt
    system_instruction = """You are a specialized medical AI assistant focused exclusively on glaucoma. 
    ... (omitted) ...
    """
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
            # Use official Google AI SDK
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Build conversation
            chat_history = []
            for msg in history[-6:]:
                role = "user" if msg["role"] == "user" else "model"
                chat_history.append({"role": role, "parts": [{"text": msg["content"]}]})
            
            chat = model.start_chat(history=chat_history)
            response = chat.send_message(f"{system_instruction}\n\nUser question: {question}")
            return response.text
            
        else:
            # Fallback to REST API (using simplified logic for this fixed version)
            # You should replace this with your original full REST API logic if needed
            return "❌ API SDK not available. Please install the `google-genai` package for chat functionality."
            
    except Exception as e:
        return f"❌ Error: {str(e)}\n\nTip: Make sure your API key is correct and unrestricted."

# -----------------------
# CHAT SUBMISSION HANDLERS <--- THE FINAL FIX IS HERE
# -----------------------
def submit_chat_question():
    """Handles the chat message submission logic using st.session_state.chat_input."""
    user_question = st.session_state.chat_input
    
    if not API_KEY or not API_KEY.strip():
        st.session_state.chat_history.append({"role": "assistant", "content": "❌ No valid API key is configured. Chatbot is disabled."})
        st.session_state.chat_input = ""
        st.experimental_rerun()
        return
        
    if user_question and user_question.strip():
        # 1. Add user question to history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        # 2. Get the response
        # Note: The spinner will only show on the *next* run unless we use st.empty or st.status
        # Since we force a rerun later, this is fine.
        response = ask_glaucoma_assistant(user_question, st.session_state.chat_history, API_KEY)
        
        # 3. Add assistant response
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # 4. Clear the input box (by updating session state)
        st.session_state.chat_input = ""
        
        # 5. Force a rerun to update the chat history display and clear the input box
        st.experimental_rerun()

def clear_chat_history():
    st.session_state.chat_history = []
    st.experimental_rerun()

def close_chat():
    st.session_state.chat_open = False
    st.experimental_rerun()


# -----------------------
# Header / Models / Helpers (omitted for brevity)
# -----------------------
st.markdown("""
<div class="header">
  <h1>👁️ OCULAIRE</h1>
  <h3>AI-Powered Glaucoma Detection Dashboard — Neon Lab v5</h3>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ... (Load Models, Helper functions, Main Layout) ...

# -----------------------
# FLOATING CHAT WIDGET (Bottom-right corner)
# -----------------------
st.markdown('<div class="chat-bubble" id="chatBubble">🤖</div><div class="floating-chat-pill" id="chatPill">Ask Assistant</div>', unsafe_allow_html=True)

# Hidden toggle button logic
_TOGGLE_UNIQUE_LABEL = "__OCULAIRE_TOGGLE_CHAT__"
st.markdown('<div style="position:absolute;left:-9999px;top:-9999px;opacity:0;">', unsafe_allow_html=True)
toggle_clicked = st.button(_TOGGLE_UNIQUE_LABEL, key="__oculaire_toggle_button__")
st.markdown('</div>', unsafe_allow_html=True)

if toggle_clicked:
    st.session_state.chat_open = not st.session_state.chat_open
    st.experimental_rerun()

# JS to click the hidden button (omitted for brevity)
st.markdown(f"""
<script>
(function(){{
  function clickHidden() {{
    const targetLabel = "{_TOGGLE_UNIQUE_LABEL}";
    // Search for button by innerText in current and parent documents
    const findAndClick = (doc) => {{
      const buttons = Array.from(doc.querySelectorAll('button'));
      for (let b of buttons) {{
        if ((b.innerText || "").trim() === targetLabel) {{
          b.click();
          return true;
        }}
      }}
      return false;
    }};
    if (findAndClick(document)) return;
    try {{ if (window.parent && window.parent.document) findAndClick(window.parent.document); }} catch(e) {{}}
  }}

  const bubble = document.getElementById('chatBubble');
  const pill = document.getElementById('chatPill');
  [bubble, pill].forEach(el => {{
    if (!el) return;
    el.style.cursor = 'pointer';
    el.addEventListener('click', function(e) {{
      e.preventDefault();
      el.style.transform = 'scale(0.98)';
      setTimeout(()=> el.style.transform = '', 120);
      clickHidden();
    }});
  }});
}})();
</script>
""", unsafe_allow_html=True)

# ... (Main content layout/analysis) ...

# -----------------------
# When chat is open, show in sidebar
# -----------------------
if st.session_state.chat_open:
    with st.sidebar:
        st.markdown("---")
        st.markdown("<div class='chat-header'>🤖 Glaucoma Assistant</div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:var(--muted); font-size:13px; margin-bottom:15px;'>Ask me anything about glaucoma!</p>", unsafe_allow_html=True)
        
        # Display chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"<div class='user-msg'><strong>You:</strong> {msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='assistant-msg'><strong>🤖:</strong> {msg['content']}</div>", unsafe_allow_html=True)
        
        # Input area using the key 'chat_input'
        # HACK: Using st.empty to create a placeholder for text_input and button group
        # This helps manage the flow and ensures the text_input key is consistent.
        
        # Standard text input. Its value is automatically stored in st.session_state.chat_input
        st.text_input("Your question:", key="chat_input", placeholder="What is glaucoma?", 
                      on_change=submit_chat_question) # Submit on ENTER key press
        
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            # Standard button with explicit on_click handler for clarity
            st.button("📤 Send", use_container_width=True, on_click=submit_chat_question)
            
        with col2:
            st.button("🗑️", use_container_width=True, on_click=clear_chat_history)
            
        with col3:
            st.button("✖️", use_container_width=True, on_click=close_chat)
