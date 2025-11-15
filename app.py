# -----------------------
# Model Selection
# -----------------------
MODEL_NAME = "models/gemini-2.5-pro"

# -----------------------
# Chatbot Function
# -----------------------
def ask_glaucoma_assistant(question, history, api_key):
    if not api_key:
        return "⚠️ Please configure your Google Gemini API key."

    system_instruction = """You are a specialized medical AI assistant ..."""

    try:
        if USE_SDK:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(MODEL_NAME)

            chat_history = []
            for msg in history[-6:]:
                role = "user" if msg["role"] == "user" else "model"
                chat_history.append({"role": role, "parts": [msg["content"]]})

            chat = model.start_chat(history=chat_history)
            response = chat.send_message(f"{system_instruction}\n\nUser question: {question}")
            return response.text
        else:
            # REST fallback
            full_prompt = ...
            url = f"https://generativelanguage.googleapis.com/v1beta/{MODEL_NAME}:generateContent?key={api_key}"

            response = requests.post(url, ...)
