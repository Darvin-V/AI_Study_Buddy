import streamlit as st
import PyPDF2
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
import random

# ==========================================
# 🔧 CONFIG & SETUP
# ==========================================
st.set_page_config(
    page_title="AI Study Buddy",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Beautiful UI
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); }
    .main-card { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 25px; 
        border-radius: 20px; 
        box-shadow: 0 8px 32px rgba(0,0,0,0.2); 
        margin: 15px 0; 
    }
    h1 { color: white; font-weight: 800; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
    h2, h3 { color: #2c3e50; }
    .stButton > button { border-radius: 12px; font-weight: 700; padding: 12px 28px; transition: all 0.3s; }
    .stButton > button:hover { transform: scale(1.05); box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
    .ai-badge { background: linear-gradient(90deg, #667eea, #764ba2); color: white; padding: 5px 15px; border-radius: 20px; font-size: 12px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 🛠️ HELPERS
# ==========================================
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def split_text(text, max_words=500):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# ==========================================
# 🤖 LOAD MODELS (Cached)
# ==========================================
@st.cache_resource
def load_summarizer():
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

@st.cache_resource
def load_chat_model():
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

@st.cache_resource
def load_classifier():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer

# ==========================================
# 🤖 AI CORE FUNCTIONS
# ==========================================
def summarize_text(text):
    try:
        model, tokenizer = load_summarizer()
        chunks = split_text(text, 500)
        summaries = []
        for chunk in chunks:
            inputs = tokenizer(chunk, return_tensors="pt", truncation=True)
            outputs = model.generate(**inputs, max_length=150, min_length=40)
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            summaries.append(summary)
        return " ".join(summaries)
    except Exception as e:
        st.error(f"Summarization Error: {e}")
        return None

def classify_text(text):
    try:
        model, tokenizer = load_classifier()
        inputs = tokenizer(text[:512], return_tensors="pt", truncation=True)
        outputs = model(**inputs)
        probs = outputs.logits.softmax(dim=-1)
        pred = probs.argmax().item()
        sentiment = "Positive" if pred == 1 else "Negative"
        confidence = probs[0][pred].item() * 100
        return {"sentiment": sentiment, "confidence": round(confidence, 2)}
    except Exception:
        return {"sentiment": "Unknown", "confidence": 0}

def extract_keywords(text):
    stop_words = set(["the", "is", "and", "to", "of", "in", "for", "on", "with", "at", "by", "a", "an", "be"])
    words = [w.lower() for w in text.split() if len(w) > 3 and w.lower() not in stop_words]
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:10]]

def generate_quiz_ai(summary):
    model, tokenizer = load_chat_model()
    prompts = [
        f"Generate a main question about this topic: {summary}",
        f"Create a question about the conclusion: {summary}",
        f"Make a comprehension question: {summary}"
    ]
    questions = []
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt", truncation=True)
        outputs = model.generate(**inputs, max_new_tokens=80)
        questions.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    questions += ["Summarize the main point.", "Explain the key takeaway."]
    return "\n\n".join([f"**Q{i+1}:** {q}" for i, q in enumerate(questions)])

def explain_concept_ai(question, context=""):
    model, tokenizer = load_chat_model()
    prompt = f"Context: {context[:800]}\n\nQuestion: {question}\nExplain clearly and simply."
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=300)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_flashcards(summary):
    model, tokenizer = load_chat_model()
    prompts = [f"Create a flashcard question about: {summary[:300]}", f"Generate another flashcard: {summary[:300]}"]
    cards = []
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt", truncation=True)
        outputs = model.generate(**inputs, max_new_tokens=80)
        cards.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    return cards

# ==========================================
# 🖥️ SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("## 🧠 AI Study Buddy")
    st.markdown("### 🤖 AI Technologies Used")
    st.markdown("""
    - **BART** — Text Summarization  
    - **FLAN-T5** — Q&A, Quiz, Flashcards  
    - **DistilBERT** — Sentiment  
    - **NLP** — Keywords Extraction
    """)
    if st.button("🗑️ Clear Session"):
        st.session_state.clear()
        st.rerun()

# ==========================================
# MAIN APP
# ==========================================
st.title("📚 AI Study Buddy")
st.markdown('<span class="ai-badge">AI PROJECT | TRANSFORMERS</span>', unsafe_allow_html=True)

if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'quiz' not in st.session_state:
    st.session_state.quiz = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'keywords' not in st.session_state:
    st.session_state.keywords = []
if 'sentiment' not in st.session_state:
    st.session_state.sentiment = {}

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Input & Summary", "🧩 AI Quiz", "💬 Ask AI", "🔑 Keywords", "💡 Flashcards"])

# --- TAB 1 ---
with tab1:
    st.markdown("#### 📄 Upload Your Study Material")
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    with col2:
        text_input = st.text_area("Or Paste Notes", height=150)

    if st.button("🚀 Generate AI Summary", use_container_width=True):
        text = ""
        if uploaded_file:
            text = extract_text_from_pdf(uploaded_file)
        elif text_input.strip():
            text = text_input.strip()
        if text:
            with st.spinner("🤖 Summarizing..."):
                summary = summarize_text(text)
                keywords = extract_keywords(text)
                sentiment = classify_text(text[:1000])
                if summary:
                    st.session_state.summary = summary
                    st.session_state.keywords = keywords
                    st.session_state.sentiment = sentiment
                    st.rerun()
                else:
                    st.error("Failed to summarize.")
        else:
            st.error("Please upload a file or paste text.")

    if st.session_state.summary:
        st.markdown("### 📋 AI Summary")
        st.markdown(f"<div class='main-card'>{st.session_state.summary}</div>", unsafe_allow_html=True)
        st.metric("💭 Sentiment", st.session_state.sentiment.get("sentiment", "N/A"))
        st.metric("🔑 Keywords Found", len(st.session_state.keywords))

# --- TAB 2 ---
with tab2:
    if not st.session_state.summary:
        st.warning("Generate a summary first.")
    else:
        if st.button("✨ Generate Quiz", use_container_width=True):
            with st.spinner("Creating quiz..."):
                st.session_state.quiz = generate_quiz_ai(st.session_state.summary)
                st.rerun()
        if st.session_state.quiz:
            st.markdown(f"<div class='main-card'>{st.session_state.quiz}</div>", unsafe_allow_html=True)

# --- TAB 3 ---
with tab3:
    question = st.text_input("Ask a question:", placeholder="Explain this concept...")
    if st.button("Send", use_container_width=True):
        st.session_state.chat_history.append({"role": "user", "content": question})
        answer = explain_concept_ai(question, st.session_state.summary)
        st.session_state.chat_history.append({"role": "ai", "content": answer})
        st.rerun()

    for msg in st.session_state.chat_history:
        st.markdown(f"**🙋 You:** {msg['content']}" if msg["role"] == "user" else f"**🤖 AI:** {msg['content']}")

# --- TAB 4 ---
with tab4:
    if not st.session_state.keywords:
        st.warning("Generate summary first.")
    else:
        st.markdown("### 🔑 Extracted Keywords")
        st.markdown(f"<div class='main-card'>{', '.join(st.session_state.keywords)}</div>", unsafe_allow_html=True)

# --- TAB 5 ---
with tab5:
    if not st.session_state.summary:
        st.warning("Generate summary first.")
    else:
        if st.button("💡 Generate Flashcards", use_container_width=True):
            with st.spinner("Creating flashcards..."):
                cards = generate_flashcards(st.session_state.summary)
                for c in cards:
                    st.markdown(f"<div class='main-card'>🗂️ {c}</div>", unsafe_allow_html=True)
