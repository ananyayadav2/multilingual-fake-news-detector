import streamlit as st
import joblib
import re
import nltk
import requests
from datetime import date
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from langdetect import detect, DetectorFactory

# Set seed for reproducible language detection
DetectorFactory.seed = 0

# Setup NLTK resources
nltk.download('stopwords', quiet=True)
stemmer = PorterStemmer()
stop_words_en = set(stopwords.words('english'))

# ==========================================
# CONFIGURATION & PAGE SETUP
# ==========================================
NEWS_API_KEY = "13a50766cc8f46f4a0729eafbce4ee69"

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Light Grey BG, Dark Grey Hero, Upgraded Right Panel)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    /* Overall Page Background */
    .stApp {
        background-color: #f3f4f6;
    }
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Remove default Streamlit top padding */
    .block-container {
        padding-top: 2rem !important;
    }
    
    /* Dark Grey Hero Container */
    .hero-container {
        background: linear-gradient(135deg, #4b5563 0%, #1f2937 100%);
        padding: 2.2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        border: 1px solid #374151;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
        color: #f9fafb;
    }
    .hero-sub {
        font-size: 1rem;
        color: #9ca3af;
        margin-bottom: 0;
    }
    
    /* ML Card Styles */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.4rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
        margin-bottom: 1.2rem;
        height: 100%;
    }
    .status-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    .badge-real { background-color: #dcfce7; color: #15803d; }
    .badge-fake { background-color: #fee2e2; color: #b91c1c; }
    
    /* Upgraded Live Article Cards */
    .article-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 1.1rem;
        margin-bottom: 1rem;
        transition: all 0.2s ease;
    }
    .article-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border-color: #cbd5e1;
        transform: translateY(-2px);
    }
    .article-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #0f172a;
        text-decoration: none;
        display: block;
        margin-bottom: 0.5rem;
        line-height: 1.4;
    }
    .article-title:hover { color: #2563eb; }
    .article-meta {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 500;
    }
    
    /* Custom API Status Alerts */
    .alert-box {
        padding: 1.1rem;
        border-radius: 8px;
        margin-top: 0.5rem;
        font-weight: 500;
        font-size: 0.95rem;
    }
    .alert-error { background-color: #fef2f2; border: 1px solid #fecaca; color: #991b1b; }
    .alert-warning { background-color: #fffbeb; border: 1px solid #fde68a; color: #92400e; }
</style>
""", unsafe_allow_html=True)

# Top Hero Header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🛡️ Fake News Detection</div>
    <p class="hero-sub">Hybrid Multilingual Misinformation Classifier & Real-Time Event Verifier</p>
</div>
""", unsafe_allow_html=True)

# Sidebar System Specs
with st.sidebar:
    st.header("⚡ System Specs")
    st.markdown("**Core Architecture**")
    st.markdown("- Dual ML Pipelines (Random Forest)")
    st.markdown("- TF-IDF Character & Word n-grams")
    st.markdown("- Real-Time Fact Retrieval Engine")
    st.divider()
    st.markdown("**Supported Languages**")
    st.markdown("🇬🇧 **English** (Global Corpus)")
    st.markdown("🇮🇳 **Hindi** (Devanagari Corpus)")
    st.divider()
    st.caption("Designed for Academic Research & Live Event Verification")

# Load models and artifacts
@st.cache_resource
def load_artifacts():
    try:
        en_v = joblib.load('english_tfidf_vectorizer.pkl')
        en_m = joblib.load('english_model.pkl')
        en_met = joblib.load('english_metrics.pkl')
        
        hi_v = joblib.load('hindi_tfidf_vectorizer.pkl')
        hi_m = joblib.load('hindi_model.pkl')
        hi_met = joblib.load('hindi_metrics.pkl')
        
        return (en_v, en_m, en_met), (hi_v, hi_m, hi_met)
    except Exception:
        return None, None

en_bundle, hi_bundle = load_artifacts()

if en_bundle is None or hi_bundle is None:
    st.error("Model artifacts missing. Please ensure all `.pkl` files reside in your workspace.")
    st.stop()

en_vec, en_model, en_metrics = en_bundle
hi_vec, hi_model, hi_metrics = hi_bundle

def preprocess_english(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text)).lower()
    return " ".join([stemmer.stem(w) for w in text.split() if w not in stop_words_en])

def preprocess_hindi(text):
    text = re.sub(r'[^\u0900-\u097F\s]', '', str(text))
    return " ".join(text.split())

def fetch_live_news(query, check_today=False):
    if not NEWS_API_KEY or NEWS_API_KEY == "YOUR_NEWS_API_KEY_HERE":
        return None
    
    today_str = date.today().isoformat()
    date_param = f"&from={today_str}&sortBy=publishedAt" if check_today else "&sortBy=publishedAt"
    
    # Changed pageSize from 3 to 5
    url = f"https://newsapi.org/v2/everything?q={query}{date_param}&pageSize=5&apiKey={NEWS_API_KEY}"
    try:
        res = requests.get(url, timeout=5).json()
        if res.get("status") == "ok":
            return res.get("articles", [])
    except Exception:
        return []
    return []

# Input Section
user_input = st.text_area(
    "Input News Headline or Claim:",
    height=110,
    placeholder="Type or paste an article statement in English or Hindi..."
)

col_btn, _ = st.columns([1, 4])
with col_btn:
    analyze_btn = st.button("Run Verification", type="primary", use_container_width=True)

if analyze_btn:
    if not user_input.strip():
        st.warning("Please supply a text input prior to analysis.")
        st.stop()

    # Language Detection
    detected_lang = "en"
    try:
        detected_lang = detect(user_input)
    except Exception:
        pass

    is_hindi = detected_lang == "hi"
    
    if is_hindi:
        clean_text = preprocess_hindi(user_input)
        vectorizer, model, metrics = hi_vec, hi_model, hi_metrics
        lang_display = "Hindi (Devanagari)"
    else:
        clean_text = preprocess_english(user_input)
        vectorizer, model, metrics = en_vec, en_model, en_metrics
        lang_display = "English"

    # Prediction
    transformed = vectorizer.transform([clean_text])
    raw_pred = model.predict(transformed)[0]
    probabilities = model.predict_proba(transformed)[0]
    
    if is_hindi:
        pred = 1 if raw_pred == 0 else 0
        confidence = probabilities[raw_pred] * 100
    else:
        pred = raw_pred
        confidence = probabilities[pred] * 100

    col1, col2 = st.columns(2)

    # Panel 1: ML Model
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>⚙️ Stylistic Pattern Analysis</h4>
            <span style="color:#64748b; font-size: 0.9rem;">Language: <b>{lang_display}</b></span>
            <hr style="margin: 0.8rem 0; border: none; border-top: 1px solid #f1f5f9;">
        """, unsafe_allow_html=True)
        
        if pred == 1:
            st.markdown('<span class="status-badge badge-real">Authentic Style</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge badge-fake">Misleading Style</span>', unsafe_allow_html=True)
            
        st.write(f"Confidence Score: **{confidence:.1f}%**")
        st.progress(int(confidence))
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Model Benchmark Metrics"):
            st.json(metrics)

    # Panel 2: Live Fact-Checking
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>🌐 Live Source Retrieval</h4>
            <span style="color:#64748b; font-size: 0.9rem;">Target: Global News Wire (Real-Time)</span>
            <hr style="margin: 0.8rem 0; border: none; border-top: 1px solid #f1f5f9;">
        """, unsafe_allow_html=True)
        
        is_today_query = "today" in user_input.lower() or "आज" in user_input
        ignore_words = {
            "was", "is", "are", "were", "there", "today", "now", "the", "this", 
            "that", "a", "an", "in", "on", "at", "for", "to", "of", "did", "have", "has",
            "में", "है", "और", "की", "गई", "आज"
        }
        cleaned_words = [re.sub(r'[^a-zA-Z0-9\u0900-\u097F]', '', w) for w in user_input.split()]
        meaningful_words = [w for w in cleaned_words if w.lower() not in ignore_words and len(w) > 2]
        search_terms = " AND ".join(meaningful_words[:4]) if meaningful_words else user_input.strip()

        with st.spinner("Querying wire services..."):
            articles = fetch_live_news(search_terms, check_today=is_today_query)

        if articles is None:
            st.markdown('<div class="alert-box alert-warning">⚠️ NewsAPI key unconfigured.</div>', unsafe_allow_html=True)
        elif len(articles) == 0:
            if is_today_query:
                st.markdown('<div class="alert-box alert-error">❌ <b>No Coverage Found</b><br>No verified publications reported this claim today.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-box alert-warning">⚠️ <b>No Active Coverage</b><br>No mainstream coverage found matching this topic.</div>', unsafe_allow_html=True)
        else:
            for art in articles:
                pub_date = art.get('publishedAt', '')[:10]
                st.markdown(f"""
                <div class="article-card">
                    <a href="{art['url']}" target="_blank" class="article-title">{art['title']}</a>
                    <div class="article-meta">📰 {art['source']['name']} &nbsp;|&nbsp; 📅 {pub_date}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Synthesis Verdict Banner
    st.subheader("🏁 Final Consensus Verdict")

    if articles and len(articles) > 0:
        st.success("✅ **VERIFIED AUTHENTIC NEWS** — Real-time reporting from credible outlets confirms the event.")
    elif is_today_query and (articles is None or len(articles) == 0):
        st.error("🚨 **UNVERIFIED / FABRICATED EVENT** — Claim is phrased as a current event, but no news wire records match.")
    elif pred == 0:
        st.error("🚨 **FLAGGED AS FAKE NEWS** — Phrasing aligns heavily with known disinformation patterns.")
    else:
        st.info("ℹ️ **UNVERIFIED CONTEXT** — Linguistic patterns match standard news, but external sourcing remains quiet.")