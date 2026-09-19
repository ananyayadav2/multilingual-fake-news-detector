import streamlit as st
import joblib
import re
import nltk
import requests
import urllib.parse
from datetime import date
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator, MyMemoryTranslator

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

# Initialize Session State for Pagination
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'page' not in st.session_state:
    st.session_state.page = 0

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    .stApp { background-color: #f3f4f6; }
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .block-container { padding-top: 2rem !important; }
    
    .hero-container {
        background: linear-gradient(135deg, #4b5563 0%, #1f2937 100%);
        padding: 2.2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        border: 1px solid #374151;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .hero-title { font-size: 2.1rem; font-weight: 700; margin-bottom: 0.4rem; color: #f9fafb; }
    .hero-sub { font-size: 1rem; color: #9ca3af; margin-bottom: 0; }
    
    .metric-card {
        background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px;
        padding: 1.4rem; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02); margin-bottom: 1.2rem; height: 100%;
    }
    .status-badge { display: inline-block; padding: 0.35rem 0.8rem; border-radius: 9999px; font-size: 0.85rem; font-weight: 600; margin-bottom: 0.8rem; }
    .badge-real { background-color: #dcfce7; color: #15803d; }
    .badge-fake { background-color: #fee2e2; color: #b91c1c; }
    
    .article-card {
        background: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #3b82f6;
        border-radius: 8px; padding: 1.1rem; margin-bottom: 1rem; transition: all 0.2s ease;
    }
    .article-card:hover { box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08); border-color: #cbd5e1; transform: translateY(-2px); }
    .article-title { font-size: 1.05rem; font-weight: 600; color: #0f172a; text-decoration: none; display: block; margin-bottom: 0.5rem; line-height: 1.4; }
    .article-title:hover { color: #2563eb; }
    .article-meta { font-size: 0.85rem; color: #64748b; font-weight: 500; }
    
    .alert-box { padding: 1.1rem; border-radius: 8px; margin-top: 0.5rem; font-weight: 500; font-size: 0.95rem; }
    .alert-error { background-color: #fef2f2; border: 1px solid #fecaca; color: #991b1b; }
    .alert-warning { background-color: #fffbeb; border: 1px solid #fde68a; color: #92400e; }
    
    div[data-testid="column"] button { width: 100%; }
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
    st.markdown("- Tri-Pipeline ML (Random Forest)")
    st.markdown("- TF-IDF Character & Word n-grams")
    st.markdown("- Real-Time Fact Retrieval Engine")
    st.divider()
    st.markdown("**Supported Languages**")
    st.markdown("🇬🇧 **English** (Global Corpus)")
    st.markdown("🇮🇳 **Hindi** (Devanagari Corpus)")
    st.markdown("🚩 **Marathi** (Devanagari Corpus)")
    st.divider()
    st.caption("Designed for Academic Research & Live Event Verification")

# ==========================================
# LAZY MODEL LOADER (Fixes Memory Crash)
# ==========================================
@st.cache_resource
def load_model(language):
    try:
        vec = joblib.load(f'{language}_tfidf_vectorizer.pkl')
        mod = joblib.load(f'{language}_model.pkl')
        met = joblib.load(f'{language}_metrics.pkl')
        return vec, mod, met
    except Exception as e:
        return None

# Preprocessing routines
def preprocess_english(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text)).lower()
    return " ".join([stemmer.stem(w) for w in text.split() if w not in stop_words_en])

def preprocess_indic(text):
    text = re.sub(r'[^\u0900-\u097F\s]', '', str(text))
    return " ".join(text.split())

def fetch_live_news(query, check_today=False):
    if not NEWS_API_KEY or NEWS_API_KEY == "YOUR_NEWS_API_KEY_HERE":
        return None
    
    today_str = date.today().isoformat()
    date_param = f"&from={today_str}&sortBy=publishedAt" if check_today else f"&sortBy=relevancy"
    safe_query = urllib.parse.quote(query)
    
    url = f"https://newsapi.org/v2/everything?q={safe_query}&searchIn=title,description{date_param}&pageSize=12&apiKey={NEWS_API_KEY}"
    try:
        res = requests.get(url, timeout=5).json()
        if res.get("status") == "ok":
            return res.get("articles", [])
    except Exception:
        return []
    return []

# Input Section
user_input = st.text_area("Input News Headline or Claim:", height=110, placeholder="Type or paste an article statement in English, Hindi, or Marathi...")

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
    is_marathi = detected_lang == "mr"
    
    # Routing & Lazy Loading
    if is_marathi:
        lang_name = 'marathi'
        lang_display = "Marathi (Devanagari)"
        clean_text = preprocess_indic(user_input)
    elif is_hindi:
        lang_name = 'hindi'
        lang_display = "Hindi (Devanagari)"
        clean_text = preprocess_indic(user_input)
    else:
        lang_name = 'english'
        lang_display = "English"
        clean_text = preprocess_english(user_input)

    with st.spinner(f"Loading {lang_display} AI Model..."):
        bundle = load_model(lang_name)
        
    if bundle is None:
        st.error(f"🚨 Missing or corrupt `.pkl` files for {lang_display}.")
        st.stop()
        
    vectorizer, model, metrics = bundle

    # Prediction
    transformed = vectorizer.transform([clean_text])
    raw_pred = model.predict(transformed)[0]
    probabilities = model.predict_proba(transformed)[0]
    
    # Label Handling (Only Hindi dataset used 0=Real, Marathi and English use 1=Real)
    if is_hindi:
        pred = 1 if raw_pred == 0 else 0
        confidence = probabilities[raw_pred] * 100
    else:
        pred = raw_pred
        confidence = probabilities[pred] * 100

    # API Prep
    is_today_query = "today" in user_input.lower() or "आज" in user_input
    
    ignore_words = {
        "was", "is", "are", "were", "there", "today", "now", "the", "this", "that", "a", "an", "in", "on", "at", 
        "for", "to", "of", "did", "have", "has", "recently", "recent", "about", "news", "tell", "me", "what", "why", "how",
        "में", "है", "और", "की", "गई", "आज", "का",
        "आहे", "नाही", "आणि", "व", "ते", "हे", "या", "का"
    }
    cleaned_words = [re.sub(r'[^a-zA-Z0-9\u0900-\u097F]', '', w) for w in user_input.split()]
    meaningful_words = [w for w in cleaned_words if w.lower() not in ignore_words and len(w) > 2]
    
    raw_search = " ".join(meaningful_words[:4]) if meaningful_words else user_input.strip()
    
    # TRANSLATION LAYER: Convert Hindi/Marathi queries to English for NewsAPI
    search_terms = raw_search
    if is_hindi or is_marathi:
        src_lang = 'mr' if is_marathi else 'hi'
        try:
            # Attempt 1: Google Translate
            translator = GoogleTranslator(source=src_lang, target='en')
            search_terms = translator.translate(raw_search)
        except Exception:
            try:
                # Attempt 2: Backup Translator if Google blocks Streamlit's IP
                translator = MyMemoryTranslator(source=src_lang, target='en')
                search_terms = translator.translate(raw_search)
            except Exception:
                st.warning("⚠️ Background translation temporarily blocked by translation servers. Live search is using original text.")

    with st.spinner("Translating query and searching global wire services..."):
        articles = fetch_live_news(search_terms, check_today=is_today_query)

    # Save to Session State
    st.session_state.analyzed = True
    st.session_state.page = 0
    st.session_state.pred = pred
    st.session_state.confidence = confidence
    st.session_state.lang_display = lang_display
    st.session_state.metrics = metrics
    st.session_state.is_today_query = is_today_query
    st.session_state.articles = articles
    st.session_state.translated_query = search_terms

# ==========================================
# DISPLAY RESULTS
# ==========================================
if st.session_state.analyzed:
    col1, col2 = st.columns(2)

    # Panel 1: ML Model
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>⚙️ Stylistic Pattern Analysis</h4>
            <span style="color:#64748b; font-size: 0.9rem;">Language: <b>{st.session_state.lang_display}</b></span>
            <hr style="margin: 0.8rem 0; border: none; border-top: 1px solid #f1f5f9;">
        """, unsafe_allow_html=True)
        
        if st.session_state.pred == 1:
            st.markdown('<span class="status-badge badge-real">Authentic Style</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge badge-fake">Misleading Style</span>', unsafe_allow_html=True)
            
        st.write(f"Confidence Score: **{st.session_state.confidence:.1f}%**")
        st.progress(int(st.session_state.confidence))
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Model Benchmark Metrics"):
            st.json(st.session_state.metrics)

    # Panel 2: Live Fact-Checking
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🌐 Live Source Retrieval</h4>
            <span style="color:#64748b; font-size: 0.9rem;">Target: Global News Wire (Real-Time)</span><br>
            <span style="color:#94a3b8; font-size: 0.8rem;"><i>API Search Term: "{st.session_state.translated_query}"</i></span>
            <hr style="margin: 0.8rem 0; border: none; border-top: 1px solid #f1f5f9;">
        """, unsafe_allow_html=True)
        
        articles = st.session_state.articles

        if articles is None:
            st.markdown('<div class="alert-box alert-warning">⚠️ NewsAPI key unconfigured.</div>', unsafe_allow_html=True)
        elif len(articles) == 0:
            if st.session_state.is_today_query:
                st.markdown('<div class="alert-box alert-error">❌ <b>No Coverage Found</b><br>No verified publications reported this claim today.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-box alert-warning">⚠️ <b>No Active Coverage</b><br>No highly relevant mainstream coverage found matching this topic.</div>', unsafe_allow_html=True)
        else:
            PAGE_SIZE = 3
            total_pages = (len(articles) - 1) // PAGE_SIZE + 1
            start_idx = st.session_state.page * PAGE_SIZE
            end_idx = start_idx + PAGE_SIZE
            
            for art in articles[start_idx:end_idx]:
                pub_date = art.get('publishedAt', '')[:10]
                st.markdown(f"""
                <div class="article-card">
                    <a href="{art['url']}" target="_blank" class="article-title">{art['title']}</a>
                    <div class="article-meta">📰 {art['source']['name']} &nbsp;|&nbsp; 📅 {pub_date}</div>
                </div>
                """, unsafe_allow_html=True)

            if total_pages > 1:
                st.markdown("<hr style='margin: 1rem 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
                p_col1, p_col2, p_col3 = st.columns([1, 1, 1])
                with p_col1:
                    if st.session_state.page > 0 and st.button("⬅️ Prev"):
                        st.session_state.page -= 1
                        st.rerun()
                with p_col2:
                    st.markdown(f"<div style='text-align:center; padding-top:0.4rem; color:#64748b; font-size:0.9rem;'>Page {st.session_state.page + 1} of {total_pages}</div>", unsafe_allow_html=True)
                with p_col3:
                    if st.session_state.page < total_pages - 1 and st.button("Next ➡️"):
                        st.session_state.page += 1
                        st.rerun()
                        
        st.markdown("</div>", unsafe_allow_html=True)

    # Synthesis Verdict Banner
    st.subheader("🏁 Final Consensus Verdict")

    if articles and len(articles) > 0:
        st.success("✅ **VERIFIED AUTHENTIC NEWS** — Real-time reporting from credible outlets confirms the event.")
    elif st.session_state.is_today_query and (articles is None or len(articles) == 0):
        st.error("🚨 **UNVERIFIED / FABRICATED EVENT** — Claim is phrased as a current event, but no news wire records match.")
    elif st.session_state.pred == 0:
        st.error("🚨 **FLAGGED AS FAKE NEWS** — Phrasing aligns heavily with known disinformation patterns.")
    else:
        st.info("ℹ️ **UNVERIFIED CONTEXT** — Linguistic patterns match standard news, but external sourcing remains quiet.")