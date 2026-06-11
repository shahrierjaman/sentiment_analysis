# ==============================================
#  Streamlit – Professional Sentiment Analyzer
# ==============================================

import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np
import re
from PIL import Image
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from collections import Counter

# ----- Page config -----
st.set_page_config(
    page_title="SentimentScope",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----- Custom CSS (clean, readable) -----
st.markdown("""
<style>
    /* Overall background */
    .stApp {
        background-color: #f8f9fa;
    }
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        color: #4a4e69;              /* solid dark color, no transparency */
        margin-bottom: 0.25rem;
        letter-spacing: -0.5px;
        text-shadow: 1px 1px 0 rgba(255,255,255,0.7);
    }
    .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .card-style {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    /* Make dataframe/text inside cards clearly dark */
    .stDataFrame, .stMarkdown, .stText {
        color: #212529;
    }
</style>
""", unsafe_allow_html=True)

# ----- Cache resources -----
@st.cache_resource
def load_nltk():
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

@st.cache_resource
def load_pipeline():
    return joblib.load('sentiment_pipeline.pkl')

@st.cache_data
def load_metrics():
    with open('metrics.json') as f:
        return json.load(f)

load_nltk()
pipeline = load_pipeline()
metrics = load_metrics()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# ----- Top influential words (from model coefficients) -----
def get_top_words(pipeline, n=20):
    vectorizer = pipeline.named_steps['tfidf']
    classifier = pipeline.named_steps['clf']
    feature_names = vectorizer.get_feature_names_out()
    coef = classifier.coef_[0]
    top_positive_idx = np.argsort(coef)[-n:][::-1]
    top_negative_idx = np.argsort(coef)[:n]
    return ([(feature_names[i], coef[i]) for i in top_positive_idx],
            [(feature_names[i], coef[i]) for i in top_negative_idx])

top_pos, top_neg = get_top_words(pipeline)

# ----- Session state for prediction history -----
if 'history' not in st.session_state:
    st.session_state.history = []

# ----- Header -----
st.markdown('<div class="main-title">SentimentScope</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Deep Analysis of Movie Reviews – Powered by Machine Learning</div>',
            unsafe_allow_html=True)

# ----- Main input section -----
with st.container():
    st.markdown('<div class="card-style">', unsafe_allow_html=True)
    col_input, col_result = st.columns([3, 2], gap="large")

    with col_input:
        st.subheader("✍️ Your Review")
        user_review = st.text_area(
            "Paste a movie review...",
            height=200,
            placeholder="e.g., 'This film was absolutely breathtaking...'"
        )

        # Example quick select
        example = st.selectbox(
            "Try an example:",
            ["Select...",
             "Absolutely wonderful! The acting was superb and the plot kept me engaged.",
             "Terrible movie. A complete waste of time and money.",
             "It was okay, nothing special but not the worst I've seen."]
        )
        if example != "Select...":
            user_review = example

        analyze_btn = st.button("🔍 Analyze Sentiment", use_container_width=True)

    # ----- Analysis & Results -----
    with col_result:
        if analyze_btn and user_review.strip():
            cleaned = clean_text(user_review)
            pred = pipeline.predict([cleaned])[0]
            proba = pipeline.predict_proba([cleaned])[0]  # [neg, pos]
            confidence = proba[1] if pred == 1 else proba[0]

            # Store in history
            st.session_state.history.append({
                'review': user_review[:200] + ('...' if len(user_review) > 200 else ''),
                'sentiment': 'Positive 😊' if pred == 1 else 'Negative 😞',
                'confidence': f'{confidence:.2%}'
            })

            # Result box
            if pred == 1:
                st.success(f"### 😊 Positive Sentiment (confidence: {confidence:.1%})")
            else:
                st.error(f"### 😞 Negative Sentiment (confidence: {confidence:.1%})")

            # Confidence gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence * 100,
                number={'suffix': "%", 'font': {'size': 28, 'color': 'darkblue'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': '#f0f0f0'},
                        {'range': [50, 100], 'color': '#e0e0e0'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

            # Word cloud of input review
            if len(cleaned.split()) > 1:
                st.markdown("**Word Cloud of Your Review**")
                wc = WordCloud(width=400, height=200, background_color='white',
                               colormap='cool', max_words=100).generate(cleaned)
                fig_wc, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig_wc)
            else:
                st.caption("Not enough words to generate a word cloud.")

        elif analyze_btn and not user_review.strip():
            st.warning("Please enter some text.")
    st.markdown('</div>', unsafe_allow_html=True)

# ----- Tabs for deeper insights -----
tab1, tab2, tab3, tab4 = st.tabs(["📈 Model Insights", "📊 Performance", "🌐 Global Word Clouds", "🕒 History"])

with tab1:
    st.markdown("### 🔝 Top Influential Words")
    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.write("**Positive words**")
        df_pos = pd.DataFrame(top_pos, columns=['Word', 'Coefficient']).head(15)
        st.dataframe(df_pos.style.background_gradient(cmap='Greens'), use_container_width=True)
    with col_neg:
        st.write("**Negative words**")
        df_neg = pd.DataFrame(top_neg, columns=['Word', 'Coefficient']).head(15)
        st.dataframe(df_neg.style.background_gradient(cmap='Reds'), use_container_width=True)

with tab2:
    st.markdown("### 📋 Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Test Accuracy", f"{metrics['accuracy']:.2%}")
        st.write("**Classification Report**")
        report_df = pd.DataFrame(metrics['classification_report']).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)
    with col2:
        st.write("**Confusion Matrix**")
        cm = np.array(metrics['confusion_matrix'])
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=['Pred Neg', 'Pred Pos'], y=['True Neg', 'True Pos'],
            colorscale='Blues', text=cm, texttemplate="%{text}",
            textfont={"color": "white"}, hoverongaps=False))
        fig_cm.update_layout(height=300)
        st.plotly_chart(fig_cm, use_container_width=True)

with tab3:
    st.markdown("### 🌍 Global Word Clouds (Full Dataset)")
    col3, col4 = st.columns(2)
    with col3:
        st.image('wordcloud_positive.png', caption='Positive Reviews', use_column_width=True)
    with col4:
        st.image('wordcloud_negative.png', caption='Negative Reviews', use_column_width=True)

with tab4:
    st.markdown("### 🕒 Recent Predictions")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history[::-1])  # newest first
        st.dataframe(hist_df, use_container_width=True)
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions yet. Analyze a review to see history.")

# ----- Footer -----
st.markdown("---")
st.markdown("<div style='text-align: center; color: #6c757d;'>Built with ❤️ using Streamlit & scikit-learn | IMDB Dataset</div>",
            unsafe_allow_html=True)