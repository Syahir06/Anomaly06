import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
from collections import Counter
import re

# -----------------------
# Page Config
# -----------------------
st.set_page_config(page_title="Advanced Airline Sentiment", layout="wide")

st.title("✈️ Advanced Airline Sentiment Insights")

# -----------------------
# Load Dataset & Models
# -----------------------
DATA_URL = "https://raw.githubusercontent.com/s22a0064-AinMaisarah/syahir/main/Tweets.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    # Basic cleaning: remove URLs for better text analysis later
    df['clean_text'] = df['text'].str.replace(r'http\S+', '', regex=True)
    return df

@st.cache_resource
def load_models():
    # Explicitly using a model that provides scores
    sentiment_pipe = pipeline("sentiment-analysis")
    emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    return sentiment_pipe, emotion_pipe

df = load_data()
sentiment_model, emotion_model = load_models()

# -----------------------
# Sidebar Controls
# -----------------------
st.sidebar.header("Analysis Settings")
sample_size = st.sidebar.slider("Sample Size", 50, 500, 200)
min_confidence = st.sidebar.slider("Min. Confidence Threshold", 0.0, 1.0, 0.5)

data_sample = df.sample(sample_size, random_state=42).copy()

# -----------------------
# Enhanced NLP Processing
# -----------------------
with st.spinner("Processing deep insights..."):
    # 1. Sentiment + Confidence Score
    sent_results = sentiment_model(data_sample['clean_text'].tolist())
    data_sample["Sentiment"] = [res['label'] for res in sent_results]
    data_sample["Confidence"] = [round(res['score'], 3) for res in sent_results]
    
    # 2. Emotion
    emot_results = emotion_model(data_sample['clean_text'].tolist())
    data_sample["Emotion"] = [res['label'] for res in emot_results]
    
    # 3. Text Length Parameter
    data_sample["Tweet_Length"] = data_sample['text'].apply(len)

# Filter by confidence threshold set in sidebar
data_sample = data_sample[data_sample["Confidence"] >= min_confidence]

# -----------------------
# UI Layout: Top Metrics
# -----------------------
col1, col2, col3 = st.columns(3)
col1.metric("Average Confidence", f"{data_sample['Confidence'].mean():.2%}")
col2.metric("Avg. Tweet Length", f"{int(data_sample['Tweet_Length'].mean())} chars")
col3.metric("Filtered Samples", len(data_sample))

# -----------------------
# Visualizations
# -----------------------
tab1, tab2, tab3 = st.tabs(["📈 Distributions", "📊 Correlations", "🔤 Keyword Analysis"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        fig_sent = px.pie(data_sample, names="Sentiment", title="Sentiment Split", hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_sent, use_container_width=True)
    with c2:
        fig_emot = px.bar(data_sample['Emotion'].value_counts(), title="Emotion Breakdown", labels={'value':'Count', 'index':'Emotion'})
        st.plotly_chart(fig_emot, use_container_width=True)

with tab2:
    # New Parameter: Correlation between length and confidence
    fig_scatter = px.scatter(
        data_sample, 
        x="Tweet_Length", 
        y="Confidence", 
        color="Sentiment",
        title="Confidence vs. Tweet Length",
        hover_data=['text']
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    # New Parameter: Word Frequency
    st.subheader("Top Keywords by Sentiment")
    target_sent = st.selectbox("Select Sentiment for Keywords", data_sample["Sentiment"].unique())
    
    words = " ".join(data_sample[data_sample["Sentiment"] == target_sent]['clean_text']).lower()
    # Very basic tokenization (removing common stop words like 'the', 'to', etc. would be next step)
    filtered_words = [w for w in re.findall(r'\w+', words) if len(w) > 3]
    word_freq = Counter(filtered_words).most_common(15)
    
    word_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
    fig_words = px.bar(word_df, x='Frequency', y='Word', orientation='h', title=f"Common words in {target_sent} tweets")
    st.plotly_chart(fig_words, use_container_width=True)

# -----------------------
# Data Explorer
# -----------------------
st.subheader("🔍 Detailed Explorer")
st.dataframe(data_sample[["text", "Sentiment", "Confidence", "Emotion", "Tweet_Length"]], use_container_width=True)
