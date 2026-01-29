import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="Airline Sentiment Analysis Dashboard",
    layout="wide"
)

st.title("✈️ Airline Tweet Sentiment Analysis")
st.markdown("Analyze customer sentiment and emotions from airline-related tweets.")

# -----------------------
# Load Dataset
# -----------------------
DATA_URL = "https://raw.githubusercontent.com/s22a0064-AinMaisarah/syahir/main/Tweets.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)

df = load_data()

# Identify tweet column safely
text_column = "text" if "text" in df.columns else df.columns[0]

st.success(f"Dataset loaded: {len(df)} tweets")

# -----------------------
# NLP Models
# -----------------------
@st.cache_resource
def load_models():
    sentiment_model = pipeline("sentiment-analysis")
    emotion_model = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=False
    )
    return sentiment_model, emotion_model

sentiment_model, emotion_model = load_models()

# -----------------------
# Sidebar Controls
# -----------------------
st.sidebar.header("Controls")
sample_size = st.sidebar.slider(
    "Number of tweets to analyze",
    min_value=50,
    max_value=min(500, len(df)),
    value=200
)

data_sample = df.sample(sample_size, random_state=42).copy()

# -----------------------
# Sentiment Analysis
# -----------------------
with st.spinner("Analyzing sentiment..."):
    data_sample["Sentiment"] = data_sample[text_column].apply(
        lambda x: sentiment_model(str(x))[0]["label"]
    )

# -----------------------
# Emotion Detection
# -----------------------
with st.spinner("Detecting emotions..."):
    data_sample["Emotion"] = data_sample[text_column].apply(
        lambda x: emotion_model(str(x))[0]["label"]
    )

# -----------------------
# Visualizations
# -----------------------
st.subheader("📊 Sentiment Distribution")

sentiment_fig = px.pie(
    data_sample,
    names="Sentiment",
    title="Sentiment Polarity of Airline Tweets",
    hole=0.4
)

st.plotly_chart(sentiment_fig, width="stretch")

# -----------------------
# Emotion Distribution (FIXED)
# -----------------------
st.subheader("😊 Emotion Distribution")

emotion_counts = (
    data_sample["Emotion"]
    .value_counts()
    .reset_index()
)

emotion_counts.columns = ["Emotion", "Count"]

emotion_fig = px.bar(
    emotion_counts,
    x="Emotion",
    y="Count",
    title="Detected Emotions in Tweets"
)

st.plotly_chart(emotion_fig, width="stretch")

# -----------------------
# Tweet Explorer
# -----------------------
st.subheader("🔍 Tweet Explorer")

selected_sentiment = st.selectbox(
    "Filter by Sentiment",
    ["All"] + sorted(data_sample["Sentiment"].unique())
)

if selected_sentiment != "All":
    filtered_df = data_sample[data_sample["Sentiment"] == selected_sentiment]
else:
    filtered_df = data_sample

st.dataframe(
    filtered_df[[text_column, "Sentiment", "Emotion"]],
    width="stretch"
)
