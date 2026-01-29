import streamlit as st
import pandas as pd
import pickle
import re
import nltk
import plotly.express as px
import os
from nltk.corpus import stopwords

# --- Page Config ---
st.set_page_config(page_title="Airline Sentiment AI", page_icon="✈️", layout="wide")

# --- Resource Loading ---
@st.cache_resource
def load_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    return set(stopwords.words('english'))

@st.cache_resource
def load_assets():
    # Looking into the 'models' folder
    model_path = os.path.join("models", "model.pkl")
    vec_path = os.path.join("models", "vectorizer.pkl")
    
    if os.path.exists(model_path) and os.path.exists(vec_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(vec_path, "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    return None, None

stop_words = load_nltk()
model, vectorizer = load_assets()

# --- Helper Functions ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# --- App UI ---
st.title("✈️ Airline Tweet Sentiment Analysis")
st.markdown("Predict sentiment and analyze the distribution of airline reviews.")

# Create two tabs: Prediction and Analysis
tab1, tab2 = st.tabs(["🔍 Predict Sentiment", "📊 Dataset Analysis"])

with tab1:
    st.header("Real-time Prediction")
    if model is None:
        st.error("Error: Models not found in 'models/' folder. Please check your GitHub structure.")
    else:
        user_input = st.text_area("Enter a tweet about an airline:", height=150)
        
        if st.button("Analyze Sentiment"):
            if user_input.strip():
                cleaned = clean_text(user_input)
                vec = vectorizer.transform([cleaned])
                prediction = model.predict(vec)[0]
                
                # Display Result
                if prediction == "positive":
                    st.success(f"The sentiment is: **{prediction.upper()}** 😊")
                elif prediction == "neutral":
                    st.info(f"The sentiment is: **{prediction.upper()}** 😐")
                else:
                    st.error(f"The sentiment is: **{prediction.upper()}** 😠")
            else:
                st.warning("Please enter some text first.")

with tab2:
    st.header("Sentiment Distribution Analysis")
    if os.path.exists("Tweets.csv"):
        df = pd.read_csv("Tweets.csv")
        
        # Calculate counts
        sentiment_counts = df['airline_sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        
        # Plotly Chart
        fig = px.bar(
            sentiment_counts, 
            x='Sentiment', 
            y='Count', 
            color='Sentiment',
            color_discrete_map={'negative': '#EF553B', 'neutral': '#636EFA', 'positive': '#00CC96'},
            title="Overall Sentiment Distribution in Dataset"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show specific airline analysis
        st.subheader("Analysis by Airline")
        airline = st.selectbox("Select an Airline", df['airline'].unique())
        airline_df = df[df['airline'] == airline]
        
        fig2 = px.pie(
            airline_df, 
            names='airline_sentiment', 
            title=f"Sentiment Breakdown for {airline}",
            hole=0.4
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Upload 'Tweets.csv' to the root directory to see the analysis graph.")

st.markdown("---")
st.caption("Machine Learning Model trained on Airline Twitter Data.")
