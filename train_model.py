import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

# Create folder if not exists
if not os.path.exists("models"):
    os.makedirs("models")

# Load
df = pd.read_csv('Tweets.csv')

# Simple Clean
def simple_clean(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

df['clean'] = df['text'].apply(simple_clean)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df['clean'])
y = df['airline_sentiment']

# Train
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save into the 'models' folder
pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("Files saved successfully in /models folder.")
