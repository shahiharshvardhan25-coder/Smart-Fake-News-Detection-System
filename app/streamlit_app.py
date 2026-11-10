
import streamlit as st
import pandas as pd
import os
import re
import joblib

# Find the main project folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load the saved model and TF-IDF vectorizer
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

model, tfidf = load_model()

# Preprocess new text exactly as in training
def preprocess_text(title, article):
    content = title + " " + article
    content = content.lower()
    content = re.sub(r"http\S+|www\S+", "", content)
    content = re.sub(r"[^a-zA-Z\s]", " ", content)
    content = re.sub(r"\s+", " ", content).strip()
    return content

# Streamlit page
st.set_page_config(page_title="Fake News Detection System", page_icon="📰")

st.title("📰 Smart Fake News Detection System")
st.write("Enter a news headline and article to get a model prediction.")

headline = st.text_input("News Headline")
article = st.text_area("News Article", height=250)

if st.button("Check News"):
    if not headline.strip() and not article.strip():
        st.warning("Please enter a headline or news article.")
    else:
        prepared_text = preprocess_text(headline, article)
        text_vector = tfidf.transform([prepared_text])
        prediction = model.predict(text_vector)[0]

        st.subheader("Prediction")

        if prediction == 0:
            st.error("The model classifies this news as: FAKE NEWS")
        else:
            st.success("The model classifies this news as: REAL NEWS")

        st.caption(
            "Disclaimer: This is a machine-learning text classification "
            "prediction based on the training dataset. It is not a definitive "
            "fact-check and may produce incorrect results."
        )
