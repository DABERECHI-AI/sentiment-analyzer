
import streamlit as st
import joblib
import gzip
import os

@st.cache_resource
def load_model():
    try:
        with gzip.open('sentiment_model.joblib.gz', 'rb') as f:
            return joblib.load(f)
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_model()

if model:
    st.title("Sentiment Analyzer")
    user_input = st.text_input("Enter text:")
    if st.button("Analyze") and user_input:
        prediction = model.predict([user_input])[0]
        st.success("Positive 😊" if prediction == 1 else "Negative 😠")
