import streamlit as st
import os
import sys

# Add src directory to path to allow imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.evaluate import load_artifacts, predict_sentiment

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊"
)

st.title("Sentiment Analysis")

st.write("This app uses a pre-trained model to predict the sentiment of Amazon reviews.")

st.write("Enter a review below and click the button to predict the sentiment.")

input_text = st.text_area("Enter a review:")
st.write("Select model for prediction:")
model_name = st.selectbox("Model", ["CNN", "RNN"])

if st.button("Predict"):
    if not input_text:
        st.write("Please enter a review to predict sentiment")
    elif not model_name:
        st.write("Please select a model to use for prediction")
    else:
        # Load model and tokenizer

        model_dir = model_name.lower()

        model_path = os.path.join(PROJECT_ROOT, 'models', model_dir, 'best_model.keras')
        tokenizer_path = os.path.join(PROJECT_ROOT, 'models', model_dir, 'tokenizer.pickle')
        model, tokenizer = load_artifacts(model_path, tokenizer_path)
        
        if model and tokenizer:
            # Get prediction
            sentiment, confidence = predict_sentiment(input_text, model, tokenizer)
            
            # Show results
            st.write(f"Sentiment: {sentiment}")
            st.write(f"Confidence: {confidence:.2%}")
            st.progress(confidence)



