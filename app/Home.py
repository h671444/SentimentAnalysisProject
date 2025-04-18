import streamlit as st

st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="📝",
    initial_sidebar_state="expanded",
    menu_items=None
)

st.title("Welcome to the Sentiment Analysis App!")

# Project Background Section
st.header("Project Background")
st.write("""
This project explores different deep learning architectures for sentiment analysis on Amazon product reviews. 
We've implemented and compared three distinct neural network models:

- **CNN (Convolutional Neural Network)**: Uses convolutional layers to extract features from text sequences
- **GRU (Gated Recurrent Unit)**: A simplified RNN variant that efficiently processes sequential data
- **LSTM (Long Short-Term Memory)**: An advanced RNN architecture designed to capture long-term dependencies

Each model was trained on a dataset of Amazon product reviews and evaluated using standard metrics like accuracy, 
precision, recall, and F1-score. The models were implemented using TensorFlow/Keras and follow best practices 
for text classification tasks.
""")

# How to Use Section
st.header("How to Use")
st.write("""
1. Select an analysis tool from the sidebar
2. Choose your preferred model architecture (CNN, GRU, or LSTM)
3. Enter the text you want to analyze
4. View the sentiment prediction and confidence score
""")

st.sidebar.success("Select an analysis above.")

