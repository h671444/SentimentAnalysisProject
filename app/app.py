import streamlit as st
import sidebar


page = sidebar.show()

# if page == "Positive/Negative":
    

st.title("Amazon Review Sentiment Analysis")

st.write("This app uses a pre-trained model to predict the sentiment of Amazon reviews.")

st.write("Enter a review below and click the button to predict the sentiment.")

input_text = st.text_area("Enter a review:")
st.write("Select model for prediction:")
model_name = st.selectbox("Model", ["CNN"])

if st.button("Predict"):
    if not input_text:
        st.write("Please enter a review to predict sentiment")
    elif not model_name:
        st.write("Please select a model to use for prediction")
    else:
        st.write("Predicting sentiment...")
        # TODO: Implement sentiment prediction
        # For now, just provide a placeholder response
        st.write("Sentiment prediction: This is a placeholder response.")
