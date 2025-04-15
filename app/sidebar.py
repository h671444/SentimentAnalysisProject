import streamlit as st

def show():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Positive/Negative", "Fake/Real"])
    return page