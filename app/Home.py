import streamlit as st

st.set_page_config(
    page_title="Sentiment Analysis App",
    # page_icon="📝",
    # initial_sidebar_state="expanded",
    menu_items=None
)

st.title("Welcome to the Text Analysis App!")
st.write("Choose an analysis tool from the sidebar to get started.")
st.sidebar.success("Select an analysis above.")

