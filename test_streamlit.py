import streamlit as st

st.title("Streamlit test")

audio = st.audio_input("Test microphone")

if audio:
    st.success("Microphone works")

if st.button("Test button"):
    st.success("Button works")