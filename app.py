import streamlit as st
from transformers import pipeline

# Load LLM dari Hugging Face
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

nlp_pipeline = load_model()

# UI dengan Streamlit
st.set_page_config(page_title="LLM Chatbot", page_icon="🤖")
st.title("🤖 Academic Chatbot with FLAN-T5")
st.write("Tanyakan apa saja seputar materi akademik!")

# Input user
user_input = st.text_input("Masukkan pertanyaanmu:")

if user_input:
    with st.spinner("Sedang berpikir..."):
        response = nlp_pipeline(user_input, max_length=200, do_sample=True)[0]["generated_text"]
    st.success("Jawaban:")
    st.write(response)