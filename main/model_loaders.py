import easyocr
import streamlit as st
from transformers import pipeline


@st.cache_resource
def get_asr_pipe():
    return pipeline("automatic-speech-recognition",
                    model="openai/whisper-small")

@st.cache_resource
def get_chatbot_pipe():
    return pipeline("text-generation",
                    model="meta-llama/Llama-3.1-8B-Instruct")

@st.cache_resource
def get_easyocr_reader():
    return easyocr.Reader(['ja', 'en'])
