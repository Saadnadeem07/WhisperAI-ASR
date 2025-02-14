import streamlit as st
import whisper
import tempfile
import os
import torch
import soundfile as sf
import numpy as np
from transformers import pipeline
from gtts import gTTS
from deep_translator import GoogleTranslator
from pydub import AudioSegment
from fpdf import FPDF  # For exporting to PDF

# ✅ Custom Modern UI Theme (Blue-Neon)
st.markdown("""
    <style>
    body { background-color: #0D1B2A; color: #00A6FB; font-family: 'Arial', sans-serif; }
    .stTextInput, .stFileUploader, .stTextArea, .stSelectbox, .stButton > button {
        background-color: #1B263B;
        color: white;
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: #00A6FB;
        color: black;
    }
    .result-card {
        background-color: #112D4E;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        box-shadow: 2px 2px 15px rgba(0, 166, 251, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# ✅ Load Whisper Model
@st.cache_resource
def load_model():
    return whisper.load_model("small", device="cpu")

model = load_model()

# ✅ Summarization Model (Using BART)
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# ✅ Export Function
def export_text(text, file_type="txt"):
    if file_type == "txt":
        with open("transcription.txt", "w") as f:
            f.write(text)
        return "transcription.txt"
    elif file_type == "pdf":
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(190, 10, text)
        pdf.output("transcription.pdf")
        return "transcription.pdf"
    return None

# ✅ Upload Audio File
st.markdown("<h1 style='text-align: center;'>🎙️ Whisper AI - Transcription & Summarization</h1>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("📂 Upload your audio file", type=["mp3", "wav", "m4a"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/mp3")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_audio_path = temp_file.name

    if st.button("📝 Transcribe & Summarize"):
        with st.spinner("Transcribing... Please wait."):
            try:
                result = model.transcribe(temp_audio_path)
                transcribed_text = result['text']
                summary_text = summarizer(transcribed_text, max_length=50, min_length=20, do_sample=False)[0]['summary_text']
                
                st.markdown("### 📜 Transcribed Text:")
                st.text_area("", transcribed_text, height=120)
                
                st.markdown("### 📄 Summarized Text:")
                st.text_area("", summary_text, height=80)
                
                # ✅ Export Options
                export_type = st.selectbox("📂 Export As", ["TXT", "PDF"])
                if st.button("💾 Download File"):
                    file_path = export_text(transcribed_text, export_type.lower())
                    with open(file_path, "rb") as f:
                        st.download_button("📥 Download", f, file_name=file_path)
                
            except Exception as e:
                st.error(f"⚠ Transcription Failed: {str(e)}")
    
    os.remove(temp_audio_path)