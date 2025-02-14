# -*- coding: utf-8 -*-
import streamlit as st
import whisper
import tempfile
import os
import torch
import soundfile as sf
import numpy as np
from textblob import TextBlob
from transformers import pipeline
from gtts import gTTS
from deep_translator import GoogleTranslator
from pydub import AudioSegment  # For audio processing

# ✅ Custom Modern UI Styling
st.markdown("""
    <style>
    body { background-color: #121212; color: #E5C100; font-family: 'Arial', sans-serif; }
    .stTextInput, .stFileUploader, .stTextArea, .stSelectbox, .stButton > button {
        background-color: #333;
        color: white;
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: #E5C100;
        color: black;
    }
    .result-card {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 15px rgba(229, 193, 0, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# ✅ Load Whisper Model (Optimized for CPU)
@st.cache_resource
def load_model():
    return whisper.load_model("small", device="cpu")  # Running on CPU for stability

model = load_model()

# ✅ Load NLP Models
@st.cache_resource
def load_nlp_models():
    sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    summarizer = pipeline("summarization", model="t5-small")  # Lighter model for Streamlit
    return sentiment_model, emotion_model, summarizer

sentiment_model, emotion_model, summarizer = load_nlp_models()

# ✅ Preprocess Audio (Convert to Mono & 16kHz)
def preprocess_audio(audio_path):
    """Ensures audio is in the correct format before transcription."""
    try:
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        processed_audio_path = audio_path.replace(".mp3", "_processed.wav")
        audio.export(processed_audio_path, format="wav")
        return processed_audio_path
    except Exception as e:
        st.error(f"⚠ Audio Processing Failed: {str(e)}")
        return audio_path

# ✅ Sentiment Analysis
def analyze_sentiment(text):
    """Performs Sentiment Analysis."""
    if not text.strip():
        return "Neutral"
    result = sentiment_model(text)[0]
    return f"🌟 **{result['label']}** (Confidence: {result['score']:.2f})"

# ✅ Emotion Detection
def detect_emotion(text):
    """Detects Emotion in Text."""
    if not text.strip():
        return "Neutral"
    result = emotion_model(text)[0]
    return f"🎭 **{result['label']}** (Confidence: {result['score']:.2f})"

# ✅ Text Summarization
def summarize_text(text):
    """Generates a high-quality summary."""
    if len(text.split()) < 30:
        return "⚠ Text too short for summarization."
    return summarizer(text, max_length=50, min_length=20, do_sample=False)[0]['summary_text']

# ✅ Translation
def translate_text(text, target_language):
    """Translates transcribed text."""
    try:
        return GoogleTranslator(source="auto", target=target_language).translate(text)
    except Exception as e:
        return f"⚠ Translation Error: {str(e)}"

# ✅ Text-to-Speech
def text_to_speech(text, lang="en"):
    """Converts text to speech."""
    try:
        output_path = "tts_output.mp3"
        tts = gTTS(text=text, lang=lang)
        tts.save(output_path)
        return output_path
    except:
        return None

# ✅ Session Storage
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""
    st.session_state.sentiment_result = ""
    st.session_state.emotion_result = ""
    st.session_state.summary_result = ""
    st.session_state.translated_text = ""

# ✅ Upload Audio File
st.markdown("<h1 style='text-align: center;'>🎙️ Whisper AI - Voice to Google Search with Analysis</h1>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("📂 Upload your audio file", type=["mp3", "wav", "m4a"])

# ✅ Language Selection
language_mapping = {"English": "en", "Urdu": "ur", "Spanish": "es", "French": "fr", "Hindi": "hi"}
language = st.selectbox("🌍 Select Language for Transcription", list(language_mapping.keys()), index=0)
language_code = language_mapping[language]

# ✅ Translation Selection
translation_mapping = {"No Translation": None, "English": "en", "Urdu": "ur", "Spanish": "es", "French": "fr", "Hindi": "hi", "German": "de", "Chinese": "zh-CN"}
translation_lang = st.selectbox("🌍 Translate Transcription To:", list(translation_mapping.keys()), index=0)

if uploaded_file:
    st.audio(uploaded_file, format="audio/mp3")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_audio_path = temp_file.name

    # ✅ Preprocess Audio
    temp_audio_path = preprocess_audio(temp_audio_path)

    if st.button("📝 Transcribe & Analyze"):
        with st.spinner(f"Transcribing in {language}... Please wait."):
            try:
                result = model.transcribe(temp_audio_path, language=language_code)
                st.session_state.transcribed_text = result['text']
                st.session_state.sentiment_result = analyze_sentiment(st.session_state.transcribed_text)
                st.session_state.emotion_result = detect_emotion(st.session_state.transcribed_text)
                st.session_state.summary_result = summarize_text(st.session_state.transcribed_text)

                if translation_mapping[translation_lang]:
                    st.session_state.translated_text = translate_text(st.session_state.transcribed_text, translation_mapping[translation_lang])

            except Exception as e:
                st.error(f"⚠ Transcription Failed: {str(e)}")

    # ✅ Display Results
    if st.session_state.transcribed_text:
        st.markdown("<h2>📝 Results</h2>", unsafe_allow_html=True)
        st.text_area("📜 **Transcribed Text:**", st.session_state.transcribed_text, height=120)
        st.write("💬 **Sentiment Analysis:**", st.session_state.sentiment_result)
        st.write("🎭 **Emotion Analysis:**", st.session_state.emotion_result)
        st.text_area("📄 **Summarized Text:**", st.session_state.summary_result, height=80)

        if translation_mapping[translation_lang]:
            st.text_area("🌍 **Translated Text:**", st.session_state.translated_text, height=80)

        if st.button("🔍 Search on Google"):
            search_query = st.session_state.transcribed_text.replace(" ", "+")
            google_url = f"https://www.google.com/search?q={search_query}"
            st.markdown(f'<a href="{google_url}" target="_blank">Click here to search</a>', unsafe_allow_html=True)

    os.remove(temp_audio_path)
