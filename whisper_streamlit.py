# -*- coding: utf-8 -*-
import streamlit as st
import whisper
import tempfile
import os
import torch
import soundfile as sf
import numpy as np
from collections import Counter
from transformers import pipeline
from pydub import AudioSegment
from sklearn.feature_extraction.text import CountVectorizer
from fuzzywuzzy import fuzz
import spacy
import spacy.cli  # Import spacy.cli globally to avoid local shadowing issues

# ✅ Ensure FFmpeg Path
# Adjust the path if necessary for your deployment environment.
AudioSegment.converter = "/usr/bin/ffmpeg"

# ✅ Load Whisper Model (Optimized for CPU)
@st.cache_resource
def load_model():
    return whisper.load_model("base", device="cpu")  # Running on CPU for stability

model = load_model()

# ✅ Load NLP Models with spaCy model download check
@st.cache_resource
def load_nlp_models():
    try:
        ner_model = spacy.load("en_core_web_sm")
    except Exception as e:
        st.warning(f"spaCy model 'en_core_web_sm' not found or failed to load due to error: {e}. Downloading the model...")
        spacy.cli.download("en_core_web_sm")
        ner_model = spacy.load("en_core_web_sm")
    return ner_model

ner_model = load_nlp_models()

# ✅ Preprocess Audio (Convert to Mono & 16kHz)
def preprocess_audio(audio_path):
    try:
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        base, _ = os.path.splitext(audio_path)
        processed_audio_path = base + "_processed.wav"
        audio.export(processed_audio_path, format="wav")
        return processed_audio_path
    except Exception as e:
        st.error(f"⚠ Audio Processing Failed: {str(e)}")
        return audio_path

# ✅ Keyword Extraction
def extract_keywords(text, num_keywords=5):
    vectorizer = CountVectorizer(stop_words="english")
    word_counts = vectorizer.fit_transform([text])
    words = vectorizer.get_feature_names_out()
    counts = word_counts.toarray().sum(axis=0)
    keyword_counts = sorted(zip(words, counts), key=lambda x: x[1], reverse=True)
    return [word for word, count in keyword_counts[:num_keywords]]

# ✅ Named Entity Recognition (NER)
def named_entity_recognition(text):
    doc = ner_model(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities if entities else "No entities found."

# ✅ Phonetic Similarity Check
def phonetic_similarity(text, reference_text):
    return fuzz.ratio(text, reference_text)

# ✅ Word Frequency Analysis
def word_frequency_analysis(text):
    words = text.lower().split()
    word_counts = Counter(words)
    return word_counts.most_common(5)

# ✅ Session Storage Initialization
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""
    st.session_state.keyword_result = ""
    st.session_state.ner_result = ""
    st.session_state.similarity_score = ""
    st.session_state.word_frequencies = ""

# ✅ Streamlit App Interface
st.markdown("<h1 style='text-align: center;'>🎙️ Whisper AI - Audio Processing & Analysis</h1>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("📂 Upload your audio file", type=["mp3", "wav", "m4a"])
reference_text = st.text_area("✏️ Enter Reference Text for Phonetic Similarity Check:", "")

if uploaded_file:
    st.audio(uploaded_file, format="audio/mp3")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_audio_path = temp_file.name

    temp_audio_path = preprocess_audio(temp_audio_path)
    
    if st.button("📝 Transcribe & Analyze"):
        with st.spinner("Transcribing... Please wait."):
            try:
                result = model.transcribe(temp_audio_path)
                st.session_state.transcribed_text = result['text']
                st.session_state.keyword_result = extract_keywords(st.session_state.transcribed_text)
                st.session_state.ner_result = named_entity_recognition(st.session_state.transcribed_text)
                st.session_state.word_frequencies = word_frequency_analysis(st.session_state.transcribed_text)
                if reference_text:
                    st.session_state.similarity_score = phonetic_similarity(
                        st.session_state.transcribed_text, reference_text
                    )
            except Exception as e:
                st.error(f"⚠ Transcription Failed: {str(e)}")
    
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)

    if st.session_state.transcribed_text:
        st.markdown("<h2>📝 Results</h2>", unsafe_allow_html=True)
        st.text_area("📜 Transcribed Text:", st.session_state.transcribed_text, height=120)
        st.write("🔑 Extracted Keywords:", ", ".join(st.session_state.keyword_result))
        st.write("🏷 Named Entities:", st.session_state.ner_result)
        st.write("📊 Word Frequency Analysis:", st.session_state.word_frequencies)
        if reference_text:
            st.write(f"🎵 Phonetic Similarity with Reference: {st.session_state.similarity_score}%")
