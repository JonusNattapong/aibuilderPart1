import streamlit as st
import pandas as pd
import json
import os
from dotenv import load_dotenv
from pathlib import Path
from translation_utils import (
    setup_logger,
    load_dataset,
    translate_text,
    save_translations,
    validate_translation,
    get_supported_languages,
    detect_language
)

# Load environment variables
load_dotenv()

# Constants
OUTPUT_DIR = "DataOutput/translations"
LOGGER = setup_logger()

def init_session_state():
    """Initialize session state variables."""
    if 'translations' not in st.session_state:
        st.session_state.translations = []
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'edited_translations' not in st.session_state:
        st.session_state.edited_translations = {}
    if 'validation_results' not in st.session_state:
        st.session_state.validation_results = {}

def display_translation_editor(text: str, translated: str, index: int):
    """Display an editor for a single translation."""
    col1, col2 = st.columns(2)
    with col1:
        st.text_area("Original Text", text, disabled=True, key=f"original_{index}")
    with col2:
        edited = st.text_area(
            "Translation",
            st.session_state.edited_translations.get(index, translated),
            key=f"translation_{index}"
        )
        st.session_state.edited_translations[index] = edited