import streamlit as st
import pandas as pd
import json
import os
import asyncio
from config import (
    OUTPUT_DIR, LOG_DIR, LOG_FILENAME, BATCH_SIZE,
    SUPPORTED_FORMATS, DEFAULT_FORMAT
)
from shared_utils import TranslationManager

# Initialize translation manager
@st.cache_resource
def get_translation_manager():
    """Get or create TranslationManager instance."""
    return TranslationManager()

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

async def main():
    st.set_page_config(page_title="Dataset Translation", layout="wide")
    st.title("Dataset Translation Tool")

    # Initialize session state
    init_session_state()

    try:
        translation_manager = get_translation_manager()
    except ValueError as e:
        st.error(f"Setup error: {str(e)}")
        st.info("Please set your DeepL API key in .env file")
        return

    # Sidebar configuration
    st.sidebar.header("Translation Settings")
    
    # Language selection
    languages = translation_manager.get_supported_languages()
    source_lang, target_lang = st.sidebar.columns(2)
    with source_lang:
        src_lang = st.selectbox("Source Language", languages, index=0)
    with target_lang:
        tgt_lang = st.selectbox("Target Language", languages, index=1)

    # Main interface
    tab1, tab2, tab3 = st.tabs(["Translation", "Review", "Export"])

    with tab1:
        st.header("Dataset Translation")
        
        # File upload
        uploaded_file = st.file_uploader("Choose a dataset file", type=["csv", "json", "jsonl"])
        
        if uploaded_file:
            try:
                dataset = translation_manager.load_dataset(uploaded_file)
                st.success(f"Loaded {len(dataset)} entries from dataset")
                
                # Display dataset preview
                st.subheader("Dataset Preview")
                st.dataframe(pd.DataFrame(dataset[:5]))

                # Translation settings
                col1, col2 = st.columns(2)
                with col1:
                    batch_size = st.number_input("Batch Size", min_value=1, max_value=100, value=BATCH_SIZE)
                with col2:
                    text_column = st.selectbox("Text Column", dataset[0].keys())

                # Start translation
                if st.button("Start Translation"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(progress: float):
                        progress_bar.progress(progress)
                        status_text.text(f"Progress: {progress:.1%}")

                    try:
                        texts = [item[text_column] for item in dataset]
                        translations = await translation_manager.translate_batch(
                            texts, src_lang, tgt_lang, batch_size,
                            progress_callback=update_progress
                        )
                        
                        st.session_state.translations = translations
                        st.success("Translation completed!")
                        
                        # Show statistics
                        total = len(translations)
                        needs_review = sum(1 for t in translations if t.get("needs_review", False))
                        errors = sum(1 for t in translations if "error" in t)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Translated", total)
                        with col2:
                            st.metric("Needs Review", needs_review)
                        with col3:
                            st.metric("Errors", errors)
                            
                    except Exception as e:
                        st.error(f"Translation error: {str(e)}")
            
            except Exception as e:
                st.error(f"Error loading dataset: {str(e)}")

    with tab2:
        st.header("Translation Review")
        if not st.session_state.translations:
            st.info("No translations to review yet. Use the Translation tab first.")
            return

        # Filter options
        show_needs_review = st.checkbox("Show Only Needs Review", value=True)
        
        # Filter translations
        filtered_translations = [
            (i, t) for i, t in enumerate(st.session_state.translations)
            if not show_needs_review or t.get("needs_review", False)
        ]

        if not filtered_translations:
            st.info("No translations to review with current filters")
            return

        # Navigation
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("Previous", disabled=st.session_state.current_index <= 0):
                st.session_state.current_index -= 1
        with col2:
            st.write(f"Reviewing {st.session_state.current_index + 1} of {len(filtered_translations)}")
        with col3:
            if st.button("Next", disabled=st.session_state.current_index >= len(filtered_translations) - 1):
                st.session_state.current_index += 1

        # Display current translation
        idx, trans = filtered_translations[st.session_state.current_index]
        display_translation_editor(
            trans["original"],
            trans["translated"],
            idx
        )

        # Show validation issues if any
        if trans.get("issues"):
            st.warning("Validation Issues:")
            for issue in trans["issues"]:
                st.write(f"- {issue}")

    with tab3:
        st.header("Export Translations")
        if not st.session_state.translations:
            st.info("No translations available to export.")
            return

        # Export options
        output_format = st.selectbox("Output Format", SUPPORTED_FORMATS, 
                                   index=SUPPORTED_FORMATS.index(DEFAULT_FORMAT))
        filename = st.text_input("Output Filename", "translated_dataset")

        if st.button("Export"):
            try:
                # Prepare export data
                export_data = []
                for i, trans in enumerate(st.session_state.translations):
                    export_data.append({
                        "original": trans["original"],
                        "translated": st.session_state.edited_translations.get(i, trans["translated"]),
                        "needs_review": trans.get("needs_review", False),
                        "issues": trans.get("issues", [])
                    })

                # Save translations
                output_path = translation_manager.save_translations(
                    export_data, filename, output_format, OUTPUT_DIR
                )
                st.success(f"Translations exported to: {output_path}")
            except Exception as e:
                st.error(f"Error exporting translations: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())