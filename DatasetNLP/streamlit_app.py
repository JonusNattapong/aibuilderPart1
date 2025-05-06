import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
from nlp_utils import (
    NLPTaskManager,
    load_model,
    save_dataset,
    process_text,
    get_supported_tasks,
    get_supported_models
)
from config import (
    OUTPUT_DIR, BATCH_SIZE, MAX_TEXT_LENGTH,
    SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE
)

# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    if 'task_manager' not in st.session_state:
        st.session_state.task_manager = None
    if 'current_task' not in st.session_state:
        st.session_state.current_task = None
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'edited_results' not in st.session_state:
        st.session_state.edited_results = {}

def display_result_editor(input_text: str, result: dict, index: int):
    """Display editor for NLP task result."""
    st.subheader(f"Result {index + 1}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.text_area("Input Text", input_text, disabled=True, key=f"input_{index}")
    
    with col2:
        if st.session_state.current_task == "text_generation":
            edited = st.text_area(
                "Generated Text",
                st.session_state.edited_results.get(index, result.get("generated_text", "")),
                key=f"result_{index}"
            )
            st.session_state.edited_results[index] = edited
            
        elif st.session_state.current_task == "token_classification":
            st.json(result.get("entities", []))
            
        elif st.session_state.current_task == "text_classification":
            st.json(result.get("labels", []))
            
        else:
            st.json(result)

def main():
    st.set_page_config(page_title="NLP Dataset Generator", layout="wide")
    st.title("NLP Dataset Generator")

    # Initialize state
    init_session_state()

    # Sidebar configuration
    st.sidebar.header("Task Configuration")
    
    # Task selection
    tasks = get_supported_tasks()
    task = st.sidebar.selectbox(
        "Select NLP Task",
        list(tasks.keys()),
        format_func=lambda x: tasks[x]["name"]
    )

    if task != st.session_state.current_task:
        st.session_state.current_task = task
        st.session_state.task_manager = NLPTaskManager(task)
        st.session_state.results = []
        st.session_state.edited_results = {}

    # Model selection
    models = get_supported_models(task)
    model_name = st.sidebar.selectbox("Select Model", models)

    # Language selection
    language = st.sidebar.selectbox(
        "Language",
        SUPPORTED_LANGUAGES,
        index=SUPPORTED_LANGUAGES.index(DEFAULT_LANGUAGE)
    )

    # Task-specific settings
    task_config = {}
    if task in tasks:
        st.sidebar.subheader("Task Settings")
        for param, details in tasks[task].get("parameters", {}).items():
            if details["type"] == "number":
                task_config[param] = st.sidebar.number_input(
                    details["name"],
                    min_value=details.get("min", 0),
                    max_value=details.get("max", 100),
                    value=details.get("default", 0)
                )
            elif details["type"] == "select":
                task_config[param] = st.sidebar.selectbox(
                    details["name"],
                    options=details["options"],
                    index=details["options"].index(details.get("default"))
                )
            elif details["type"] == "boolean":
                task_config[param] = st.sidebar.checkbox(
                    details["name"],
                    value=details.get("default", False)
                )

    # Main content area
    tab1, tab2, tab3 = st.tabs(["Generate", "Review", "Export"])

    with tab1:
        st.header("Dataset Generation")

        # Input method selection
        input_method = st.radio(
            "Input Method",
            ["Text Input", "File Upload"],
            horizontal=True
        )

        texts_to_process = []

        if input_method == "Text Input":
            text_input = st.text_area(
                "Enter Text",
                height=200,
                help=f"Maximum {MAX_TEXT_LENGTH} characters"
            )
            if text_input:
                texts_to_process = [text_input]

        else:
            uploaded_file = st.file_uploader(
                "Upload Text File",
                type=["txt", "csv", "json", "jsonl"]
            )
            
            if uploaded_file:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    text_column = st.selectbox("Select Text Column", df.columns)
                    texts_to_process = df[text_column].tolist()
                else:
                    content = uploaded_file.read().decode()
                    texts_to_process = [line.strip() for line in content.split('\n') if line.strip()]

        if texts_to_process:
            st.write(f"Found {len(texts_to_process)} texts to process")

            # Batch size selection
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=len(texts_to_process),
                value=min(BATCH_SIZE, len(texts_to_process))
            )

            if st.button("Process Texts"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Load model
                    model = load_model(task, model_name)
                    
                    # Process texts in batches
                    results = []
                    for i in range(0, len(texts_to_process), batch_size):
                        batch = texts_to_process[i:i + batch_size]
                        
                        # Pre-process batch
                        processed_texts = [
                            process_text(text, task, language)
                            for text in batch
                        ]

                        # Run model inference
                        batch_results = st.session_state.task_manager.process_batch(
                            model, processed_texts, task_config
                        )
                        results.extend(batch_results)

                        # Update progress
                        progress = (i + len(batch)) / len(texts_to_process)
                        progress_bar.progress(progress)
                        status_text.text(f"Processed {i + len(batch)}/{len(texts_to_process)} texts")

                    st.session_state.results = list(zip(texts_to_process, results))
                    st.success("Processing completed!")

                except Exception as e:
                    st.error(f"Error processing texts: {str(e)}")

    with tab2:
        st.header("Review Results")
        
        if not st.session_state.results:
            st.info("No results to review. Generate data first.")
            return

        # Filter options based on task
        if task in ["text_classification", "token_classification"]:
            show_uncertain = st.checkbox("Show Only Uncertain Predictions", value=False)
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.8
            )

            filtered_results = [
                (i, text, result) for i, (text, result) in enumerate(st.session_state.results)
                if not show_uncertain or any(
                    score < confidence_threshold 
                    for item in result.get("labels" if task == "text_classification" else "entities", [])
                    for score in [item.get("confidence", 1.0)]
                )
            ]
        else:
            filtered_results = list(enumerate(st.session_state.results))

        # Display results
        for idx, text, result in filtered_results:
            display_result_editor(text, result, idx)

    with tab3:
        st.header("Export Dataset")
        
        if not st.session_state.results:
            st.info("No data to export. Generate data first.")
            return

        # Export options
        col1, col2 = st.columns(2)
        with col1:
            format_type = st.selectbox(
                "Export Format",
                ["JSON", "CSV", "JSONL"]
            )
        with col2:
            filename = st.text_input("Filename", "nlp_dataset")

        if st.button("Export Dataset"):
            try:
                # Prepare export data
                export_data = []
                for text, result in st.session_state.results:
                    data_point = {
                        "task": task,
                        "model": model_name,
                        "language": language,
                        "config": task_config,
                        "input": text,
                        "result": result
                    }
                    export_data.append(data_point)

                # Save dataset
                output_path = save_dataset(
                    export_data,
                    filename,
                    format_type,
                    OUTPUT_DIR
                )
                st.success(f"Dataset exported to: {output_path}")

            except Exception as e:
                st.error(f"Error exporting dataset: {str(e)}")

if __name__ == "__main__":
    main()