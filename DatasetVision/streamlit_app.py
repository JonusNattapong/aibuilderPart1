import streamlit as st
import os
import numpy as np
import pandas as pd
import json
from PIL import Image
import torch
from pathlib import Path
from config_vision import (
    OUTPUT_DIR, SUPPORTED_MODELS, DEFAULT_MODEL,
    SUPPORTED_TASKS, IMAGE_SIZE, BATCH_SIZE
)
from vision_utils import (
    VisionTaskManager,
    load_model,
    process_image,
    save_dataset
)

# Initialize session state
def init_session_state():
    if 'task_manager' not in st.session_state:
        st.session_state.task_manager = None
    if 'current_task' not in st.session_state:
        st.session_state.current_task = None
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'preview_images' not in st.session_state:
        st.session_state.preview_images = []

def display_image_preview(image, task_output, idx):
    """Display image and its task output."""
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption=f"Image {idx+1}", use_column_width=True)
    with col2:
        st.json(task_output)

def main():
    st.set_page_config(page_title="Vision Dataset Generator", layout="wide")
    st.title("Vision Dataset Generator")

    # Initialize state
    init_session_state()

    # Sidebar configuration
    st.sidebar.header("Task Configuration")
    
    # Task selection
    task = st.sidebar.selectbox(
        "Select Vision Task",
        SUPPORTED_TASKS.keys(),
        format_func=lambda x: SUPPORTED_TASKS[x]['name']
    )

    if task != st.session_state.current_task:
        st.session_state.current_task = task
        st.session_state.task_manager = VisionTaskManager(task)
        st.session_state.results = []
        st.session_state.preview_images = []

    # Model selection
    model_name = st.sidebar.selectbox(
        "Select Model",
        SUPPORTED_MODELS.get(task, [DEFAULT_MODEL]),
        index=0
    )

    # Task-specific settings
    task_config = {}
    if task in SUPPORTED_TASKS:
        st.sidebar.subheader("Task Settings")
        for param, details in SUPPORTED_TASKS[task].get('parameters', {}).items():
            if details['type'] == 'number':
                task_config[param] = st.sidebar.number_input(
                    details['name'],
                    min_value=details.get('min', 0),
                    max_value=details.get('max', 100),
                    value=details.get('default', 0)
                )
            elif details['type'] == 'select':
                task_config[param] = st.sidebar.selectbox(
                    details['name'],
                    options=details['options'],
                    index=details['options'].index(details.get('default'))
                )
            elif details['type'] == 'boolean':
                task_config[param] = st.sidebar.checkbox(
                    details['name'],
                    value=details.get('default', False)
                )

    # Main content
    tab1, tab2, tab3 = st.tabs(["Generate", "Preview", "Export"])

    with tab1:
        st.header("Dataset Generation")

        # File upload
        uploaded_files = st.file_uploader(
            "Upload Images",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True
        )

        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} images")

            # Batch size selection
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=len(uploaded_files),
                value=min(BATCH_SIZE, len(uploaded_files))
            )

            if st.button("Process Images"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Load model
                    model = load_model(task, model_name)
                    
                    # Process images in batches
                    results = []
                    preview_images = []
                    
                    for i in range(0, len(uploaded_files), batch_size):
                        batch_files = uploaded_files[i:i + batch_size]
                        
                        # Process batch
                        batch_images = []
                        for file in batch_files:
                            image = Image.open(file)
                            preview_images.append(image)
                            processed_image = process_image(image, task)
                            batch_images.append(processed_image)

                        # Run inference
                        batch_results = st.session_state.task_manager.process_batch(
                            model, batch_images, task_config
                        )
                        results.extend(batch_results)

                        # Update progress
                        progress = (i + len(batch_files)) / len(uploaded_files)
                        progress_bar.progress(progress)
                        status_text.text(f"Processed {i + len(batch_files)}/{len(uploaded_files)} images")

                    st.session_state.results = results
                    st.session_state.preview_images = preview_images
                    st.success("Processing completed!")

                except Exception as e:
                    st.error(f"Error processing images: {str(e)}")

    with tab2:
        st.header("Results Preview")
        
        if not st.session_state.results:
            st.info("No results to preview. Generate data first.")
            return

        # Display results
        for idx, (image, result) in enumerate(zip(st.session_state.preview_images, st.session_state.results)):
            st.subheader(f"Result {idx+1}")
            display_image_preview(image, result, idx)

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
            filename = st.text_input("Filename", "vision_dataset")

        if st.button("Export Dataset"):
            try:
                # Prepare export data with metadata
                export_data = []
                for image, result in zip(st.session_state.preview_images, st.session_state.results):
                    data_point = {
                        "task": task,
                        "model": model_name,
                        "config": task_config,
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