import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
from dataset_utils import (
    DatasetManager,
    load_dataset,
    save_dataset,
    get_supported_formats,
    get_supported_tasks,
    validate_dataset
)
from config import (
    OUTPUT_DIR, BATCH_SIZE, CSV_CONFIG, PARQUET_CONFIG,
    SUPPORTED_FORMATS, DEFAULT_FORMAT
)

# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    if 'dataset_manager' not in st.session_state:
        st.session_state.dataset_manager = None
    if 'current_dataset' not in st.session_state:
        st.session_state.current_dataset = None
    if 'edited_data' not in st.session_state:
        st.session_state.edited_data = {}
    if 'validation_results' not in st.session_state:
        st.session_state.validation_results = {}

def display_dataset_preview(df: pd.DataFrame, format_type: str):
    """Display dataset preview with format-specific visualizations."""
    st.subheader("Dataset Preview")
    
    # Basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    # Column info
    st.write("Column Information:")
    column_info = pd.DataFrame({
        'Type': df.dtypes,
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Unique Values': df.nunique()
    })
    st.dataframe(column_info)

    # Data preview
    st.write("Data Sample:")
    st.dataframe(df.head())

    # Format-specific visualizations
    if format_type == "CSV":
        for col in df.select_dtypes(include=['number']).columns:
            st.write(f"Distribution of {col}:")
            st.line_chart(df[col].value_counts())
    elif format_type == "PARQUET":
        if 'timestamp' in df.columns:
            st.write("Temporal Distribution:")
            st.line_chart(df.groupby('timestamp').size())

def main():
    st.set_page_config(page_title="Dataset Generator", layout="wide")
    st.title("Dataset Generator")

    # Initialize state
    init_session_state()

    # Sidebar configuration
    st.sidebar.header("Dataset Configuration")
    
    # Format selection
    format_type = st.sidebar.selectbox(
        "Select Format",
        SUPPORTED_FORMATS,
        index=SUPPORTED_FORMATS.index(DEFAULT_FORMAT)
    )

    # Task selection
    tasks = get_supported_tasks(format_type)
    task = st.sidebar.selectbox(
        "Select Task",
        list(tasks.keys()),
        format_func=lambda x: tasks[x]["name"]
    )

    # Format-specific settings
    config = CSV_CONFIG if format_type == "CSV" else PARQUET_CONFIG
    task_config = {}
    
    if task in tasks:
        st.sidebar.subheader("Task Settings")
        for param, details in tasks[task].get("parameters", {}).items():
            if details["type"] == "number":
                task_config[param] = st.sidebar.number_input(
                    details["name"],
                    min_value=details.get("min", 0),
                    max_value=details.get("max", 1000),
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

    # Main content
    tab1, tab2, tab3 = st.tabs(["Generate", "Review", "Export"])

    with tab1:
        st.header("Dataset Generation")

        # Dataset manager initialization
        if not st.session_state.dataset_manager or \
           st.session_state.dataset_manager.format_type != format_type:
            st.session_state.dataset_manager = DatasetManager(format_type)

        # Generation settings
        col1, col2 = st.columns(2)
        with col1:
            num_rows = st.number_input(
                "Number of Rows",
                min_value=1,
                max_value=1000000,
                value=1000
            )
        with col2:
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=num_rows,
                value=min(BATCH_SIZE, num_rows)
            )

        if st.button("Generate Dataset"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # Generate dataset
                df = st.session_state.dataset_manager.generate_dataset(
                    task, num_rows, task_config,
                    lambda progress: progress_bar.progress(progress)
                )
                
                st.session_state.current_dataset = df
                st.success("Dataset generation completed!")
                
                # Display preview
                display_dataset_preview(df, format_type)

            except Exception as e:
                st.error(f"Error generating dataset: {str(e)}")

    with tab2:
        st.header("Dataset Review")
        
        if st.session_state.current_dataset is None:
            st.info("No dataset to review. Generate data first.")
            return

        # Validation
        if st.button("Validate Dataset"):
            validation_results = validate_dataset(
                st.session_state.current_dataset,
                format_type,
                task_config
            )
            
            if validation_results["is_valid"]:
                st.success("Dataset validation passed!")
            else:
                st.error("Dataset validation failed!")
                st.write("Issues found:")
                for issue in validation_results["issues"]:
                    st.write(f"- {issue}")

        # Data editor
        st.subheader("Edit Data")
        edited_df = st.data_editor(
            st.session_state.current_dataset,
            num_rows="dynamic"
        )
        st.session_state.current_dataset = edited_df

    with tab3:
        st.header("Export Dataset")
        
        if st.session_state.current_dataset is None:
            st.info("No data to export. Generate dataset first.")
            return

        # Export options
        col1, col2 = st.columns(2)
        with col1:
            export_format = st.selectbox(
                "Export Format",
                get_supported_formats()
            )
        with col2:
            filename = st.text_input("Filename", f"dataset_{task}")

        if st.button("Export Dataset"):
            try:
                # Save dataset
                output_path = save_dataset(
                    st.session_state.current_dataset,
                    filename,
                    export_format,
                    OUTPUT_DIR
                )
                st.success(f"Dataset exported to: {output_path}")

                # Download button
                with open(output_path, 'rb') as f:
                    st.download_button(
                        "Download Dataset",
                        f,
                        file_name=os.path.basename(output_path)
                    )

            except Exception as e:
                st.error(f"Error exporting dataset: {str(e)}")

if __name__ == "__main__":
    main()