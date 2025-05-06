import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
from reasoning_utils import (
    ReasoningTaskManager,
    load_model,
    save_dataset,
    get_supported_tasks,
    get_supported_models
)
from config import OUTPUT_DIR, BATCH_SIZE, MAX_PROMPT_LENGTH

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

def display_reasoning_editor(prompt: str, result: dict, index: int):
    """Display editor for reasoning task result."""
    st.subheader(f"Result {index + 1}")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.text_area("Prompt", prompt, disabled=True, key=f"prompt_{index}")
    
    with col2:
        if "reasoning_steps" in result:
            for i, step in enumerate(result["reasoning_steps"]):
                step_text = st.text_area(
                    f"Step {i+1}",
                    st.session_state.edited_results.get(f"{index}_step_{i}", step),
                    key=f"step_{index}_{i}"
                )
                st.session_state.edited_results[f"{index}_step_{i}"] = step_text
        
        final_answer = st.text_area(
            "Final Answer",
            st.session_state.edited_results.get(f"{index}_answer", result.get("answer", "")),
            key=f"answer_{index}"
        )
        st.session_state.edited_results[f"{index}_answer"] = final_answer

def main():
    st.set_page_config(page_title="Reasoning Dataset Generator", layout="wide")
    st.title("Reasoning Dataset Generator")

    # Initialize state
    init_session_state()

    # Sidebar configuration
    st.sidebar.header("Task Configuration")
    
    # Task selection
    tasks = get_supported_tasks()
    task = st.sidebar.selectbox(
        "Select Reasoning Task",
        list(tasks.keys()),
        format_func=lambda x: tasks[x]["name"]
    )

    if task != st.session_state.current_task:
        st.session_state.current_task = task
        st.session_state.task_manager = ReasoningTaskManager(task)
        st.session_state.results = []
        st.session_state.edited_results = {}

    # Model selection
    models = get_supported_models(task)
    model_name = st.sidebar.selectbox("Select Model", models)

    # Task-specific settings
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

    # Main content area
    tab1, tab2, tab3 = st.tabs(["Generate", "Review", "Export"])

    with tab1:
        st.header("Dataset Generation")

        # Input method selection
        input_method = st.radio(
            "Input Method",
            ["Direct Input", "File Upload"],
            horizontal=True
        )

        prompts_to_process = []

        if input_method == "Direct Input":
            prompt_input = st.text_area(
                "Enter Prompts (one per line)",
                height=200,
                help=f"Maximum {MAX_PROMPT_LENGTH} characters per prompt"
            )
            if prompt_input:
                prompts_to_process = [p.strip() for p in prompt_input.split("\n") if p.strip()]

        else:
            uploaded_file = st.file_uploader(
                "Upload Prompts File",
                type=["txt", "csv", "json", "jsonl"]
            )
            
            if uploaded_file:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    prompt_column = st.selectbox("Select Prompt Column", df.columns)
                    prompts_to_process = df[prompt_column].tolist()
                else:
                    content = uploaded_file.read().decode()
                    prompts_to_process = [line.strip() for line in content.split('\n') if line.strip()]

        if prompts_to_process:
            st.write(f"Found {len(prompts_to_process)} prompts to process")

            # Batch size selection
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=len(prompts_to_process),
                value=min(BATCH_SIZE, len(prompts_to_process))
            )

            if st.button("Generate Reasoning"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Load model
                    model = load_model(task, model_name)
                    
                    # Process prompts in batches
                    results = []
                    for i in range(0, len(prompts_to_process), batch_size):
                        batch = prompts_to_process[i:i + batch_size]
                        
                        # Process batch
                        batch_results = st.session_state.task_manager.process_batch(
                            model, batch, task_config
                        )
                        results.extend(batch_results)

                        # Update progress
                        progress = (i + len(batch)) / len(prompts_to_process)
                        progress_bar.progress(progress)
                        status_text.text(f"Processed {i + len(batch)}/{len(prompts_to_process)} prompts")

                    st.session_state.results = list(zip(prompts_to_process, results))
                    st.success("Processing completed!")

                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")

    with tab2:
        st.header("Review Results")
        
        if not st.session_state.results:
            st.info("No results to review. Generate reasoning first.")
            return

        # Display results
        for idx, (prompt, result) in enumerate(st.session_state.results):
            display_reasoning_editor(prompt, result, idx)
            st.markdown("---")

    with tab3:
        st.header("Export Dataset")
        
        if not st.session_state.results:
            st.info("No data to export. Generate reasoning first.")
            return

        # Export options
        col1, col2 = st.columns(2)
        with col1:
            format_type = st.selectbox(
                "Export Format",
                ["JSON", "CSV", "JSONL"]
            )
        with col2:
            filename = st.text_input("Filename", "reasoning_dataset")

        if st.button("Export Dataset"):
            try:
                # Prepare export data
                export_data = []
                for prompt, result in st.session_state.results:
                    # Get edited version if available
                    if task in ["cot", "react", "tot"]:
                        steps = []
                        for i in range(len(result.get("reasoning_steps", []))):
                            step_key = f"{st.session_state.results.index((prompt, result))}_step_{i}"
                            steps.append(st.session_state.edited_results.get(step_key, result["reasoning_steps"][i]))
                        
                        final_answer = st.session_state.edited_results.get(
                            f"{st.session_state.results.index((prompt, result))}_answer",
                            result.get("answer", "")
                        )
                        
                        export_item = {
                            "task": task,
                            "model": model_name,
                            "prompt": prompt,
                            "reasoning_steps": steps,
                            "answer": final_answer,
                            "config": task_config
                        }
                    else:
                        export_item = {
                            "task": task,
                            "model": model_name,
                            "prompt": prompt,
                            "result": result,
                            "config": task_config
                        }
                    
                    export_data.append(export_item)

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