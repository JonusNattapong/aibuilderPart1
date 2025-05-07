import os
import json
import requests
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DATASET_DOWNLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "DatasetDownload")

def build_prompt(text, src_lang="en", tgt_lang="th"):
    if src_lang == "en":
        src_label = "English"
        tgt_label = "Thai"
    elif src_lang == "zh":
        src_label = "Chinese"
        tgt_label = "Thai"
    else:
        src_label = src_lang
        tgt_label = tgt_lang
    return (
        "You are an expert medical translator specializing in {src_label}-to-{tgt_label} translation for clinical, academic, and patient communication. "
        "Translate the following {src_label} medical text to {tgt_label} with maximum accuracy, clarity, and naturalness. "
        "Preserve all medical terminology, context, and nuances. If the text contains abbreviations or medical jargon, provide the most appropriate {tgt_label} equivalent or a clear explanation in parentheses. "
        "Format the output for readability and, if relevant, include {tgt_label} medical terms in bold. Do not omit any important details.\n\n"
        f"{src_label}:\n{text}\n\n{tgt_label}:"
    ).format(src_label=src_label, tgt_label=tgt_label)

def translate_medical_deepseek(text, api_key=None, src_lang="en", tgt_lang="th"):
    if api_key is None:
        api_key = DEEPSEEK_API_KEY
    if not api_key:
        return "DeepSeek API key not set."
    prompt = build_prompt(text, src_lang, tgt_lang)
    payload = json.dumps({
        "messages": [
            {
                "role": "system",
                "content": (
                    f"You are an expert medical translator specializing in {src_lang}-to-{tgt_lang} translation for clinical, academic, and patient communication. "
                    f"Translate {src_lang} medical text to {tgt_lang} with maximum accuracy, clarity, and naturalness. "
                    f"Preserve all medical terminology, context, and nuances. If the text contains abbreviations or medical jargon, provide the most appropriate {tgt_lang} equivalent or a clear explanation in parentheses. "
                    f"Format the output for readability and, if relevant, include {tgt_lang} medical terms in bold. Do not omit any important details."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "model": "deepseek-chat",
        "frequency_penalty": 0,
        "max_tokens": 2048,
        "presence_penalty": 0,
        "response_format": {"type": "text"},
        "stop": None,
        "stream": False,
        "stream_options": None,
        "temperature": 1,
        "top_p": 1,
        "tools": None,
        "tool_choice": "none",
        "logprobs": False,
        "top_logprobs": None
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, data=payload)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        if f"{tgt_lang.capitalize()}:" in content:
            return content.split(f"{tgt_lang.capitalize()}:")[-1].strip()
        return content.strip()
    except Exception as e:
        return f"Error: {e}"

st.title("DeepSeek Medical Dataset Translation (EN/ZH → TH)")

api_key = st.text_input("DeepSeek API Key", value=DEEPSEEK_API_KEY, type="password")
src_lang = st.selectbox("Source Language", options=[("English", "en"), ("Chinese", "zh")], format_func=lambda x: x[0])[1]
tgt_lang = "th"

# List datasets in DatasetDownload and allow upload
dataset_files = []
for root, dirs, files in os.walk(DATASET_DOWNLOAD_DIR):
    for file in files:
        if file.endswith(".csv") or file.endswith(".arrow") or file == "dataset_dict.json":
            dataset_files.append(os.path.join(root, file))
dataset_files = sorted(dataset_files)
st.markdown("### Select or Upload Dataset")
selected_dataset = st.selectbox("Select Dataset from DatasetDownload or upload below", options=[""] + dataset_files)
uploaded_file = st.file_uploader(
    "Or upload your own CSV/Arrow/HuggingFace dataset file (max 50GB, chunked processing for large files)",
    type=["csv", "arrow"],
    accept_multiple_files=False,
    key="csv_uploader",
    help="Supports CSV and Arrow files up to 50GB. For Arrow, upload the .arrow file directly."
)

fields = []
auto_fields = []
df = None
filename = None
ext = None

if selected_dataset or uploaded_file:
    import pandas as pd
    import pyarrow as pa
    import tempfile

    # Initialize variables
    file_for_read = None
    df_preview = None

    # Set filename and file_for_read
    if selected_dataset:
        filename = selected_dataset
        ext = os.path.splitext(filename)[-1].lower()
        file_for_read = filename
    elif uploaded_file:
        filename = getattr(uploaded_file, "name", None)
        ext = os.path.splitext(filename)[-1].lower() if filename else None
        file_for_read = uploaded_file
    else:
        ext = None

    if ext == ".csv" and file_for_read:
        try:
            # Reset file pointer if it's a file upload
            if uploaded_file:
                uploaded_file.seek(0)
                df_preview = pd.read_csv(uploaded_file, nrows=1)
                uploaded_file.seek(0)  # Reset for later use
            else:
                df_preview = pd.read_csv(file_for_read, nrows=1)
            
            if df_preview is not None:
                # Get fields from preview
                fields = list(df_preview.columns)
                auto_fields = st.multiselect("Auto-detected fields to translate", fields, default=[f for f in fields if "question" in f.lower() or "text" in f.lower() or "content" in f.lower() or "response" in f.lower()])
        except Exception as e:
            st.error(f"Failed to preview CSV: {e}")
            fields = []
            auto_fields = []
    elif ext == ".arrow" or (selected_dataset and filename and isinstance(filename, str) and filename.endswith("dataset_dict.json")):
        try:
            from datasets import load_from_disk, DatasetDict
            if selected_dataset and filename and isinstance(filename, str) and filename.endswith("dataset_dict.json"):
                dataset_dir = os.path.dirname(filename)
            elif ext == ".arrow" and filename:
                dataset_dir = os.path.dirname(os.path.dirname(filename))
            else:
                dataset_dir = os.path.dirname(filename) if filename else None
            if dataset_dir is not None:
                with st.spinner("Loading HuggingFace dataset..."):
                    dataset = load_from_disk(dataset_dir)
                if isinstance(dataset, DatasetDict):
                    dataset = dataset["train"]
                df = dataset.to_pandas()
                if not hasattr(df, "columns"):
                    st.error("Failed to convert HuggingFace dataset to DataFrame.")
                    fields = []
                    auto_fields = []
                else:
                    fields = list(df.columns)
                    auto_fields = st.multiselect("Auto-detected fields to translate", fields, default=[f for f in fields if "question" in f.lower() or "text" in f.lower() or "content" in f.lower() or "response" in f.lower()])
            else:
                st.error("Dataset directory could not be determined.")
                fields = []
                auto_fields = []
        except Exception as e:
            st.error(f"Failed to load HuggingFace dataset: {e}")
            auto_fields = []
    else:
        auto_fields = st.text_input("Fields to translate (comma-separated)", value="Question,Response").split(",")

chunk_size = st.number_input("Batch size (rows per chunk)", min_value=1, max_value=1000, value=20, step=1)
output_dir = st.text_input("Output Folder", value=os.getcwd())
output_filename = st.text_input("Output File Name", value="translated.csv")

if (selected_dataset or uploaded_file) and st.button("Translate Dataset"):
    import pandas as pd
    import tempfile

    if ext == ".csv":
        try:
            from pathlib import Path
            from typing import Union
            
            # Initialize variables
            df_chunks = []
            field_list = [f.strip() for f in auto_fields if f.strip() in fields]
            total_rows = 0
            
            # Define helper function to process DataFrame chunk
            def process_chunk(chunk_df: pd.DataFrame, fields: list) -> pd.DataFrame:
                processed_df = chunk_df.copy()
                for field in fields:
                    if field in processed_df.columns:
                        processed_df[field + "_th"] = processed_df[field].apply(
                            lambda x: translate_medical_deepseek(str(x), api_key, src_lang, tgt_lang) 
                            if pd.notnull(x) else ""
                        )
                return processed_df

            # Process file based on type
            if uploaded_file:
                uploaded_file.seek(0)
                try:
                    for chunk in pd.read_csv(uploaded_file, chunksize=chunk_size):
                        processed_chunk = process_chunk(chunk, field_list)
                        df_chunks.append(processed_chunk)
                        total_rows += len(processed_chunk)
                        st.info(f"Translated {total_rows} rows so far...")
                except Exception as e:
                    st.error(f"Error processing uploaded file: {e}")
            elif isinstance(file_for_read, str):
                try:
                    for chunk in pd.read_csv(Path(file_for_read), chunksize=chunk_size):
                        processed_chunk = process_chunk(chunk, field_list)
                        df_chunks.append(processed_chunk)
                        total_rows += len(processed_chunk)
                        st.info(f"Translated {total_rows} rows so far...")
                except Exception as e:
                    st.error(f"Error processing file: {e}")
            else:
                st.error("Invalid file input")

            # Combine all chunks and save
            if df_chunks:
                df_final = pd.concat(df_chunks, ignore_index=True)
                st.success("Batch translation complete!")
                st.dataframe(df_final)
                
                # Save to file and create download button
                output_path = os.path.join(output_dir, output_filename)
                df_final.to_csv(output_path, index=False)
                csv_data = df_final.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Translated CSV",
                    csv_data,
                    output_filename,
                    "text/csv"
                )
                st.info(f"File saved to: {output_path}")
            else:
                st.error("No data to translate.")
        except Exception as e:
            st.error(f"Failed to process CSV: {e}")
    elif ext == ".arrow" or (selected_dataset and filename and isinstance(filename, str) and filename.endswith("dataset_dict.json")):
        try:
            from datasets import load_from_disk, DatasetDict
            if selected_dataset and filename and isinstance(filename, str) and filename.endswith("dataset_dict.json"):
                dataset_dir = os.path.dirname(filename)
            elif ext == ".arrow" and filename:
                dataset_dir = os.path.dirname(os.path.dirname(filename))
            else:
                dataset_dir = os.path.dirname(filename) if filename else None
            if dataset_dir is not None:
                with st.spinner("Loading HuggingFace dataset..."):
                    dataset = load_from_disk(dataset_dir)
                if isinstance(dataset, DatasetDict):
                    dataset = dataset["train"]
                df = dataset.to_pandas()
                if df is None or not hasattr(df, "columns"):
                    st.error("Failed to convert HuggingFace dataset to DataFrame.")
                else:
                    field_list = [f.strip() for f in auto_fields if f.strip() in df.columns]
                    if not field_list:
                        st.warning("No valid fields selected for translation.")
                    else:
                        for field in field_list:
                            with st.spinner(f"Translating field '{field}'..."):
                                df[field + "_th"] = df[field].apply(
                                    lambda x: translate_medical_deepseek(str(x), api_key, src_lang, tgt_lang) 
                                    if pd.notnull(x) else ""
                                )
                        
                        output_path = os.path.join(output_dir, output_filename.replace(".csv", ".csv"))
                        try:
                            df.to_csv(output_path, index=False)
                            st.success("Batch translation complete!")
                            st.dataframe(df)
                            
                            # Create download button
                            csv_data = df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "Download Translated CSV",
                                csv_data,
                                output_filename,
                                "text/csv"
                            )
                            st.info(f"CSV file saved to: {output_path}")
                        except Exception as e:
                            st.error(f"Error saving file: {e}")
            else:
                st.error("No dataset loaded for translation.")
        except Exception as e:
            st.error(f"Failed to process HuggingFace dataset: {e}")
