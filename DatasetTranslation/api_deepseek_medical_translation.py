import os
import json
import requests
from flask import Flask, request, jsonify

# Automatically load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DATASET_DOWNLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "DatasetDownload")

def build_prompt(text):
    return (
        "You are a professional medical translator. "
        "Translate the following English medical text to Thai with high accuracy, preserving all medical terminology and context.\n\n"
        f"English:\n{text}\n\nThai:"
    )

def translate_medical_deepseek(text, api_key=None):
    if api_key is None:
        api_key = DEEPSEEK_API_KEY
    if not api_key:
        raise ValueError("DeepSeek API key not set. Set DEEPSEEK_API_KEY environment variable.")
    prompt = build_prompt(text)
    payload = json.dumps({
        "messages": [
            {"content": "You are a helpful assistant", "role": "system"},
            {"content": prompt, "role": "user"}
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
    response = requests.post(DEEPSEEK_API_URL, headers=headers, data=payload)
    response.raise_for_status()
    result = response.json()
    content = result["choices"][0]["message"]["content"]
    if "Thai:" in content:
        return content.split("Thai:")[-1].strip()
    return content.strip()

def translate_dataset_arrow(arrow_dir, output_path, text_fields):
    from datasets import load_from_disk, DatasetDict
    import pandas as pd

    dataset = load_from_disk(arrow_dir)
    if isinstance(dataset, DatasetDict):
        dataset = dataset["train"]
    df = dataset.to_pandas()
    for field in text_fields:
        if field in df.columns:
            df[field + "_th"] = df[field].apply(lambda x: translate_medical_deepseek(str(x)) if pd.notnull(x) else "")
    df.to_csv(output_path, index=False)
    return output_path

# Flask API
app = Flask(__name__)

@app.route("/translate", methods=["POST"])
def api_translate():
    data = request.json
    text = data.get("text")
    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400
    try:
        translation = translate_medical_deepseek(text)
        return jsonify({"translation": translation})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/translate-dataset", methods=["POST"])
def api_translate_dataset():
    data = request.json
    arrow_dir = data.get("arrow_dir")
    output_path = data.get("output_path")
    fields = data.get("fields")
    if not arrow_dir or not output_path or not fields:
        return jsonify({"error": "Missing required fields: arrow_dir, output_path, fields"}), 400
    try:
        result_path = translate_dataset_arrow(arrow_dir, output_path, fields)
        return jsonify({"output": result_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Medical translation (English to Thai) using DeepSeek API (requests).")
    parser.add_argument("--text", type=str, help="English medical text to translate")
    parser.add_argument("--dataset", type=str, help="Dataset file name in DatasetDownload (CSV or JSON)")
    parser.add_argument("--output", type=str, help="Output file for translated dataset")
    parser.add_argument("--fields", type=str, nargs="+", help="Text fields to translate in dataset")
    parser.add_argument("--arrow_dir", type=str, help="Path to Arrow dataset directory (HuggingFace format)")
    parser.add_argument("--api_key", type=str, default=None, help="DeepSeek API key (or set DEEPSEEK_API_KEY env var)")
    parser.add_argument("--run_api", action="store_true", help="Run as Flask API server")
    args = parser.parse_args()

    if args.run_api:
        app.run(host="0.0.0.0", port=8000)
    elif args.text:
        translation = translate_medical_deepseek(args.text, args.api_key)
        print("=== Translation Result ===")
        print(translation)
    elif args.arrow_dir and args.output and args.fields:
        translate_dataset_arrow(args.arrow_dir, args.output, args.fields)
        print(f"Translation completed. Output saved to {args.output}")
    elif args.dataset and args.output and args.fields:
        # CSV/JSON support (legacy)
        import pandas as pd
        import csv
        input_path = os.path.join(DATASET_DOWNLOAD_DIR, args.dataset)
        ext = os.path.splitext(input_path)[-1].lower()
        if ext in [".csv"]:
            df = pd.read_csv(input_path)
            for field in args.fields:
                if field in df.columns:
                    df[field + "_th"] = df[field].apply(lambda x: translate_medical_deepseek(str(x)) if pd.notnull(x) else "")
            df.to_csv(args.output, index=False, quoting=csv.QUOTE_NONNUMERIC)
        elif ext in [".json"]:
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                for field in args.fields:
                    if field in item:
                        item[field + "_th"] = translate_medical_deepseek(str(item[field]))
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError("Unsupported file format. Only CSV and JSON are supported.")
        print(f"Translation completed. Output saved to {args.output}")
    else:
        parser.print_help()