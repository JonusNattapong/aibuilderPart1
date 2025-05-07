import os
import argparse
import requests
import json
import time
import csv
import random
from config_generate_nlp import (
    NLP_TASKS, NLP_TOPICS
)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate NLP datasets using Ollama API.')
    parser.add_argument('--output_dir', type=str, default='Script/Generate/OLLAMA_OUTPUT', help='Directory to save the generated datasets.')
    parser.add_argument('--max_retries', type=int, default=3, help='Maximum number of retries for API requests.')
    parser.add_argument('--retry_delay', type=int, default=2, help='Delay between retries in seconds.')
    parser.add_argument('--num_samples_per_task', type=int, default=100, help='Number of samples to generate per NLP task.')
    parser.add_argument('--ollama_api_url', type=str, default='http://localhost:11434/api/generate', help='Ollama API URL.')
    return parser.parse_args()

def invoke_ollama(system_prompt, user_prompt, ollama_api_url, max_retries, delay):
    payload = {
        "model": "mistral",
        "prompt": f"{system_prompt}\n{user_prompt}",
        "stream": False
    }
    for attempt in range(max_retries):
        try:
            response = requests.post(ollama_api_url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            if result and 'response' in result:
                return result['response']
            print(f"Warning: Unexpected Ollama API response: {result}")
        except Exception as e:
            print(f"Ollama API request failed (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
    return None

def generate_nlp_samples(task, topics, n, ollama_api_url, max_retries, retry_delay):
    system_prompt = (
        f"คุณคือ AI ผู้ช่วยสร้างข้อมูลสำหรับงาน {task} ภาษาไทย "
        "โปรดสร้างข้อความที่สมจริงและเป็นธรรมชาติ"
    )
    user_prompt_template = (
        "หัวข้อ: {topic}\n"
        f"โปรดสร้างข้อมูลสำหรับงาน {task} โดยใช้หัวข้อนี้"
    )
    samples = []
    for i in range(n):
        topic = random.choice(NLP_TOPICS.get('general', []))
        user_prompt = user_prompt_template.format(topic=topic)
        output = invoke_ollama(system_prompt, user_prompt, ollama_api_url, max_retries, retry_delay)
        if output:
            samples.append([str(i+1), output])
            print(f"Generated {task} sample {len(samples)+1}/{n}")
        else:
            print(f"Failed to generate {task} sample {len(samples)+1}/{n}")
    return samples

def main():
    args = parse_args()
    OUTPUT_DIR = args.output_dir
    MAX_RETRIES = args.max_retries
    RETRY_DELAY = args.retry_delay
    NUM_SAMPLES_PER_TASK = args.num_samples_per_task
    OLLAMA_API_URL = args.ollama_api_url

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for task, topics in NLP_TASKS.items():
        samples = generate_nlp_samples(task, NLP_TOPICS, NUM_SAMPLES_PER_TASK, OLLAMA_API_URL, MAX_RETRIES, RETRY_DELAY)
        out_path = os.path.join(OUTPUT_DIR, f"{task}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "text"])
            writer.writerows(samples)
        print(f"Saved {len(samples)} {task} samples to {out_path}")

if __name__ == "__main__":
    main()