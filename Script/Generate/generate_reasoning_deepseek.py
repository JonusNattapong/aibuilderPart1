import os
import argparse
import requests
import json
import time
import csv
import random
from config_generate_nlp import (
    NLP_TOPICS
)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate reasoning datasets using DeepSeek API.')
    parser.add_argument('--output_dir', type=str, default='Script/Generate/REASONING_OUTPUT', help='Directory to save the generated datasets.')
    parser.add_argument('--max_retries', type=int, default=3, help='Maximum number of retries for API requests.')
    parser.add_argument('--retry_delay', type=int, default=2, help='Delay between retries in seconds.')
    parser.add_argument('--num_samples_per_task', type=int, default=100, help='Number of samples to generate per reasoning task.')
    parser.add_argument('--deepseek_api_url', type=str, default='https://api.deepseek.com/v1/chat/completions', help='DeepSeek API URL.')
    parser.add_argument('--deepseek_model', type=str, default='deepseek-chat', help='DeepSeek model to use.')
    return parser.parse_args()

def invoke_deepseek(system_prompt, user_prompt, deepseek_model, deepseek_api_url, max_retries, delay, temperature=0.7, max_tokens=512):
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY environment variable not set.")
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    payload = {
        "model": deepseek_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    for attempt in range(max_retries):
        try:
            response = requests.post(deepseek_api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            if result and 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0].get('message', {}).get('content')
                if content:
                    return content.strip()
            print(f"Warning: Unexpected DeepSeek API response: {result}")
        except Exception as e:
            print(f"DeepSeek API request failed (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
    return None

def parse_json_output(llm_output):
    try:
        match = llm_output
        if "```json" in llm_output:
            match = llm_output.split("```json")[-1].split("```")[0]
        return json.loads(match)
    except Exception as e:
        print(f"Error parsing JSON: {e}\nOutput: {llm_output}")
        return None

def generate_reasoning_samples(task, topics, n, deepseek_model, deepseek_api_url, max_retries, retry_delay):
    task_prompts = {
        'chain_of_thought': {
            'system': 'สร้างโจทย์และขั้นตอนการคิด',
            'example': '{"question": "โจทย์", "steps": ["ขั้นตอน1", "ขั้นตอน2"], "answer": "คำตอบ"}'
        },
        'meta_reasoning': {
            'system': 'สร้างโจทย์และกระบวนการคิดเชิงวิเคราะห์',
            'example': '{"question": "โจทย์", "thought_process": "กระบวนการคิด", "strategy": "กลยุทธ์", "evaluation": "การประเมิน"}'
        },
        'pattern_recognition': {
            'system': 'สร้างลำดับและรูปแบบที่ซ่อนอยู่',
            'example': '{"sequence": "ลำดับ", "pattern": "รูปแบบ", "next_value": "ค่าถัดไป", "explanation": "คำอธิบาย"}'
        },
        'react': {
            'system': 'สร้างสถานการณ์และปฏิกิริยาตอบสนอง',
            'example': '{"situation": "สถานการณ์", "thoughts": ["ความคิด1", "ความคิด2"], "actions": ["การกระทำ1", "การกระทำ2"], "outcome": "ผลลัพธ์"}'
        },
        'reflection': {
            'system': 'สร้างประสบการณ์และการสะท้อนคิด',
            'example': '{"experience": "ประสบการณ์", "analysis": ["การวิเคราะห์1", "การวิเคราะห์2"], "lessons_learned": "บทเรียนที่ได้", "future_actions": "การกระทำในอนาคต"}'
        },
        'toolformer': {
            'system': 'สร้างปัญหาและเครื่องมือที่เหมาะสม',
            'example': '{"problem": "ปัญหา", "available_tools": ["เครื่องมือ1", "เครื่องมือ2"], "selected_tool": "เครื่องมือที่เลือก", "reasoning": "เหตุผลในการเลือก"}'
        }
    }

    system_prompt = (
        f"คุณคือ AI ผู้ช่วยสร้างข้อมูลสำหรับงาน {task} ภาษาไทย "
        f"หน้าที่ของคุณคือ{task_prompts[task]['system']} "
        "โดยใช้หัวข้อที่กำหนดให้ ให้เนื้อหามีความสมจริง หลากหลาย ไม่ซ้ำ และสอดคล้องกับบริบท "
        "ควรใช้ภาษาไทยที่เป็นธรรมชาติและหลีกเลี่ยงการใช้รูปแบบซ้ำๆ "
        "ผลลัพธ์ต้องอยู่ในรูปแบบ JSON เท่านั้น เช่น "
        f"{task_prompts[task]['example']}"
    )
    user_prompt_template = (
        "หัวข้อ: {topic}\n"
        f"โปรดสร้างข้อมูลสำหรับงาน {task} โดยใช้หัวข้อนี้ "
        "เนื้อหาต้องสมจริง หลีกเลี่ยงการซ้ำหรือใช้โครงสร้างเดิมๆ"
    )
    samples = []
    for i in range(n):
        topic = random.choice(topics)
        user_prompt = user_prompt_template.format(topic=topic)
        output = invoke_deepseek(system_prompt, user_prompt, deepseek_model, deepseek_api_url, max_retries, retry_delay)
        if output:
            parsed = parse_json_output(output)
            if parsed:
                if task == 'chain_of_thought':
                    samples.append([parsed.get("question", ""), ",".join(parsed.get("steps", [])), parsed.get("answer", "")])
                elif task == 'meta_reasoning':
                    samples.append([parsed.get("question", ""), parsed.get("thought_process", ""), parsed.get("strategy", ""), parsed.get("evaluation", "")])
                elif task == 'pattern_recognition':
                    samples.append([parsed.get("sequence", ""), parsed.get("pattern", ""), parsed.get("next_value", ""), parsed.get("explanation", "")])
                elif task == 'react':
                    samples.append([parsed.get("situation", ""), ",".join(parsed.get("thoughts", [])), ",".join(parsed.get("actions", [])), parsed.get("outcome", "")])
                elif task == 'reflection':
                    samples.append([parsed.get("experience", ""), ",".join(parsed.get("analysis", [])), parsed.get("lessons_learned", ""), parsed.get("future_actions", "")])
                elif task == 'toolformer':
                    samples.append([parsed.get("problem", ""), ",".join(parsed.get("available_tools", [])), parsed.get("selected_tool", ""), parsed.get("reasoning", "")])
                print(f"Generated {task} sample {len(samples)+1}/{n}")
            else:
                print(f"Invalid output: {output}")
        else:
            print(f"Failed to generate {task} sample {len(samples)+1}/{n}")
    return samples

def main():
    args = parse_args()
    OUTPUT_DIR = args.output_dir
    MAX_RETRIES = args.max_retries
    RETRY_DELAY = args.retry_delay
    NUM_SAMPLES_PER_TASK = args.num_samples_per_task
    DEEPSEEK_API_URL = args.deepseek_api_url
    DEEPSEEK_MODEL = args.deepseek_model

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    task_headers = {
        'chain_of_thought': ["id", "question", "steps", "answer"],
        'meta_reasoning': ["id", "question", "thought_process", "strategy", "evaluation"],
        'pattern_recognition': ["id", "sequence", "pattern", "next_value", "explanation"],
        'react': ["id", "situation", "thoughts", "actions", "outcome"],
        'reflection': ["id", "experience", "analysis", "lessons_learned", "future_actions"],
        'toolformer': ["id", "problem", "available_tools", "selected_tool", "reasoning"]
    }

    for task in task_headers.keys():
        samples = generate_reasoning_samples(task, NLP_TOPICS, NUM_SAMPLES_PER_TASK, DEEPSEEK_MODEL, DEEPSEEK_API_URL, MAX_RETRIES, RETRY_DELAY)
        out_path = os.path.join(OUTPUT_DIR, f"{task}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(task_headers[task])
            writer.writerows([[str(i+1)] + list(row) for i, row in enumerate(samples)])
        print(f"Saved {len(samples)} {task} samples to {out_path}")

if __name__ == "__main__":
    main()