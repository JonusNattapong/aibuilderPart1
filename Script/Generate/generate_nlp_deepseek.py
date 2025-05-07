import argparse
import os
import requests
import json
import time
import csv
import random
from config_generate_nlp import (
    NLP_TASKS, NLP_TOPICS
)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate NLP datasets using DeepSeek API.')
    parser.add_argument('--output_dir', type=str, default='Script/Generate/NLP_OUTPUT', help='Directory to save the generated datasets.')
    parser.add_argument('--max_retries', type=int, default=3, help='Maximum number of retries for API requests.')
    parser.add_argument('--retry_delay', type=int, default=2, help='Delay between retries in seconds.')
    parser.add_argument('--num_samples_per_task', type=int, default=100, help='Number of samples to generate per NLP task.')
    parser.add_argument('--deepseek_api_url', type=str, default='https://api.deepseek.com/v1/chat/completions', help='DeepSeek API URL.')
    parser.add_argument('--deepseek_model', type=str, default='deepseek-chat', help='DeepSeek model to use.')
    return parser.parse_args()

def invoke_deepseek(system_prompt, user_prompt, deepseek_model, deepseek_api_url, max_retries, delay, temperature=0.7, max_tokens=128):
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

def generate_nlp_samples(task, topics, n, deepseek_model, deepseek_api_url, max_retries, retry_delay):
    task_prompts = {
        'text_classification': {
            'system': 'สร้างข้อความภาษาไทยที่สมจริงและระบุประเภทหมวดหมู่ที่เหมาะสม โดยข้อความควรมีความหลากหลาย สื่อความหมายชัดเจน และเกี่ยวข้องกับหัวข้อที่กำหนด',
            'example': '{"text": "การเรียนออนไลน์ช่วยประหยัดเวลาแต่ต้องมีวินัยในการเรียนรู้ด้วยตนเอง", "label": "การศึกษา"}'
        },
        'ner': {
            'system': 'สร้างข้อความภาษาไทยที่มีการระบุ entities ต่างๆ เช่น ชื่อคน องค์กร สถานที่ วันเวลา พร้อมตำแหน่งเริ่มต้นและสิ้นสุดของแต่ละ entity',
            'example': '{"text": "นายสมชาย ทำงานที่บริษัทไทยเทคในกรุงเทพ", "entities": [{"type": "PERSON", "start": 4, "end": 10, "text": "สมชาย"}]}'
        },
        'table_question_answering': {
            'system': 'สร้างตารางข้อมูลภาษาไทยที่มีความซับซ้อน พร้อมคำถามและคำตอบที่เกี่ยวข้องกับข้อมูลในตาราง',
            'example': '{"table": "ชื่อ,อายุ,อาชีพ\nสมชาย,25,วิศวกร\nสมหญิง,30,หมอ", "question": "ใครอายุมากที่สุด?", "answer": "สมหญิง อายุ 30 ปี"}'
        },
        'question_answering': {
            'system': 'สร้างบริบทภาษาไทยที่มีเนื้อหาสมบูรณ์ พร้อมคำถามที่เกี่ยวข้องและคำตอบที่ถูกต้อง',
            'example': '{"context": "ปัจจุบันเทคโนโลยี AI มีบทบาทสำคัญในชีวิตประจำวัน", "question": "อะไรมีบทบาทสำคัญ?", "answer": "เทคโนโลยี AI"}'
        },
        'zero_shot_classification': {
            'system': 'สร้างข้อความภาษาไทยและระบุประเภทที่เป็นไปได้ พร้อมทำนายประเภทที่เหมาะสมที่สุด',
            'example': '{"text": "อากาศร้อนมากต้องเปิดแอร์ตลอด", "labels": ["สภาพอากาศ", "การใช้พลังงาน"], "prediction": "สภาพอากาศ"}'
        },
        'translation': {
            'system': 'สร้างคู่ประโยคภาษาไทย-อังกฤษที่มีความหมายตรงกันและเป็นธรรมชาติ',
            'example': '{"source": "วันนี้อากาศดีมาก", "target": "The weather is very nice today"}'
        },
        'summarization': {
            'system': 'สร้างเนื้อหาภาษาไทยที่มีความยาวและสรุปใจความสำคัญให้กระชับ',
            'example': '{"text": "เนื้อหาต้นฉบับที่ยาว...", "summary": "สรุปสั้นๆ"}'
        },
        'text_generation': {
            'system': 'สร้างข้อความภาษาไทยต่อเนื่องจากประโยคที่กำหนด',
            'example': '{"prompt": "วันนี้อากาศดี", "generated_text": "วันนี้อากาศดี เหมาะแก่การออกไปเดินเล่นในสวน"}'
        },
        'text2text_generation': {
            'system': 'แปลงข้อความภาษาไทยจากรูปแบบหนึ่งเป็นอีกรูปแบบหนึ่ง',
            'example': '{"input": "ข้อความทางการ", "output": "ข้อความไม่เป็นทางการ"}'
        },
        'fill_mask': {
            'system': 'สร้างประโยคภาษาไทยที่มีคำปิดบังและระบุคำที่ควรเติม',
            'example': '{"text": "วันนี้อากาศร้อน [MASK] ต้องเปิดแอร์", "masked_text": "[MASK]", "filled_text": "มาก"}'
        },
        'sentence_similarity': {
            'system': 'สร้างคู่ประโยคภาษาไทยและระบุค่าความคล้ายคลึงระหว่าง 0 ถึง 1',
            'example': '{"sentence1": "วันนี้อากาศดี", "sentence2": "อากาศดีจังเลย", "similarity": 0.9}'
        },
        'text_ranking': {
            'system': 'สร้างข้อความภาษาไทยและจัดลำดับความสำคัญ',
            'example': '{"text": "ข่าวด่วน: เกิดเหตุฉุกเฉิน", "rank": 1}'
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
                if task == 'text_classification':
                    samples.append([parsed.get("text", ""), parsed.get("label", "")])
                elif task == 'ner':
                    for entity in parsed.get("entities", []):
                        samples.append([parsed.get("text", ""), entity.get("type", ""), entity.get("start", 0), entity.get("end", 0), entity.get("text", "")])
                elif task == 'table_question_answering':
                    samples.append([parsed.get("table", ""), parsed.get("question", ""), parsed.get("answer", "")])
                elif task == 'question_answering':
                    samples.append([parsed.get("context", ""), parsed.get("question", ""), parsed.get("answer", "")])
                elif task == 'zero_shot_classification':
                    samples.append([parsed.get("text", ""), ",".join(parsed.get("labels", [])), parsed.get("prediction", "")])
                elif task == 'translation':
                    samples.append([parsed.get("source", ""), parsed.get("target", "")])
                elif task == 'summarization':
                    samples.append([parsed.get("text", ""), parsed.get("summary", "")])
                elif task == 'feature_extraction':
                    samples.append([parsed.get("text", ""), ",".join(parsed.get("features", []))])
                elif task == 'text_generation':
                    samples.append([parsed.get("prompt", ""), parsed.get("generated_text", "")])
                elif task == 'text2text_generation':
                    samples.append([parsed.get("input", ""), parsed.get("output", "")])
                elif task == 'fill_mask':
                    samples.append([parsed.get("text", ""), parsed.get("masked_text", ""), parsed.get("filled_text", "")])
                elif task == 'sentence_similarity':
                    samples.append([parsed.get("sentence1", ""), parsed.get("sentence2", ""), parsed.get("similarity", 0.0)])
                elif task == 'text_ranking':
                    samples.append([parsed.get("text", ""), parsed.get("rank", 0)])
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
    task_samples = {}
    task_headers = {
        'text_classification': ["id", "text", "label"],
        'ner': ["id", "text", "entity_type", "start_offset", "end_offset", "entity_text"],
        'table_question_answering': ["id", "table", "question", "answer"],
        'question_answering': ["id", "context", "question", "answer"],
        'zero_shot_classification': ["id", "text", "possible_labels", "prediction"],
        'translation': ["id", "source_text", "target_text"],
        'summarization': ["id", "text", "summary"],
        'feature_extraction': ["id", "text", "features"],
        'text_generation': ["id", "prompt", "generated_text"],
        'text2text_generation': ["id", "input_text", "output_text"],
        'fill_mask': ["id", "text", "masked_text", "filled_text"],
        'sentence_similarity': ["id", "sentence1", "sentence2", "similarity_score"],
        'text_ranking': ["id", "text", "rank"]
    }

    for task, topics in NLP_TASKS.items():
        samples = generate_nlp_samples(task, NLP_TOPICS, NUM_SAMPLES_PER_TASK, DEEPSEEK_MODEL, DEEPSEEK_API_URL, MAX_RETRIES, RETRY_DELAY)
        out_path = os.path.join(OUTPUT_DIR, f"{task}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(task_headers[task])
            writer.writerows([[str(i+1)] + list(row) for i, row in enumerate(samples)])
        print(f"Saved {len(samples)} {task} samples to {out_path}")

if __name__ == "__main__":
    main()