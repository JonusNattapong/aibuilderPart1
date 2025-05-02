import os
import json
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from config_generate import OUTPUT_DIR, TABLE_QA_FILENAME
from gen_utils import parse_json_output, invoke_llm_with_retry

# Template for Table Question Answering
table_qa_template_text = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
คุณคือ AI ผู้ช่วยสร้างข้อมูลสำหรับงาน Table Question Answering ภาษาไทย
หน้าที่ของคุณคือสร้าง 'table' (ตารางข้อมูลอย่างง่ายในรูปแบบ JSON string), 'question' (คำถามที่ตอบได้จากตาราง), และ 'answer' (คำตอบสั้นๆ จากตาราง) 1 ชุด
สร้างเฉพาะ table, question, และ answer เท่านั้น ในรูปแบบ JSON ตามตัวอย่าง

ตัวอย่าง:
หัวข้อ: ผลไม้และราคา
ผลลัพธ์ JSON:
```json
{{
  "table": "[{{\\"ผลไม้\\": \\"แอปเปิ้ล\\", \\"ราคา (บาท/กก.)\\": 80}}, {{\\"ผลไม้\\": \\"ส้ม\\", \\"ราคา (บาท/กก.)\\": 50}}, {{\\"ผลไม้\\": \\"กล้วย\\", \\"ราคา (บาท/กก.)\\": 30}}]",
  "question": "ส้มราคากิโลกรัมละเท่าไหร่?",
  "answer": "50"
}}
```
<|eot_id|><|start_header_id|>user<|end_header_id|>
หัวข้อ: {topic}
ผลลัพธ์ JSON:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```json
"""
table_qa_prompt = PromptTemplate(
    input_variables=["topic"],
    template=table_qa_template_text
)

def generate_table_qa_data(llm, num_samples, topics, filename=TABLE_QA_FILENAME):
    """Generates Table Question Answering data."""
    if not llm:
        print("LLM not initialized. Skipping Table QA data generation.")
        return

    chain = LLMChain(llm=llm, prompt=table_qa_prompt)
    data = []
    print(f"\nGenerating {num_samples} Table QA samples...")

    generated_count = 0
    while generated_count < num_samples:
        topic = topics[generated_count % len(topics)]
        print(f"Generating sample {generated_count + 1}/{num_samples} (Topic: {topic})...")

        llm_output = invoke_llm_with_retry(chain, {"topic": topic})

        if llm_output:
            parsed_data = parse_json_output(llm_output)
            if parsed_data and 'table' in parsed_data and 'question' in parsed_data and 'answer' in parsed_data:
                 # Basic validation: check if table is valid JSON string and other fields are non-empty
                 try:
                     json.loads(parsed_data['table']) # Check if table string is valid JSON
                     if parsed_data['question'].strip() and str(parsed_data['answer']).strip():
                         data.append(parsed_data)
                         generated_count += 1
                     else:
                         print("Warning: Generated question or answer is empty. Skipping.")
                 except json.JSONDecodeError:
                     print(f"Warning: Generated table is not a valid JSON string: {parsed_data['table']}. Skipping.")
                 except Exception as e:
                     print(f"Warning: Error validating table QA data: {e}. Skipping.")
            else:
                print("Warning: Failed to parse valid Table QA data from LLM output.")
        else:
             print("Warning: LLM invocation failed after retries.")

    # Save to CSV
    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} samples to {output_path}")
    else:
        print("No Table QA data was generated.")
