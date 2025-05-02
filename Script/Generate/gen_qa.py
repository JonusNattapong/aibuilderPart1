import os
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from config_generate import OUTPUT_DIR, QA_FILENAME
from gen_utils import parse_json_output, invoke_llm_with_retry

# Template for Question Answering (Extractive)
qa_template_text = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
คุณคือ AI ผู้ช่วยสร้างข้อมูลสำหรับงาน Question Answering (QA) ภาษาไทย
หน้าที่ของคุณคือสร้าง 'context' (เนื้อหา), 'question' (คำถามที่สามารถตอบได้จาก context), และ 'answer' (คำตอบสั้นๆ ที่อยู่ใน context) 1 ชุด
สร้างเฉพาะ context, question, และ answer เท่านั้น ในรูปแบบ JSON ตามตัวอย่าง

ตัวอย่าง:
หัวข้อ: ประวัติศาสตร์ไทย
ผลลัพธ์ JSON:
```json
{{
  "context": "กรุงศรีอยุธยาเคยเป็นราชธานีของอาณาจักรสยามนานถึง 417 ปี ก่อนจะเสียกรุงครั้งที่สองในปี พ.ศ. 2310",
  "question": "กรุงศรีอยุธยาเสียกรุงครั้งที่สองในปีใด?",
  "answer": "พ.ศ. 2310"
}}
```
<|eot_id|><|start_header_id|>user<|end_header_id|>
หัวข้อ: {topic}
ผลลัพธ์ JSON:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```json
"""
qa_prompt = PromptTemplate(
    input_variables=["topic"],
    template=qa_template_text
)

def generate_qa_data(llm, num_samples, topics, filename=QA_FILENAME):
    """Generates question answering data."""
    if not llm:
        print("LLM not initialized. Skipping QA data generation.")
        return

    chain = LLMChain(llm=llm, prompt=qa_prompt)
    data = []
    print(f"\nGenerating {num_samples} QA samples...")

    generated_count = 0
    while generated_count < num_samples:
        topic = topics[generated_count % len(topics)] # Cycle through topics
        print(f"Generating sample {generated_count + 1}/{num_samples} (Topic: {topic})...")

        llm_output = invoke_llm_with_retry(chain, {"topic": topic})

        if llm_output:
            parsed_data = parse_json_output(llm_output)
            if parsed_data and 'context' in parsed_data and 'question' in parsed_data and 'answer' in parsed_data:
                 # Basic validation: check if answer is roughly in context
                 if parsed_data['answer'].strip() and parsed_data['answer'] in parsed_data['context']:
                     data.append(parsed_data)
                     generated_count += 1
                 else:
                     print(f"Warning: Generated answer '{parsed_data['answer']}' not found in context or is empty. Skipping.")
            else:
                print("Warning: Failed to parse valid QA data from LLM output.")
        else:
             print("Warning: LLM invocation failed after retries.")

    # Save to CSV
    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} samples to {output_path}")
    else:
        print("No QA data was generated.")
