import os
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from config_generate import OUTPUT_DIR, CLASSIFICATION_FILENAME, CLASSIFICATION_CATEGORIES
from gen_utils import parse_json_output, invoke_llm_with_retry

# Template for Text Classification
classification_template_text = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
คุณคือ AI ผู้ช่วยสร้างข้อมูลสำหรับงาน Text Classification ภาษาไทย
หน้าที่ของคุณคือสร้างข้อความตัวอย่าง 1 รายการ พร้อมระบุหมวดหมู่ (label) ที่เหมาะสมที่สุดจากรายการที่กำหนดให้
สร้างเฉพาะข้อความและหมวดหมู่เท่านั้น ในรูปแบบ JSON ตามตัวอย่าง

ตัวอย่าง:
หัวข้อ: กีฬา
หมวดหมู่ที่เป็นไปได้: {categories}
ผลลัพธ์ JSON:
```json
{{
  "text": "ทีมชาติไทยเอาชนะคู่แข่งไป 3-0 ในการแข่งขันฟุตบอลโลกรอบคัดเลือกเมื่อวานนี้",
  "label": "กีฬา"
}}
```
<|eot_id|><|start_header_id|>user<|end_header_id|>
หัวข้อ: {topic}
หมวดหมู่ที่เป็นไปได้: {categories}
ผลลัพธ์ JSON:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```json
"""
classification_prompt = PromptTemplate(
    input_variables=["topic", "categories"],
    template=classification_template_text
)

def generate_text_classification_data(llm, num_samples, topics, categories=CLASSIFICATION_CATEGORIES, filename=CLASSIFICATION_FILENAME):
    """Generates text classification data."""
    if not llm:
        print("LLM not initialized. Skipping classification data generation.")
        return

    chain = LLMChain(llm=llm, prompt=classification_prompt)
    data = []
    print(f"\nGenerating {num_samples} text classification samples...")

    generated_count = 0
    while generated_count < num_samples:
        topic = topics[generated_count % len(topics)] # Cycle through topics
        print(f"Generating sample {generated_count + 1}/{num_samples} (Topic: {topic})...")

        llm_output = invoke_llm_with_retry(chain, {"topic": topic, "categories": categories})

        if llm_output:
            parsed_data = parse_json_output(llm_output)
            if parsed_data and 'text' in parsed_data and 'label' in parsed_data:
                if parsed_data['label'] in categories:
                    data.append(parsed_data)
                    generated_count += 1
                else:
                    print(f"Warning: Generated label '{parsed_data['label']}' not in allowed categories. Skipping.")
            else:
                print("Warning: Failed to parse valid classification data from LLM output.")
        else:
             print("Warning: LLM invocation failed after retries.")
             # Optionally break or continue trying with the next topic

    # Save to CSV
    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} samples to {output_path}")
    else:
        print("No classification data was generated.")
