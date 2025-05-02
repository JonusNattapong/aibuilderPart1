import os
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from config_generate import OUTPUT_DIR, FILL_MASK_FILENAME
from gen_utils import parse_json_output, invoke_llm_with_retry

# Template for Fill-Mask Data
fill_mask_template_text = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
คุณคือ AI ผู้ช่วยสร้างข้อมูลสำหรับงาน Fill-Mask ภาษาไทย
หน้าที่ของคุณคือสร้าง 'masked_sentence' (ประโยคที่มีคำว่า '<mask>' แทนที่คำหนึ่งคำ) และ 'target_word' (คำที่ถูกแทนที่) 1 ชุด
สร้างเฉพาะ masked_sentence และ target_word เท่านั้น ในรูปแบบ JSON ตามตัวอย่าง

ตัวอย่าง:
หัวข้อ: สุภาษิต
ผลลัพธ์ JSON:
```json
{{
  "masked_sentence": "น้ำขึ้นให้รีบ <mask>",
  "target_word": "ตัก"
}}
```
<|eot_id|><|start_header_id|>user<|end_header_id|>
หัวข้อ: {topic}
ผลลัพธ์ JSON:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```json
"""
fill_mask_prompt = PromptTemplate(
    input_variables=["topic"],
    template=fill_mask_template_text
)

def generate_fill_mask_data(llm, num_samples, topics, filename=FILL_MASK_FILENAME):
    """Generates Fill-Mask data."""
    if not llm:
        print("LLM not initialized. Skipping Fill-Mask data generation.")
        return

    chain = LLMChain(llm=llm, prompt=fill_mask_prompt)
    data = []
    print(f"\nGenerating {num_samples} Fill-Mask samples...")

    generated_count = 0
    while generated_count < num_samples:
        topic = topics[generated_count % len(topics)]
        print(f"Generating sample {generated_count + 1}/{num_samples} (Topic: {topic})...")

        llm_output = invoke_llm_with_retry(chain, {"topic": topic})

        if llm_output:
            parsed_data = parse_json_output(llm_output)
            if parsed_data and 'masked_sentence' in parsed_data and 'target_word' in parsed_data:
                 # Basic validation: check mask exists and target is single word
                 if '<mask>' in parsed_data['masked_sentence'] and parsed_data['target_word'].strip() and ' ' not in parsed_data['target_word'].strip():
                     data.append(parsed_data)
                     generated_count += 1
                 else:
                     print("Warning: Generated sentence doesn't contain '<mask>' or target is invalid. Skipping.")
            else:
                print("Warning: Failed to parse valid Fill-Mask data from LLM output.")
        else:
             print("Warning: LLM invocation failed after retries.")

    # Save to CSV
    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} samples to {output_path}")
    else:
        print("No Fill-Mask data was generated.")
