import os
import json
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from config_generate import OUTPUT_DIR, TEXT_RANKING_FILENAME
from gen_utils import parse_json_output, invoke_llm_with_retry

# Template for Text Ranking Data
text_ranking_template_text = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
คุณคือ AI ผู้ช่วยสร้างข้อมูลสำหรับงาน Text Ranking ภาษาไทย
หน้าที่ของคุณคือสร้าง 'query' (คำค้นหา), 'positive_passage' (เนื้อหาที่เกี่ยวข้องโดยตรงกับ query), และ 'negative_passages' (รายการเนื้อหาที่ไม่เกี่ยวข้อง หรือเกี่ยวข้องน้อยมาก 2-3 รายการ) 1 ชุด
สร้างเฉพาะ query, positive_passage, และ negative_passages (list of strings) เท่านั้น ในรูปแบบ JSON ตามตัวอย่าง

ตัวอย่าง:
หัวข้อ: สถานที่ท่องเที่ยวในเชียงใหม่
ผลลัพธ์ JSON:
```json
{{
  "query": "ดอยสุเทพอยู่ที่ไหน",
  "positive_passage": "ดอยสุเทพเป็นภูเขาสำคัญ ตั้งอยู่ในอุทยานแห่งชาติดอยสุเทพ-ปุย จังหวัดเชียงใหม่ เป็นที่ประดิษฐานวัดพระธาตุดอยสุเทพราชวรวิหาร",
  "negative_passages": [
    "เชียงใหม่มีอาหารอร่อยมากมาย เช่น ข้าวซอย น้ำเงี้ยว",
    "การเดินทางไปเชียงใหม่สามารถไปได้ทั้งเครื่องบิน รถไฟ และรถโดยสาร",
    "ดอยอินทนนท์เป็นยอดเขาที่สูงที่สุดในประเทศไทย"
  ]
}}
```
<|eot_id|><|start_header_id|>user<|end_header_id|>
หัวข้อ: {topic}
ผลลัพธ์ JSON:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```json
"""
text_ranking_prompt = PromptTemplate(
    input_variables=["topic"],
    template=text_ranking_template_text
)

def generate_text_ranking_data(llm, num_samples, topics, filename=TEXT_RANKING_FILENAME):
    """Generates Text Ranking data."""
    if not llm:
        print("LLM not initialized. Skipping Text Ranking data generation.")
        return

    chain = LLMChain(llm=llm, prompt=text_ranking_prompt)
    data = []
    print(f"\nGenerating {num_samples} Text Ranking samples...")

    generated_count = 0
    while generated_count < num_samples:
        topic = topics[generated_count % len(topics)]
        print(f"Generating sample {generated_count + 1}/{num_samples} (Topic: {topic})...")

        llm_output = invoke_llm_with_retry(chain, {"topic": topic})

        if llm_output:
            parsed_data = parse_json_output(llm_output)
            if parsed_data and 'query' in parsed_data and 'positive_passage' in parsed_data and 'negative_passages' in parsed_data:
                 # Basic validation: check fields non-empty and negatives is a list
                 if parsed_data['query'].strip() and parsed_data['positive_passage'].strip() and \
                    isinstance(parsed_data['negative_passages'], list) and len(parsed_data['negative_passages']) > 0 and \
                    all(isinstance(neg, str) and neg.strip() for neg in parsed_data['negative_passages']):
                     # Convert list of negatives to a single string for CSV simplicity
                     parsed_data['negative_passages'] = json.dumps(parsed_data['negative_passages'], ensure_ascii=False)
                     data.append(parsed_data)
                     generated_count += 1
                 else:
                     print("Warning: Generated query/passage empty or invalid negative passages format. Skipping.")
            else:
                print("Warning: Failed to parse valid Text Ranking data from LLM output.")
        else:
             print("Warning: LLM invocation failed after retries.")

    # Save to CSV
    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} samples to {output_path}")
    else:
        print("No Text Ranking data was generated.")
