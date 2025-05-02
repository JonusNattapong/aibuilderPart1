import os
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from config_generate import OUTPUT_DIR, NER_FILENAME
from gen_utils import parse_json_output, invoke_llm_with_retry

# Template for Token Classification (NER)
ner_template_text = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
คุณคือ AI ผู้ช่วยสร้างข้อมูลสำหรับงาน Token Classification (Named Entity Recognition - NER) ภาษาไทย
หน้าที่ของคุณคือสร้างประโยคตัวอย่าง ('tokens') และรายการ 'ner_tags' ที่สอดคล้องกัน 1 ชุด
ใช้ Tag รูปแบบ BIO: B-TYPE (Beginning), I-TYPE (Inside), O (Outside)
ประเภท Entity ที่ใช้: PER (บุคคล), ORG (องค์กร), LOC (สถานที่), DATE (วันที่), MISC (อื่นๆ)
สร้างเฉพาะ tokens (list of strings) และ ner_tags (list of strings) เท่านั้น ในรูปแบบ JSON ตามตัวอย่าง

ตัวอย่าง:
หัวข้อ: ข่าวการเมืองไทย
ผลลัพธ์ JSON:
```json
{{
  "tokens": ["คุณ", "เศรษฐา", "ทวีสิน", "เดินทาง", "ไป", "ประเทศ", "ญี่ปุ่น", "เมื่อวานนี้"],
  "ner_tags": ["O", "B-PER", "I-PER", "O", "O", "B-LOC", "I-LOC", "B-DATE"]
}}
```
<|eot_id|><|start_header_id|>user<|end_header_id|>
หัวข้อ: {topic}
ผลลัพธ์ JSON:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```json
"""
ner_prompt = PromptTemplate(
    input_variables=["topic"],
    template=ner_template_text
)

def generate_ner_data(llm, num_samples, topics, filename=NER_FILENAME):
    """Generates NER data."""
    if not llm:
        print("LLM not initialized. Skipping NER data generation.")
        return

    chain = LLMChain(llm=llm, prompt=ner_prompt)
    data = []
    print(f"\nGenerating {num_samples} NER samples...")

    generated_count = 0
    while generated_count < num_samples:
        topic = topics[generated_count % len(topics)]
        print(f"Generating sample {generated_count + 1}/{num_samples} (Topic: {topic})...")

        llm_output = invoke_llm_with_retry(chain, {"topic": topic})

        if llm_output:
            parsed_data = parse_json_output(llm_output)
            if parsed_data and 'tokens' in parsed_data and 'ner_tags' in parsed_data:
                # Basic validation: check if lists have same length
                if isinstance(parsed_data['tokens'], list) and \
                   isinstance(parsed_data['ner_tags'], list) and \
                   len(parsed_data['tokens']) == len(parsed_data['ner_tags']):
                    # Convert lists to space-separated strings for CSV simplicity, or handle lists directly
                    data.append({
                        "tokens": " ".join(parsed_data['tokens']),
                        "ner_tags": " ".join(parsed_data['ner_tags'])
                    })
                    generated_count += 1
                else:
                    print("Warning: Mismatched lengths or invalid format for tokens/ner_tags. Skipping.")
            else:
                print("Warning: Failed to parse valid NER data from LLM output.")
        else:
             print("Warning: LLM invocation failed after retries.")

    # Save to CSV
    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} samples to {output_path}")
    else:
        print("No NER data was generated.")
