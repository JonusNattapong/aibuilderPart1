import os
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from config_generate import OUTPUT_DIR, TRANSLATION_FILENAME
from gen_utils import parse_json_output, invoke_llm_with_retry

# Template for Translation (Thai to English)
translation_template_text = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
คุณคือ AI ผู้ช่วยสร้างข้อมูลสำหรับงาน Translation (Thai to English)
หน้าที่ของคุณคือสร้างประโยคภาษาไทย ('th') และคำแปลภาษาอังกฤษ ('en') ที่ถูกต้อง 1 คู่
สร้างเฉพาะ th และ en เท่านั้น ในรูปแบบ JSON ตามตัวอย่าง

ตัวอย่าง:
หัวข้อ: การทักทาย
ผลลัพธ์ JSON:
```json
{{
  "th": "สวัสดีตอนเช้า วันนี้อากาศดีมาก",
  "en": "Good morning. The weather is very nice today."
}}
```
<|eot_id|><|start_header_id|>user<|end_header_id|>
หัวข้อ: {topic}
ผลลัพธ์ JSON:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```json
"""
translation_prompt = PromptTemplate(
    input_variables=["topic"],
    template=translation_template_text
)

def generate_translation_data(llm, num_samples, topics, filename=TRANSLATION_FILENAME):
    """Generates Translation (Thai to English) data."""
    if not llm:
        print("LLM not initialized. Skipping translation data generation.")
        return

    chain = LLMChain(llm=llm, prompt=translation_prompt)
    data = []
    print(f"\nGenerating {num_samples} translation samples (TH-EN)...")

    generated_count = 0
    while generated_count < num_samples:
        topic = topics[generated_count % len(topics)]
        print(f"Generating sample {generated_count + 1}/{num_samples} (Topic: {topic})...")

        llm_output = invoke_llm_with_retry(chain, {"topic": topic})

        if llm_output:
            parsed_data = parse_json_output(llm_output)
            if parsed_data and 'th' in parsed_data and 'en' in parsed_data:
                 if parsed_data['th'].strip() and parsed_data['en'].strip():
                     data.append(parsed_data)
                     generated_count += 1
                 else:
                     print("Warning: Generated Thai or English text is empty. Skipping.")
            else:
                print("Warning: Failed to parse valid translation data from LLM output.")
        else:
             print("Warning: LLM invocation failed after retries.")

    # Save to CSV
    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} samples to {output_path}")
    else:
        print("No translation data was generated.")
