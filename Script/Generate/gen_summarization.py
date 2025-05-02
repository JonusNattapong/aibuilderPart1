import os
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from config_generate import OUTPUT_DIR, SUMMARIZATION_FILENAME
from gen_utils import parse_json_output, invoke_llm_with_retry

# Template for Summarization
summarization_template_text = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
คุณคือ AI ผู้ช่วยสร้างข้อมูลสำหรับงาน Summarization ภาษาไทย
หน้าที่ของคุณคือสร้าง 'document' (ข้อความต้นฉบับที่ค่อนข้างยาว) และ 'summary' (บทสรุปสั้นๆ ของ document) 1 คู่
สร้างเฉพาะ document และ summary เท่านั้น ในรูปแบบ JSON ตามตัวอย่าง

ตัวอย่าง:
หัวข้อ: ข่าวเศรษฐกิจ
ผลลัพธ์ JSON:
```json
{{
  "document": "ธนาคารแห่งประเทศไทยประกาศคงอัตราดอกเบี้ยนโยบายไว้ที่ระดับเดิม เนื่องจากเศรษฐกิจไทยยังคงฟื้นตัวอย่างค่อยเป็นค่อยไป ท่ามกลางความผันผวนของเศรษฐกิจโลก อย่างไรก็ตาม ธปท. จะติดตามสถานการณ์อย่างใกล้ชิดและพร้อมปรับนโยบายหากจำเป็น",
  "summary": "ธปท. คงดอกเบี้ยนโยบาย เหตุเศรษฐกิจไทยฟื้นตัวช้า จับตาสถานการณ์ใกล้ชิด"
}}
```
<|eot_id|><|start_header_id|>user<|end_header_id|>
หัวข้อ: {topic}
ผลลัพธ์ JSON:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```json
"""
summarization_prompt = PromptTemplate(
    input_variables=["topic"],
    template=summarization_template_text
)

def generate_summarization_data(llm, num_samples, topics, filename=SUMMARIZATION_FILENAME):
    """Generates Summarization data."""
    if not llm:
        print("LLM not initialized. Skipping summarization data generation.")
        return

    chain = LLMChain(llm=llm, prompt=summarization_prompt)
    data = []
    print(f"\nGenerating {num_samples} summarization samples...")

    generated_count = 0
    while generated_count < num_samples:
        topic = topics[generated_count % len(topics)]
        print(f"Generating sample {generated_count + 1}/{num_samples} (Topic: {topic})...")

        llm_output = invoke_llm_with_retry(chain, {"topic": topic})

        if llm_output:
            parsed_data = parse_json_output(llm_output)
            if parsed_data and 'document' in parsed_data and 'summary' in parsed_data:
                 # Basic validation: check non-empty and summary is shorter
                 if parsed_data['document'].strip() and parsed_data['summary'].strip() and len(parsed_data['summary']) < len(parsed_data['document']):
                     data.append(parsed_data)
                     generated_count += 1
                 else:
                     print("Warning: Generated document/summary is empty or summary is not shorter. Skipping.")
            else:
                print("Warning: Failed to parse valid summarization data from LLM output.")
        else:
             print("Warning: LLM invocation failed after retries.")

    # Save to CSV
    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} samples to {output_path}")
    else:
        print("No summarization data was generated.")
