import os
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from config_generate import OUTPUT_DIR, STYLE_TRANSFER_FILENAME
from gen_utils import parse_json_output, invoke_llm_with_retry

# Template for Style Transfer (Formal/Informal)
style_transfer_template_text = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
คุณคือ AI ผู้ช่วยสร้างข้อมูลสำหรับงาน Style Transfer (Formal/Informal) ภาษาไทย
หน้าที่ของคุณคือสร้างประโยคภาษาไทยแบบ 'formal' (ทางการ) และ 'informal' (ไม่ทางการ) ที่มีความหมายเหมือนกัน 1 คู่
สร้างเฉพาะ formal และ informal เท่านั้น ในรูปแบบ JSON ตามตัวอย่าง

ตัวอย่าง:
หัวข้อ: การสอบถามข้อมูล
ผลลัพธ์ JSON:
```json
{{
  "formal": "หากท่านมีข้อสงสัยเพิ่มเติม สามารถติดต่อสอบถามได้ที่แผนกบริการลูกค้า",
  "informal": "ถ้าสงสัยอะไรอีก ถามได้ที่ฝ่ายบริการลูกค้าเลยนะ"
}}
```
<|eot_id|><|start_header_id|>user<|end_header_id|>
หัวข้อ: {topic}
ผลลัพธ์ JSON:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```json
"""
style_transfer_prompt = PromptTemplate(
    input_variables=["topic"],
    template=style_transfer_template_text
)

def generate_style_transfer_data(llm, num_samples, topics, filename=STYLE_TRANSFER_FILENAME):
    """Generates Style Transfer (Formal/Informal) data."""
    if not llm:
        print("LLM not initialized. Skipping style transfer data generation.")
        return

    chain = LLMChain(llm=llm, prompt=style_transfer_prompt)
    data = []
    print(f"\nGenerating {num_samples} style transfer samples...")

    generated_count = 0
    while generated_count < num_samples:
        topic = topics[generated_count % len(topics)]
        print(f"Generating sample {generated_count + 1}/{num_samples} (Topic: {topic})...")

        llm_output = invoke_llm_with_retry(chain, {"topic": topic})

        if llm_output:
            parsed_data = parse_json_output(llm_output)
            if parsed_data and 'formal' in parsed_data and 'informal' in parsed_data:
                 if parsed_data['formal'].strip() and parsed_data['informal'].strip():
                     data.append(parsed_data)
                     generated_count += 1
                 else:
                     print("Warning: Generated formal or informal text is empty. Skipping.")
            else:
                print("Warning: Failed to parse valid style transfer data from LLM output.")
        else:
             print("Warning: LLM invocation failed after retries.")

    # Save to CSV
    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} samples to {output_path}")
    else:
        print("No style transfer data was generated.")
