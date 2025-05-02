import os
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from config_generate import OUTPUT_DIR, TEXT_GEN_FILENAME
from gen_utils import parse_json_output, invoke_llm_with_retry

# Template for Text Generation
text_generation_template_text = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
คุณคือ AI ผู้ช่วยสร้างข้อมูลสำหรับงาน Text Generation ภาษาไทย
หน้าที่ของคุณคือสร้าง 'prompt' (ข้อความเริ่มต้น) และ 'generated_text' (ข้อความที่แต่งต่อจาก prompt) 1 คู่
สร้างเฉพาะ prompt และ generated_text เท่านั้น ในรูปแบบ JSON ตามตัวอย่าง

ตัวอย่าง:
หัวข้อ: การเริ่มต้นเขียนนิทาน
ผลลัพธ์ JSON:
```json
{{
  "prompt": "กาลครั้งหนึ่งนานมาแล้ว ในหมู่บ้านเล็กๆ ริมชายป่า มีเด็กหญิงคนหนึ่งชื่อว่า",
  "generated_text": "หนูน้อยหมวกแดง เธอมักจะสวมหมวกสีแดงสดใสที่คุณยายถักให้เสมอ วันหนึ่งคุณแม่ใช้ให้เธอเอาขนมไปเยี่ยมคุณยายซึ่งป่วยอยู่ที่บ้านกลางป่า..."
}}
```
<|eot_id|><|start_header_id|>user<|end_header_id|>
หัวข้อ: {topic}
ผลลัพธ์ JSON:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```json
"""
text_generation_prompt = PromptTemplate(
    input_variables=["topic"],
    template=text_generation_template_text
)

def generate_text_generation_data(llm, num_samples, topics, filename=TEXT_GEN_FILENAME):
    """Generates Text Generation data."""
    if not llm:
        print("LLM not initialized. Skipping text generation data generation.")
        return

    chain = LLMChain(llm=llm, prompt=text_generation_prompt)
    data = []
    print(f"\nGenerating {num_samples} text generation samples...")

    generated_count = 0
    while generated_count < num_samples:
        topic = topics[generated_count % len(topics)]
        print(f"Generating sample {generated_count + 1}/{num_samples} (Topic: {topic})...")

        llm_output = invoke_llm_with_retry(chain, {"topic": topic})

        if llm_output:
            parsed_data = parse_json_output(llm_output)
            if parsed_data and 'prompt' in parsed_data and 'generated_text' in parsed_data:
                 if parsed_data['prompt'].strip() and parsed_data['generated_text'].strip():
                     data.append(parsed_data)
                     generated_count += 1
                 else:
                     print("Warning: Generated prompt or text is empty. Skipping.")
            else:
                print("Warning: Failed to parse valid text generation data from LLM output.")
        else:
             print("Warning: LLM invocation failed after retries.")

    # Save to CSV
    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} samples to {output_path}")
    else:
        print("No text generation data was generated.")
