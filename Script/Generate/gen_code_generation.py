import os
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from config_generate import OUTPUT_DIR, CODE_GENERATION_FILENAME
from gen_utils import parse_json_output, invoke_llm_with_retry

# Template for Code Generation
code_generation_template_text = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
คุณคือ AI ผู้ช่วยสร้างข้อมูลสำหรับงาน Code Generation ภาษาไทย
หน้าที่ของคุณคือสร้าง 'description' (คำอธิบายโจทย์สั้นๆ) และ 'code_snippet' (โค้ด Python สั้นๆ ที่แก้โจทย์นั้น) 1 คู่
สร้างเฉพาะ description และ code_snippet เท่านั้น ในรูปแบบ JSON ตามตัวอย่าง

ตัวอย่าง:
หัวข้อ: ฟังก์ชัน Python พื้นฐาน
ผลลัพธ์ JSON:
```json
{{
  "description": "เขียนฟังก์ชัน Python ที่รับ list ของตัวเลข และคืนค่าผลรวมของตัวเลขทั้งหมด",
  "code_snippet": "def sum_list(numbers):\\n    total = 0\\n    for num in numbers:\\n        total += num\\n    return total"
}}
```
<|eot_id|><|start_header_id|>user<|end_header_id|>
หัวข้อ: {topic}
ผลลัพธ์ JSON:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```json
"""
code_generation_prompt = PromptTemplate(
    input_variables=["topic"],
    template=code_generation_template_text
)

def generate_code_generation_data(llm, num_samples, topics, filename=CODE_GENERATION_FILENAME):
    """Generates Code Generation data."""
    if not llm:
        print("LLM not initialized. Skipping code generation data generation.")
        return

    chain = LLMChain(llm=llm, prompt=code_generation_prompt)
    data = []
    print(f"\nGenerating {num_samples} code generation samples...")

    generated_count = 0
    while generated_count < num_samples:
        topic = topics[generated_count % len(topics)]
        print(f"Generating sample {generated_count + 1}/{num_samples} (Topic: {topic})...")

        llm_output = invoke_llm_with_retry(chain, {"topic": topic})

        if llm_output:
            parsed_data = parse_json_output(llm_output)
            if parsed_data and 'description' in parsed_data and 'code_snippet' in parsed_data:
                 if parsed_data['description'].strip() and parsed_data['code_snippet'].strip():
                     data.append(parsed_data)
                     generated_count += 1
                 else:
                     print("Warning: Generated description or code snippet is empty. Skipping.")
            else:
                print("Warning: Failed to parse valid code generation data from LLM output.")
        else:
             print("Warning: LLM invocation failed after retries.")

    # Save to CSV
    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} samples to {output_path}")
    else:
        print("No code generation data was generated.")

# Example usage (for testing purposes)
# if __name__ == "__main__":
#     from config_generate import CODE_GENERATION_TOPICS
#     from generate_datasets_langchain import setup_llm # Assuming setup_llm is accessible
#     llm = setup_llm()
#     if llm:
#         generate_code_generation_data(llm, 5, CODE_GENERATION_TOPICS)
#     else:
#         print("Could not initialize LLM for testing.")
