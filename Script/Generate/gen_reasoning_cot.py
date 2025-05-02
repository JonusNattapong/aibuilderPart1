import os
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from config_generate import OUTPUT_DIR, REASONING_COT_FILENAME
from gen_utils import parse_json_output, invoke_llm_with_retry

# Template for Reasoning (Chain-of-Thought)
reasoning_cot_template_text = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
คุณคือ AI ผู้ช่วยสร้างข้อมูลสำหรับงาน Reasoning แบบ Chain-of-Thought (CoT) ภาษาไทย
หน้าที่ของคุณคือสร้าง 'problem' (โจทย์ปัญหา), 'reasoning_steps' (ขั้นตอนการคิดเพื่อแก้ปัญหา), และ 'answer' (คำตอบสุดท้าย) 1 ชุด
สร้างเฉพาะ problem, reasoning_steps, และ answer เท่านั้น ในรูปแบบ JSON ตามตัวอย่าง

ตัวอย่าง:
หัวข้อ: ปัญหาคณิตศาสตร์พื้นฐาน
ผลลัพธ์ JSON:
```json
{{
  "problem": "มานีมีแอปเปิ้ล 5 ผล ซื้อมาเพิ่มอีก 3 ผล แล้วแบ่งให้เพื่อนไป 2 ผล มานีจะเหลือแอปเปิ้ลกี่ผล?",
  "reasoning_steps": "1. ตอนแรกมานีมีแอปเปิ้ล 5 ผล\\n2. ซื้อเพิ่มอีก 3 ผล ทำให้มีแอปเปิ้ลทั้งหมด 5 + 3 = 8 ผล\\n3. แบ่งให้เพื่อนไป 2 ผล จะเหลือแอปเปิ้ล 8 - 2 = 6 ผล",
  "answer": "6 ผล"
}}
```
<|eot_id|><|start_header_id|>user<|end_header_id|>
หัวข้อ: {topic}
ผลลัพธ์ JSON:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```json
"""
reasoning_cot_prompt = PromptTemplate(
    input_variables=["topic"],
    template=reasoning_cot_template_text
)

def generate_reasoning_cot_data(llm, num_samples, topics, filename=REASONING_COT_FILENAME):
    """Generates Reasoning (Chain-of-Thought) data."""
    if not llm:
        print("LLM not initialized. Skipping reasoning CoT data generation.")
        return

    chain = LLMChain(llm=llm, prompt=reasoning_cot_prompt)
    data = []
    print(f"\nGenerating {num_samples} reasoning CoT samples...")

    generated_count = 0
    while generated_count < num_samples:
        topic = topics[generated_count % len(topics)]
        print(f"Generating sample {generated_count + 1}/{num_samples} (Topic: {topic})...")

        llm_output = invoke_llm_with_retry(chain, {"topic": topic})

        if llm_output:
            parsed_data = parse_json_output(llm_output)
            if parsed_data and 'problem' in parsed_data and 'reasoning_steps' in parsed_data and 'answer' in parsed_data:
                 if parsed_data['problem'].strip() and parsed_data['reasoning_steps'].strip() and parsed_data['answer'].strip():
                     data.append(parsed_data)
                     generated_count += 1
                 else:
                     print("Warning: Generated problem, steps, or answer is empty. Skipping.")
            else:
                print("Warning: Failed to parse valid reasoning CoT data from LLM output.")
        else:
             print("Warning: LLM invocation failed after retries.")

    # Save to CSV
    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} samples to {output_path}")
    else:
        print("No reasoning CoT data was generated.")

# Example usage (for testing purposes)
# if __name__ == "__main__":
#     from config_generate import REASONING_COT_TOPICS
#     from generate_datasets_langchain import setup_llm # Assuming setup_llm is accessible
#     llm = setup_llm()
#     if llm:
#         generate_reasoning_cot_data(llm, 5, REASONING_COT_TOPICS)
#     else:
#         print("Could not initialize LLM for testing.")
