import os
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from config_generate import OUTPUT_DIR, ZERO_SHOT_FILENAME, ZERO_SHOT_POTENTIAL_LABELS
from gen_utils import parse_json_output, invoke_llm_with_retry

# Template for Zero-Shot Classification Data
zero_shot_template_text = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
คุณคือ AI ผู้ช่วยสร้างข้อมูลสำหรับงาน Zero-Shot Text Classification ภาษาไทย
หน้าที่ของคุณคือสร้าง 'sequence' (ข้อความตัวอย่าง) และ 'expected_label' (หมวดหมู่ที่ถูกต้องที่สุดสำหรับข้อความนั้น จากรายการหมวดหมู่ที่เป็นไปได้) 1 ชุด
สร้างเฉพาะ sequence และ expected_label เท่านั้น ในรูปแบบ JSON ตามตัวอย่าง

ตัวอย่าง:
หัวข้อ: ข่าวบันเทิง
หมวดหมู่ที่เป็นไปได้: {potential_labels}
ผลลัพธ์ JSON:
```json
{{
  "sequence": "นักแสดงหนุ่มชื่อดังประกาศแต่งงานสายฟ้าแลบกับแฟนสาวนอกวงการ",
  "expected_label": "บันเทิง"
}}
```
<|eot_id|><|start_header_id|>user<|end_header_id|>
หัวข้อ: {topic}
หมวดหมู่ที่เป็นไปได้: {potential_labels}
ผลลัพธ์ JSON:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```json
"""
zero_shot_prompt = PromptTemplate(
    input_variables=["topic", "potential_labels"],
    template=zero_shot_template_text
)

def generate_zero_shot_data(llm, num_samples, topics, potential_labels=ZERO_SHOT_POTENTIAL_LABELS, filename=ZERO_SHOT_FILENAME):
    """Generates Zero-Shot Classification data."""
    if not llm:
        print("LLM not initialized. Skipping Zero-Shot data generation.")
        return

    chain = LLMChain(llm=llm, prompt=zero_shot_prompt)
    data = []
    print(f"\nGenerating {num_samples} Zero-Shot samples...")

    generated_count = 0
    while generated_count < num_samples:
        topic = topics[generated_count % len(topics)]
        print(f"Generating sample {generated_count + 1}/{num_samples} (Topic: {topic})...")

        llm_output = invoke_llm_with_retry(chain, {"topic": topic, "potential_labels": potential_labels})

        if llm_output:
            parsed_data = parse_json_output(llm_output)
            if parsed_data and 'sequence' in parsed_data and 'expected_label' in parsed_data:
                 # Basic validation: check if label is in potential labels and sequence is non-empty
                 if parsed_data['sequence'].strip() and parsed_data['expected_label'] in potential_labels:
                     # Add potential labels to the output row for context
                     parsed_data['candidate_labels'] = ", ".join(potential_labels) # Store as string for CSV
                     data.append(parsed_data)
                     generated_count += 1
                 else:
                     print(f"Warning: Generated sequence empty or expected label '{parsed_data['expected_label']}' not in potential labels. Skipping.")
            else:
                print("Warning: Failed to parse valid Zero-Shot data from LLM output.")
        else:
             print("Warning: LLM invocation failed after retries.")

    # Save to CSV
    if data:
        # Ensure consistent column order
        df = pd.DataFrame(data, columns=['sequence', 'expected_label', 'candidate_labels'])
        output_path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} samples to {output_path}")
    else:
        print("No Zero-Shot data was generated.")
