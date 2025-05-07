import os
import pandas as pd
import random
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from config_generate_sentiment import OUTPUT_DIR, NUM_SAMPLES_PER_TASK, HAPPY_TOPICS, SAD_TOPICS, ANGRY_TOPICS

def get_llm():
    from langchain_huggingface import HuggingFaceEndpoint
    model_id = os.environ.get("SENTIMENT_LLM_MODEL", "scb10x/llama3.2-typhoon2-3b-instruct")
    return HuggingFaceEndpoint(model=model_id, temperature=0.7, max_new_tokens=128)

sentiment_prompt_text = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
คุณคือ AI ผู้ช่วยสร้างข้อมูลสำหรับงาน Sentiment Classification ภาษาไทย
หน้าที่ของคุณคือสร้างข้อความตัวอย่าง 1 รายการที่แสดงอารมณ์ '{label}' อย่างชัดเจน โดยใช้หัวข้อที่กำหนดให้
สร้างเฉพาะข้อความและหมวดหมู่เท่านั้น ในรูปแบบ JSON ตามตัวอย่าง

ตัวอย่าง:
หัวข้อ: {topic}
อารมณ์: {label}
ผลลัพธ์ JSON:
```json
{{
  "text": "วันนี้ฉันสอบผ่าน ดีใจมาก!",
  "label": "{label}"
}}
```
<|eot_id|><|start_header_id|>user<|end_header_id|>
หัวข้อ: {topic}
อารมณ์: {label}
ผลลัพธ์ JSON:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```json
"""

sentiment_prompt = PromptTemplate(
    input_variables=["topic", "label"],
    template=sentiment_prompt_text
)

def generate_sentiment_data(llm, label, topics, n):
    chain = LLMChain(llm=llm, prompt=sentiment_prompt)
    data = []
    for i in range(n):
        topic = random.choice(topics)
        llm_output = chain.run({"topic": topic, "label": label})
        try:
            json_str = llm_output.split("```json")[-1].split("```")[0]
            parsed = pd.read_json(f"[{json_str}]")[0]
            if parsed["label"] == label:
                data.append(parsed)
                print(f"Generated {label} sample {i+1}/{n}")
        except Exception as e:
            print(f"Parse error: {e} | Output: {llm_output}")
    return data

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    llm = get_llm()
    all_data = []
    for label, topics in [("happy", HAPPY_TOPICS), ("sad", SAD_TOPICS), ("angry", ANGRY_TOPICS)]:
        samples = generate_sentiment_data(llm, label, topics, NUM_SAMPLES_PER_TASK)
        all_data.extend(samples)
    df = pd.DataFrame(all_data)
    out_path = os.path.join(OUTPUT_DIR, "thai_sentiment_langchain.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved {len(df)} samples to {out_path}")

if __name__ == "__main__":
    main()