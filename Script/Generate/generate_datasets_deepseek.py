import os
import requests
import json
import time
import pandas as pd
import re

# Import configuration (assuming config_generate.py is in the same directory)
from config_generate import (
    OUTPUT_DIR, MAX_RETRIES, RETRY_DELAY, NUM_SAMPLES_PER_TASK,
    DEEPSEEK_API_URL, DEEPSEEK_MODEL, # New imports
    CLASSIFICATION_TOPICS, CLASSIFICATION_CATEGORIES, CLASSIFICATION_FILENAME,
    QA_TOPICS, QA_FILENAME,
    TABLE_QA_TOPICS, TABLE_QA_FILENAME,
    ZERO_SHOT_TOPICS, ZERO_SHOT_POTENTIAL_LABELS, ZERO_SHOT_FILENAME,
    NER_TOPICS, NER_FILENAME,
    TRANSLATION_TOPICS, TRANSLATION_FILENAME,
    SUMMARIZATION_TOPICS, SUMMARIZATION_FILENAME,
    SENTENCE_SIMILARITY_TOPICS, SENTENCE_SIMILARITY_FILENAME,
    TEXT_GEN_TOPICS, TEXT_GEN_FILENAME,
    STYLE_TRANSFER_TOPICS, STYLE_TRANSFER_FILENAME,
    FILL_MASK_TOPICS, FILL_MASK_FILENAME,
    TEXT_RANKING_TOPICS, TEXT_RANKING_FILENAME
)

# --- DeepSeek API Configuration ---
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
# DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions" # Now imported
# DEEPSEEK_MODEL = "deepseek-chat" # Now imported

# --- Helper Functions ---

def parse_json_output(llm_output):
    """Attempts to parse JSON from the LLM output, handling markdown code blocks."""
    if not llm_output:
        return None
    try:
        # Look for JSON within ```json ... ```
        match = re.search(r'```json\s*(\{.*?\})\s*```', llm_output, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(1)
            # Basic cleaning for common issues like trailing commas
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            return json.loads(json_str)
        else:
            # Fallback: try parsing the whole output if no code block found
            # Basic cleaning for common issues like trailing commas
            cleaned_output = re.sub(r',\s*([}\]])', r'\1', llm_output)
            return json.loads(cleaned_output)
    except json.JSONDecodeError:
        print(f"Warning: Could not parse JSON from LLM output: {llm_output}")
        return None
    except Exception as e:
        print(f"Error parsing JSON: {e}\nOutput: {llm_output}")
        return None

def invoke_deepseek_with_retry(system_prompt, user_prompt, max_retries=MAX_RETRIES, delay=RETRY_DELAY, temperature=0.7, max_tokens=512):
    """Invokes the DeepSeek API with retry logic."""
    if not DEEPSEEK_API_KEY:
        print("Error: DEEPSEEK_API_KEY environment variable not set.")
        return None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    payload = {
        "model": DEEPSEEK_MODEL, # Use imported config
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False # We want the full response
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=60) # Use imported config
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            result = response.json()

            if result and 'choices' in result and len(result['choices']) > 0:
                message = result['choices'][0].get('message', {})
                content = message.get('content')
                if content:
                    return content.strip()
                else:
                    print(f"Warning: No content found in DeepSeek response choice: {result['choices'][0]}")
                    return None
            else:
                print(f"Warning: Unexpected DeepSeek API response format: {result}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"DeepSeek API request failed (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Skipping this item.")
                return None
        except Exception as e: # Catch other potential errors like JSON parsing
             print(f"An unexpected error occurred (Attempt {attempt + 1}/{max_retries}): {e}")
             if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
             else:
                print("Max retries reached. Skipping this item.")
                return None
    return None # Should not be reached if max_retries > 0

# --- Prompt Definitions (Adapting from Langchain structure) ---

# System prompts define the AI's role. User prompts contain the specific task details.

SYS_TEXT_CLASSIFICATION = "คุณคือ AI ผู้ช่วยสร้างข้อมูลสำหรับงาน Text Classification ภาษาไทย. หน้าที่ของคุณคือสร้างข้อความตัวอย่าง 1 รายการ พร้อมระบุหมวดหมู่ (label) ที่เหมาะสมที่สุดจากรายการที่กำหนดให้. สร้างเฉพาะข้อความและหมวดหมู่เท่านั้น ในรูปแบบ JSON ตามตัวอย่าง."
USR_TEXT_CLASSIFICATION = """
ตัวอย่าง:
หัวข้อ: กีฬา
หมวดหมู่ที่เป็นไปได้: [กีฬา, การเมือง, บันเทิง, เทคโนโลยี, ธุรกิจ]
ผลลัพธ์ JSON:
```json
{{
  "text": "ทีมชาติไทยเอาชนะคู่แข่งไป 3-0 ในการแข่งขันฟุตบอลโลกรอบคัดเลือกเมื่อวานนี้",
  "label": "กีฬา"
}}
```

หัวข้อ: {topic}
หมวดหมู่ที่เป็นไปได้: {categories}
ผลลัพธ์ JSON:
"""

SYS_QA = "คุณคือ AI ผู้ช่วยสร้างข้อมูลสำหรับงาน Question Answering (QA) ภาษาไทย. หน้าที่ของคุณคือสร้าง 'context' (เนื้อหา), 'question' (คำถามที่สามารถตอบได้จาก context), และ 'answer' (คำตอบสั้นๆ ที่อยู่ใน context) 1 ชุด. สร้างเฉพาะ context, question, และ answer เท่านั้น ในรูปแบบ JSON ตามตัวอย่าง."
USR_QA = """
ตัวอย่าง:
หัวข้อ: ประวัติศาสตร์ไทย
ผลลัพธ์ JSON:
```json
{{
  "context": "กรุงศรีอยุธยาเคยเป็นราชธานีของอาณาจักรสยามนานถึง 417 ปี ก่อนจะเสียกรุงครั้งที่สองในปี พ.ศ. 2310",
  "question": "กรุงศรีอยุธยาเสียกรุงครั้งที่สองในปีใด?",
  "answer": "พ.ศ. 2310"
}}
```

หัวข้อ: {topic}
ผลลัพธ์ JSON:
"""

SYS_TABLE_QA = "คุณคือ AI ผู้ช่วยสร้างข้อมูลสำหรับงาน Table Question Answering ภาษาไทย. หน้าที่ของคุณคือสร้าง 'table' (ตารางข้อมูลอย่างง่ายในรูปแบบ JSON string), 'question' (คำถามที่ตอบได้จากตาราง), และ 'answer' (คำตอบสั้นๆ จากตาราง) 1 ชุด. สร้างเฉพาะ table, question, และ answer เท่านั้น ในรูปแบบ JSON ตามตัวอย่าง."
USR_TABLE_QA = """
ตัวอย่าง:
หัวข้อ: ผลไม้และราคา
ผลลัพธ์ JSON:
```json
{{
  "table": "[{{\\"ผลไม้\\": \\"แอปเปิ้ล\\", \\"ราคา (บาท/กก.)\\": 80}}, {{\\"ผลไม้\\": \\"ส้ม\\", \\"ราคา (บาท/กก.)\\": 50}}, {{\\"ผลไม้\\": \\"กล้วย\\", \\"ราคา (บาท/กก.)\\": 30}}]",
  "question": "ส้มราคากิโลกรัมละเท่าไหร่?",
  "answer": "50"
}}
```

หัวข้อ: {topic}
ผลลัพธ์ JSON:
"""

SYS_ZERO_SHOT = "คุณคือ AI ผู้ช่วยสร้างข้อมูลสำหรับงาน Zero-Shot Text Classification ภาษาไทย. หน้าที่ของคุณคือสร้าง 'sequence' (ข้อความตัวอย่าง) และ 'expected_label' (หมวดหมู่ที่ถูกต้องที่สุดสำหรับข้อความนั้น จากรายการหมวดหมู่ที่เป็นไปได้) 1 ชุด. สร้างเฉพาะ sequence และ expected_label เท่านั้น ในรูปแบบ JSON ตามตัวอย่าง."
USR_ZERO_SHOT = """
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

หัวข้อ: {topic}
หมวดหมู่ที่เป็นไปได้: {potential_labels}
ผลลัพธ์ JSON:
"""

SYS_NER = "คุณคือ AI ผู้ช่วยสร้างข้อมูลสำหรับงาน Token Classification (Named Entity Recognition - NER) ภาษาไทย. หน้าที่ของคุณคือสร้างประโยคตัวอย่าง ('tokens') และรายการ 'ner_tags' ที่สอดคล้องกัน 1 ชุด. ใช้ Tag รูปแบบ BIO: B-TYPE (Beginning), I-TYPE (Inside), O (Outside). ประเภท Entity ที่ใช้: PER (บุคคล), ORG (องค์กร), LOC (สถานที่), DATE (วันที่), MISC (อื่นๆ). สร้างเฉพาะ tokens (list of strings) และ ner_tags (list of strings) เท่านั้น ในรูปแบบ JSON ตามตัวอย่าง."
USR_NER = """
ตัวอย่าง:
หัวข้อ: ข่าวการเมืองไทย
ผลลัพธ์ JSON:
```json
{{
  "tokens": ["คุณ", "เศรษฐา", "ทวีสิน", "เดินทาง", "ไป", "ประเทศ", "ญี่ปุ่น", "เมื่อวานนี้"],
  "ner_tags": ["O", "B-PER", "I-PER", "O", "O", "B-LOC", "I-LOC", "B-DATE"]
}}
```

หัวข้อ: {topic}
ผลลัพธ์ JSON:
"""

SYS_TRANSLATION = "คุณคือ AI ผู้ช่วยสร้างข้อมูลสำหรับงาน Translation (Thai to English). หน้าที่ของคุณคือสร้างประโยคภาษาไทย ('th') และคำแปลภาษาอังกฤษ ('en') ที่ถูกต้อง 1 คู่. สร้างเฉพาะ th และ en เท่านั้น ในรูปแบบ JSON ตามตัวอย่าง."
USR_TRANSLATION = """
ตัวอย่าง:
หัวข้อ: การทักทาย
ผลลัพธ์ JSON:
```json
{{
  "th": "สวัสดีตอนเช้า วันนี้อากาศดีมาก",
  "en": "Good morning. The weather is very nice today."
}}
```

หัวข้อ: {topic}
ผลลัพธ์ JSON:
"""

SYS_SUMMARIZATION = "คุณคือ AI ผู้ช่วยสร้างข้อมูลสำหรับงาน Summarization ภาษาไทย. หน้าที่ของคุณคือสร้าง 'document' (ข้อความต้นฉบับที่ค่อนข้างยาว) และ 'summary' (บทสรุปสั้นๆ ของ document) 1 คู่. สร้างเฉพาะ document และ summary เท่านั้น ในรูปแบบ JSON ตามตัวอย่าง."
USR_SUMMARIZATION = """
ตัวอย่าง:
หัวข้อ: ข่าวเศรษฐกิจ
ผลลัพธ์ JSON:
```json
{{
  "document": "ธนาคารแห่งประเทศไทยประกาศคงอัตราดอกเบี้ยนโยบายไว้ที่ระดับเดิม เนื่องจากเศรษฐกิจไทยยังคงฟื้นตัวอย่างค่อยเป็นค่อยไป ท่ามกลางความผันผวนของเศรษฐกิจโลก อย่างไรก็ตาม ธปท. จะติดตามสถานการณ์อย่างใกล้ชิดและพร้อมปรับนโยบายหากจำเป็น",
  "summary": "ธปท. คงดอกเบี้ยนโยบาย เหตุเศรษฐกิจไทยฟื้นตัวช้า จับตาสถานการณ์ใกล้ชิด"
}}
```

หัวข้อ: {topic}
ผลลัพธ์ JSON:
"""

SYS_SENTENCE_SIMILARITY = "คุณคือ AI ผู้ช่วยสร้างข้อมูลสำหรับงาน Sentence Similarity ภาษาไทย. หน้าที่ของคุณคือสร้าง 'sentence1', 'sentence2' และ 'label' (similar หรือ dissimilar) ที่บ่งบอกความสัมพันธ์ของสองประโยค 1 ชุด. สร้างเฉพาะ sentence1, sentence2, และ label เท่านั้น ในรูปแบบ JSON ตามตัวอย่าง."
USR_SENTENCE_SIMILARITY = """
ตัวอย่าง:
หัวข้อ: ความหมายเหมือนกัน
ผลลัพธ์ JSON:
```json
{{
  "sentence1": "แมวกำลังนอนหลับอยู่บนโซฟา",
  "sentence2": "เจ้าเหมียวกำลังพักผ่อนบนเก้าอี้ยาว",
  "label": "similar"
}}
```
ตัวอย่าง:
หัวข้อ: ความหมายต่างกัน
ผลลัพธ์ JSON:
```json
{{
  "sentence1": "วันนี้อากาศร้อนมาก",
  "sentence2": "ฉันชอบกินไอศกรีมช็อกโกแลต",
  "label": "dissimilar"
}}
```

หัวข้อ: {topic}
ผลลัพธ์ JSON:
"""

SYS_TEXT_GENERATION = "คุณคือ AI ผู้ช่วยสร้างข้อมูลสำหรับงาน Text Generation ภาษาไทย. หน้าที่ของคุณคือสร้าง 'prompt' (ข้อความเริ่มต้น) และ 'generated_text' (ข้อความที่แต่งต่อจาก prompt) 1 คู่. สร้างเฉพาะ prompt และ generated_text เท่านั้น ในรูปแบบ JSON ตามตัวอย่าง."
USR_TEXT_GENERATION = """
ตัวอย่าง:
หัวข้อ: การเริ่มต้นเขียนนิทาน
ผลลัพธ์ JSON:
```json
{{
  "prompt": "กาลครั้งหนึ่งนานมาแล้ว ในหมู่บ้านเล็กๆ ริมชายป่า มีเด็กหญิงคนหนึ่งชื่อว่า",
  "generated_text": "หนูน้อยหมวกแดง เธอมักจะสวมหมวกสีแดงสดใสที่คุณยายถักให้เสมอ วันหนึ่งคุณแม่ใช้ให้เธอเอาขนมไปเยี่ยมคุณยายซึ่งป่วยอยู่ที่บ้านกลางป่า..."
}}
```

หัวข้อ: {topic}
ผลลัพธ์ JSON:
"""

SYS_STYLE_TRANSFER = "คุณคือ AI ผู้ช่วยสร้างข้อมูลสำหรับงาน Style Transfer (Formal/Informal) ภาษาไทย. หน้าที่ของคุณคือสร้างประโยคภาษาไทยแบบ 'formal' (ทางการ) และ 'informal' (ไม่ทางการ) ที่มีความหมายเหมือนกัน 1 คู่. สร้างเฉพาะ formal และ informal เท่านั้น ในรูปแบบ JSON ตามตัวอย่าง."
USR_STYLE_TRANSFER = """
ตัวอย่าง:
หัวข้อ: การสอบถามข้อมูล
ผลลัพธ์ JSON:
```json
{{
  "formal": "หากท่านมีข้อสงสัยเพิ่มเติม สามารถติดต่อสอบถามได้ที่แผนกบริการลูกค้า",
  "informal": "ถ้าสงสัยอะไรอีก ถามได้ที่ฝ่ายบริการลูกค้าเลยนะ"
}}
```

หัวข้อ: {topic}
ผลลัพธ์ JSON:
"""

SYS_FILL_MASK = "คุณคือ AI ผู้ช่วยสร้างข้อมูลสำหรับงาน Fill-Mask ภาษาไทย. หน้าที่ของคุณคือสร้าง 'masked_sentence' (ประโยคที่มีคำว่า '<mask>' แทนที่คำหนึ่งคำ) และ 'target_word' (คำที่ถูกแทนที่) 1 ชุด. สร้างเฉพาะ masked_sentence และ target_word เท่านั้น ในรูปแบบ JSON ตามตัวอย่าง."
USR_FILL_MASK = """
ตัวอย่าง:
หัวข้อ: สุภาษิต
ผลลัพธ์ JSON:
```json
{{
  "masked_sentence": "น้ำขึ้นให้รีบ <mask>",
  "target_word": "ตัก"
}}
```

หัวข้อ: {topic}
ผลลัพธ์ JSON:
"""

SYS_TEXT_RANKING = "คุณคือ AI ผู้ช่วยสร้างข้อมูลสำหรับงาน Text Ranking ภาษาไทย. หน้าที่ของคุณคือสร้าง 'query' (คำค้นหา), 'positive_passage' (เนื้อหาที่เกี่ยวข้องโดยตรงกับ query), และ 'negative_passages' (รายการเนื้อหาที่ไม่เกี่ยวข้อง หรือเกี่ยวข้องน้อยมาก 2-3 รายการ) 1 ชุด. สร้างเฉพาะ query, positive_passage, และ negative_passages (list of strings) เท่านั้น ในรูปแบบ JSON ตามตัวอย่าง."
USR_TEXT_RANKING = """
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

หัวข้อ: {topic}
ผลลัพธ์ JSON:
"""

# --- Data Generation Functions (Using DeepSeek API) ---

def generate_text_classification_data(num_samples, topics, categories=CLASSIFICATION_CATEGORIES, filename=CLASSIFICATION_FILENAME):
    """Generates text classification data using DeepSeek."""
    data = []
    print(f"\nGenerating {num_samples} text classification samples via DeepSeek...")
    generated_count = 0
    while generated_count < num_samples:
        topic = topics[generated_count % len(topics)]
        print(f"Generating sample {generated_count + 1}/{num_samples} (Topic: {topic})...")
        user_prompt = USR_TEXT_CLASSIFICATION.format(topic=topic, categories=categories)
        llm_output = invoke_deepseek_with_retry(SYS_TEXT_CLASSIFICATION, user_prompt)

        if llm_output:
            parsed_data = parse_json_output(llm_output)
            if parsed_data and 'text' in parsed_data and 'label' in parsed_data:
                if parsed_data['label'] in categories:
                    data.append(parsed_data)
                    generated_count += 1
                else:
                    print(f"Warning: Generated label '{parsed_data['label']}' not in allowed categories. Skipping.")
            else:
                print("Warning: Failed to parse valid classification data from LLM output.")
        else:
             print("Warning: LLM invocation failed after retries.")

    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(OUTPUT_DIR, filename.replace(".csv", "_deepseek.csv"))
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} samples to {output_path}")
    else:
        print("No classification data was generated.")

def generate_qa_data(num_samples, topics, filename=QA_FILENAME):
    """Generates question answering data using DeepSeek."""
    data = []
    print(f"\nGenerating {num_samples} QA samples via DeepSeek...")
    generated_count = 0
    while generated_count < num_samples:
        topic = topics[generated_count % len(topics)]
        print(f"Generating sample {generated_count + 1}/{num_samples} (Topic: {topic})...")
        user_prompt = USR_QA.format(topic=topic)
        llm_output = invoke_deepseek_with_retry(SYS_QA, user_prompt)

        if llm_output:
            parsed_data = parse_json_output(llm_output)
            if parsed_data and 'context' in parsed_data and 'question' in parsed_data and 'answer' in parsed_data:
                 if parsed_data['answer'].strip() and parsed_data['answer'] in parsed_data['context']:
                     data.append(parsed_data)
                     generated_count += 1
                 else:
                     print(f"Warning: Generated answer '{parsed_data['answer']}' not found in context or is empty. Skipping.")
            else:
                print("Warning: Failed to parse valid QA data from LLM output.")
        else:
             print("Warning: LLM invocation failed after retries.")

    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(OUTPUT_DIR, filename.replace(".csv", "_deepseek.csv"))
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} samples to {output_path}")
    else:
        print("No QA data was generated.")

def generate_table_qa_data(num_samples, topics, filename=TABLE_QA_FILENAME):
    """Generates Table Question Answering data using DeepSeek."""
    data = []
    print(f"\nGenerating {num_samples} Table QA samples via DeepSeek...")
    generated_count = 0
    while generated_count < num_samples:
        topic = topics[generated_count % len(topics)]
        print(f"Generating sample {generated_count + 1}/{num_samples} (Topic: {topic})...")
        user_prompt = USR_TABLE_QA.format(topic=topic)
        llm_output = invoke_deepseek_with_retry(SYS_TABLE_QA, user_prompt)

        if llm_output:
            parsed_data = parse_json_output(llm_output)
            if parsed_data and 'table' in parsed_data and 'question' in parsed_data and 'answer' in parsed_data:
                 try:
                     json.loads(parsed_data['table']) # Check if table string is valid JSON
                     if parsed_data['question'].strip() and str(parsed_data['answer']).strip():
                         data.append(parsed_data)
                         generated_count += 1
                     else:
                         print("Warning: Generated question or answer is empty. Skipping.")
                 except json.JSONDecodeError:
                     print(f"Warning: Generated table is not a valid JSON string: {parsed_data['table']}. Skipping.")
                 except Exception as e:
                     print(f"Warning: Error validating table QA data: {e}. Skipping.")
            else:
                print("Warning: Failed to parse valid Table QA data from LLM output.")
        else:
             print("Warning: LLM invocation failed after retries.")

    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(OUTPUT_DIR, filename.replace(".csv", "_deepseek.csv"))
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} samples to {output_path}")
    else:
        print("No Table QA data was generated.")

def generate_zero_shot_data(num_samples, topics, potential_labels=ZERO_SHOT_POTENTIAL_LABELS, filename=ZERO_SHOT_FILENAME):
    """Generates Zero-Shot Classification data using DeepSeek."""
    data = []
    print(f"\nGenerating {num_samples} Zero-Shot samples via DeepSeek...")
    generated_count = 0
    while generated_count < num_samples:
        topic = topics[generated_count % len(topics)]
        print(f"Generating sample {generated_count + 1}/{num_samples} (Topic: {topic})...")
        user_prompt = USR_ZERO_SHOT.format(topic=topic, potential_labels=potential_labels)
        llm_output = invoke_deepseek_with_retry(SYS_ZERO_SHOT, user_prompt)

        if llm_output:
            parsed_data = parse_json_output(llm_output)
            if parsed_data and 'sequence' in parsed_data and 'expected_label' in parsed_data:
                 if parsed_data['sequence'].strip() and parsed_data['expected_label'] in potential_labels:
                     parsed_data['candidate_labels'] = ", ".join(potential_labels)
                     data.append(parsed_data)
                     generated_count += 1
                 else:
                     print(f"Warning: Generated sequence empty or expected label '{parsed_data['expected_label']}' not in potential labels. Skipping.")
            else:
                print("Warning: Failed to parse valid Zero-Shot data from LLM output.")
        else:
             print("Warning: LLM invocation failed after retries.")

    if data:
        df = pd.DataFrame(data, columns=['sequence', 'expected_label', 'candidate_labels'])
        output_path = os.path.join(OUTPUT_DIR, filename.replace(".csv", "_deepseek.csv"))
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} samples to {output_path}")
    else:
        print("No Zero-Shot data was generated.")

def generate_ner_data(num_samples, topics, filename=NER_FILENAME):
    """Generates NER data using DeepSeek."""
    data = []
    print(f"\nGenerating {num_samples} NER samples via DeepSeek...")
    generated_count = 0
    while generated_count < num_samples:
        topic = topics[generated_count % len(topics)]
        print(f"Generating sample {generated_count + 1}/{num_samples} (Topic: {topic})...")
        user_prompt = USR_NER.format(topic=topic)
        llm_output = invoke_deepseek_with_retry(SYS_NER, user_prompt)

        if llm_output:
            parsed_data = parse_json_output(llm_output)
            if parsed_data and 'tokens' in parsed_data and 'ner_tags' in parsed_data:
                if isinstance(parsed_data['tokens'], list) and \
                   isinstance(parsed_data['ner_tags'], list) and \
                   len(parsed_data['tokens']) == len(parsed_data['ner_tags']):
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

    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(OUTPUT_DIR, filename.replace(".csv", "_deepseek.csv"))
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} samples to {output_path}")
    else:
        print("No NER data was generated.")

def generate_translation_data(num_samples, topics, filename=TRANSLATION_FILENAME):
    """Generates Translation (Thai to English) data using DeepSeek."""
    data = []
    print(f"\nGenerating {num_samples} translation samples (TH-EN) via DeepSeek...")
    generated_count = 0
    while generated_count < num_samples:
        topic = topics[generated_count % len(topics)]
        print(f"Generating sample {generated_count + 1}/{num_samples} (Topic: {topic})...")
        user_prompt = USR_TRANSLATION.format(topic=topic)
        llm_output = invoke_deepseek_with_retry(SYS_TRANSLATION, user_prompt)

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

    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(OUTPUT_DIR, filename.replace(".csv", "_deepseek.csv"))
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} samples to {output_path}")
    else:
        print("No translation data was generated.")

def generate_summarization_data(num_samples, topics, filename=SUMMARIZATION_FILENAME):
    """Generates Summarization data using DeepSeek."""
    data = []
    print(f"\nGenerating {num_samples} summarization samples via DeepSeek...")
    generated_count = 0
    while generated_count < num_samples:
        topic = topics[generated_count % len(topics)]
        print(f"Generating sample {generated_count + 1}/{num_samples} (Topic: {topic})...")
        user_prompt = USR_SUMMARIZATION.format(topic=topic)
        # Increase max_tokens for summarization input/output if needed
        llm_output = invoke_deepseek_with_retry(SYS_SUMMARIZATION, user_prompt, max_tokens=1024)

        if llm_output:
            parsed_data = parse_json_output(llm_output)
            if parsed_data and 'document' in parsed_data and 'summary' in parsed_data:
                 if parsed_data['document'].strip() and parsed_data['summary'].strip() and len(parsed_data['summary']) < len(parsed_data['document']):
                     data.append(parsed_data)
                     generated_count += 1
                 else:
                     print("Warning: Generated document/summary is empty or summary is not shorter. Skipping.")
            else:
                print("Warning: Failed to parse valid summarization data from LLM output.")
        else:
             print("Warning: LLM invocation failed after retries.")

    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(OUTPUT_DIR, filename.replace(".csv", "_deepseek.csv"))
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} samples to {output_path}")
    else:
        print("No summarization data was generated.")

def generate_sentence_similarity_data(num_samples, topics, filename=SENTENCE_SIMILARITY_FILENAME):
    """Generates Sentence Similarity data using DeepSeek."""
    data = []
    print(f"\nGenerating {num_samples} Sentence Similarity samples via DeepSeek...")
    generated_count = 0
    while generated_count < num_samples:
        topic = topics[generated_count % len(topics)]
        print(f"Generating sample {generated_count + 1}/{num_samples} (Topic: {topic})...")
        user_prompt = USR_SENTENCE_SIMILARITY.format(topic=topic)
        llm_output = invoke_deepseek_with_retry(SYS_SENTENCE_SIMILARITY, user_prompt)

        if llm_output:
            parsed_data = parse_json_output(llm_output)
            if parsed_data and 'sentence1' in parsed_data and 'sentence2' in parsed_data and 'label' in parsed_data:
                 if parsed_data['sentence1'].strip() and parsed_data['sentence2'].strip() and parsed_data['label'] in ['similar', 'dissimilar']:
                     data.append(parsed_data)
                     generated_count += 1
                 else:
                     print("Warning: Generated sentences empty or invalid label. Skipping.")
            else:
                print("Warning: Failed to parse valid Sentence Similarity data from LLM output.")
        else:
             print("Warning: LLM invocation failed after retries.")

    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(OUTPUT_DIR, filename.replace(".csv", "_deepseek.csv"))
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} samples to {output_path}")
    else:
        print("No Sentence Similarity data was generated.")

def generate_text_generation_data(num_samples, topics, filename=TEXT_GEN_FILENAME):
    """Generates Text Generation data using DeepSeek."""
    data = []
    print(f"\nGenerating {num_samples} text generation samples via DeepSeek...")
    generated_count = 0
    while generated_count < num_samples:
        topic = topics[generated_count % len(topics)]
        print(f"Generating sample {generated_count + 1}/{num_samples} (Topic: {topic})...")
        user_prompt = USR_TEXT_GENERATION.format(topic=topic)
        llm_output = invoke_deepseek_with_retry(SYS_TEXT_GENERATION, user_prompt)

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

    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(OUTPUT_DIR, filename.replace(".csv", "_deepseek.csv"))
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} samples to {output_path}")
    else:
        print("No text generation data was generated.")

def generate_style_transfer_data(num_samples, topics, filename=STYLE_TRANSFER_FILENAME):
    """Generates Style Transfer (Formal/Informal) data using DeepSeek."""
    data = []
    print(f"\nGenerating {num_samples} style transfer samples via DeepSeek...")
    generated_count = 0
    while generated_count < num_samples:
        topic = topics[generated_count % len(topics)]
        print(f"Generating sample {generated_count + 1}/{num_samples} (Topic: {topic})...")
        user_prompt = USR_STYLE_TRANSFER.format(topic=topic)
        llm_output = invoke_deepseek_with_retry(SYS_STYLE_TRANSFER, user_prompt)

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

    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(OUTPUT_DIR, filename.replace(".csv", "_deepseek.csv"))
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} samples to {output_path}")
    else:
        print("No style transfer data was generated.")

def generate_fill_mask_data(num_samples, topics, filename=FILL_MASK_FILENAME):
    """Generates Fill-Mask data using DeepSeek."""
    data = []
    print(f"\nGenerating {num_samples} Fill-Mask samples via DeepSeek...")
    generated_count = 0
    while generated_count < num_samples:
        topic = topics[generated_count % len(topics)]
        print(f"Generating sample {generated_count + 1}/{num_samples} (Topic: {topic})...")
        user_prompt = USR_FILL_MASK.format(topic=topic)
        llm_output = invoke_deepseek_with_retry(SYS_FILL_MASK, user_prompt)

        if llm_output:
            parsed_data = parse_json_output(llm_output)
            if parsed_data and 'masked_sentence' in parsed_data and 'target_word' in parsed_data:
                 if '<mask>' in parsed_data['masked_sentence'] and parsed_data['target_word'].strip() and ' ' not in parsed_data['target_word'].strip():
                     data.append(parsed_data)
                     generated_count += 1
                 else:
                     print("Warning: Generated sentence doesn't contain '<mask>' or target is invalid. Skipping.")
            else:
                print("Warning: Failed to parse valid Fill-Mask data from LLM output.")
        else:
             print("Warning: LLM invocation failed after retries.")

    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(OUTPUT_DIR, filename.replace(".csv", "_deepseek.csv"))
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} samples to {output_path}")
    else:
        print("No Fill-Mask data was generated.")

def generate_text_ranking_data(num_samples, topics, filename=TEXT_RANKING_FILENAME):
    """Generates Text Ranking data using DeepSeek."""
    data = []
    print(f"\nGenerating {num_samples} Text Ranking samples via DeepSeek...")
    generated_count = 0
    while generated_count < num_samples:
        topic = topics[generated_count % len(topics)]
        print(f"Generating sample {generated_count + 1}/{num_samples} (Topic: {topic})...")
        user_prompt = USR_TEXT_RANKING.format(topic=topic)
        llm_output = invoke_deepseek_with_retry(SYS_TEXT_RANKING, user_prompt)

        if llm_output:
            parsed_data = parse_json_output(llm_output)
            if parsed_data and 'query' in parsed_data and 'positive_passage' in parsed_data and 'negative_passages' in parsed_data:
                 if parsed_data['query'].strip() and parsed_data['positive_passage'].strip() and \
                    isinstance(parsed_data['negative_passages'], list) and len(parsed_data['negative_passages']) > 0 and \
                    all(isinstance(neg, str) and neg.strip() for neg in parsed_data['negative_passages']):
                     parsed_data['negative_passages'] = json.dumps(parsed_data['negative_passages'], ensure_ascii=False)
                     data.append(parsed_data)
                     generated_count += 1
                 else:
                     print("Warning: Generated query/passage empty or invalid negative passages format. Skipping.")
            else:
                print("Warning: Failed to parse valid Text Ranking data from LLM output.")
        else:
             print("Warning: LLM invocation failed after retries.")

    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(OUTPUT_DIR, filename.replace(".csv", "_deepseek.csv"))
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully generated and saved {len(data)} samples to {output_path}")
    else:
        print("No Text Ranking data was generated.")


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting dataset generation process using DeepSeek API...")

    if not DEEPSEEK_API_KEY:
        print("Error: DEEPSEEK_API_KEY environment variable is not set. Exiting.")
        exit()

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Generate Data for Each Task ---
    generate_text_classification_data(NUM_SAMPLES_PER_TASK, CLASSIFICATION_TOPICS, CLASSIFICATION_CATEGORIES)
    generate_qa_data(NUM_SAMPLES_PER_TASK, QA_TOPICS)
    generate_table_qa_data(NUM_SAMPLES_PER_TASK, TABLE_QA_TOPICS)
    generate_zero_shot_data(NUM_SAMPLES_PER_TASK, ZERO_SHOT_TOPICS, ZERO_SHOT_POTENTIAL_LABELS)
    generate_ner_data(NUM_SAMPLES_PER_TASK, NER_TOPICS)
    generate_translation_data(NUM_SAMPLES_PER_TASK, TRANSLATION_TOPICS)
    generate_summarization_data(NUM_SAMPLES_PER_TASK, SUMMARIZATION_TOPICS)
    generate_sentence_similarity_data(NUM_SAMPLES_PER_TASK, SENTENCE_SIMILARITY_TOPICS)
    generate_text_generation_data(NUM_SAMPLES_PER_TASK, TEXT_GEN_TOPICS)
    generate_style_transfer_data(NUM_SAMPLES_PER_TASK, STYLE_TRANSFER_TOPICS)
    generate_fill_mask_data(NUM_SAMPLES_PER_TASK, FILL_MASK_TOPICS)
    generate_text_ranking_data(NUM_SAMPLES_PER_TASK, TEXT_RANKING_TOPICS)

    print("\nDeepSeek dataset generation process finished.")
