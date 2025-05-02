# -*- coding: utf-8 -*-
"""
Sample dataset for the Medical domain covering various NLP tasks.
Can also be run to generate medical_data.parquet in the DataOutput directory.
Requires pandas and pyarrow: pip install pandas pyarrow
"""
import os
import pandas as pd # Added import

# --- 1. Summarization ---
medical_summarization_data = [
    {
        "document": "ผู้ป่วยชายอายุ 65 ปี มีประวัติเป็นโรคความดันโลหิตสูงและเบาหวาน เข้ารับการรักษาด้วยอาการเจ็บแน่นหน้าอก ผลการตรวจคลื่นไฟฟ้าหัวใจพบความผิดปกติ เข้าได้กับภาวะกล้ามเนื้อหัวใจขาดเลือดเฉียบพลัน แพทย์ได้ให้การรักษาด้วยยาละลายลิ่มเลือดและทำการสวนหัวใจ พบหลอดเลือดหัวใจตีบ 2 เส้น จึงได้ทำการขยายหลอดเลือดด้วยบอลลูนและใส่ขดลวด หลังการรักษาผู้ป่วยอาการดีขึ้น",
        "summary": "ผู้ป่วยชายสูงอายุ ความดันสูง เบาหวาน เจ็บหน้าอก ตรวจพบกล้ามเนื้อหัวใจขาดเลือด รักษาด้วยยาละลายลิ่มเลือดและสวนหัวใจ พบหลอดเลือดตีบ 2 เส้น ทำบอลลูนใส่ขดลวด อาการดีขึ้น"
    },
    # ... more summarization examples
]

# --- 2. Open QA ---
medical_open_qa_data = [
    {
        "context": "โรคเบาหวานชนิดที่ 2 เป็นภาวะที่ร่างกายไม่สามารถใช้อินซูลินได้อย่างมีประสิทธิภาพ (ภาวะดื้ออินซูลิน) หรือตับอ่อนผลิตอินซูลินได้ไม่เพียงพอ ทำให้ระดับน้ำตาลในเลือดสูงผิดปกติ ปัจจัยเสี่ยงสำคัญคือ น้ำหนักเกิน ขาดการออกกำลังกาย และมีประวัติครอบครัวเป็นเบาหวาน",
        "question": "อะไรคือสาเหตุหลักของโรคเบาหวานชนิดที่ 2?",
        "answer": "เกิดจากภาวะดื้ออินซูลินหรือตับอ่อนผลิตอินซูลินไม่พอ ทำให้น้ำตาลในเลือดสูง ปัจจัยเสี่ยงคือน้ำหนักเกิน ขาดการออกกำลังกาย และกรรมพันธุ์"
    },
    # ... more open QA examples
]

# --- 3. Close QA (Extractive) ---
medical_close_qa_data = [
    {
        "context": "วัคซีนป้องกันไข้หวัดใหญ่แนะนำให้ฉีดปีละ 1 ครั้ง โดยเฉพาะในกลุ่มเสี่ยง เช่น ผู้สูงอายุ ผู้มีโรคประจำตัว เด็กเล็ก และหญิงตั้งครรภ์ เพื่อลดความรุนแรงของโรคและภาวะแทรกซ้อน",
        "question": "กลุ่มใดบ้างที่แนะนำให้ฉีดวัคซีนไข้หวัดใหญ่เป็นพิเศษ?",
        "answer_text": "ผู้สูงอายุ ผู้มีโรคประจำตัว เด็กเล็ก และหญิงตั้งครรภ์"
    },
    # ... more close QA examples
]

# --- 4. Classification ---
# Example labels: symptom_check, appointment_booking, medication_query, general_health_info
medical_classification_data = [
    {"text": "มีไข้สูง ไอ เจ็บคอ ควรทำอย่างไร", "label": "symptom_check"},
    {"text": "ต้องการนัดพบแพทย์เฉพาะทางโรคหัวใจ", "label": "appointment_booking"},
    {"text": "ยาพาราเซตามอลกินติดต่อกันนานๆ ได้ไหม", "label": "medication_query"},
    {"text": "วิธีดูแลสุขภาพให้แข็งแรง", "label": "general_health_info"},
    # ... more classification examples
]

# --- 5. Creative Writing ---
medical_creative_writing_data = [
    {
        "prompt": "เขียนเรื่องสั้นเกี่ยวกับประสบการณ์ของผู้ป่วยที่เพิ่งได้รับการวินิจฉัยว่าเป็นโรคร้ายแรง แต่มีกำลังใจที่จะต่อสู้",
        "generated_text": "วันที่หมอบอกว่าผมเป็นมะเร็งระยะที่สาม โลกทั้งใบเหมือนหยุดหมุน ความกลัวถาโถมเข้ามา แต่เมื่อมองหน้าลูกสาวตัวน้อย ผมรู้ว่าต้องสู้ ผมจะสู้เพื่อเธอ เพื่อครอบครัว แม้หนทางจะยากลำบาก แต่ผมจะไม่ยอมแพ้..."
    },
    {
        "prompt": "เขียนบทความให้ความรู้เกี่ยวกับการป้องกันโรคหัวใจสำหรับคนทั่วไป",
        "generated_text": "โรคหัวใจ ภัยเงียบที่ป้องกันได้ เริ่มต้นง่ายๆ ด้วยการปรับเปลี่ยนพฤติกรรม เลือกทานอาหารที่มีประโยชน์ ออกกำลังกายสม่ำเสมอ ควบคุมน้ำหนัก งดสูบบุหรี่ และตรวจสุขภาพประจำปี เพื่อหัวใจที่แข็งแรงของคุณ..."
    },
    # ... more creative writing examples
]

# --- 6. Brainstorming ---
medical_brainstorming_data = [
    {
        "topic": "แนวทางการพัฒนาระบบ Telemedicine ในพื้นที่ห่างไกล",
        "ideas": [
            "ใช้แอปพลิเคชันบนมือถือสำหรับการปรึกษาทางไกล",
            "จัดตั้งศูนย์สุขภาพชุมชนพร้อมอุปกรณ์เชื่อมต่อ",
            "อบรมอาสาสมัครสาธารณสุข (อสม.) ให้ใช้เทคโนโลยี",
            "พัฒนาระบบส่งยาทางไปรษณีย์หรือโดรน",
            "ร่วมมือกับหน่วยงานท้องถิ่นในการเข้าถึงอินเทอร์เน็ต"
        ]
    },
    # ... more brainstorming examples
]

# --- 7. Multiple Choice QA ---
medical_mcq_data = [
    {
        "context": "การปฐมพยาบาลเบื้องต้นสำหรับผู้ที่หยุดหายใจคือการทำ CPR (Cardiopulmonary Resuscitation) ซึ่งประกอบด้วยการกดหน้าอกและการช่วยหายใจ อัตราส่วนการกดหน้าอกต่อการช่วยหายใจที่แนะนำสำหรับผู้ใหญ่คือ 30:2",
        "question": "อัตราส่วนการกดหน้าอกต่อการช่วยหายใจในการทำ CPR สำหรับผู้ใหญ่คือเท่าใด?",
        "choices": ["15:1", "30:2", "15:2", "30:1"],
        "answer_index": 1 # Index of "30:2"
    },
    # ... more multiple choice QA examples
]

# You can combine all data into one dictionary if needed
medical_domain_data = {
    "summarization": medical_summarization_data,
    "open_qa": medical_open_qa_data,
    "close_qa": medical_close_qa_data,
    "classification": medical_classification_data,
    "creative_writing": medical_creative_writing_data,
    "brainstorming": medical_brainstorming_data,
    "multiple_choice_qa": medical_mcq_data,
}

# Function to preprocess data into a standard format for Parquet
def preprocess_data_for_parquet(domain_name, task_name, data_list):
    processed = []
    for item in data_list:
        input_text = ""
        target_text = ""
        prefix = f"{domain_name} {task_name}: " # Keep prefix for potential model use

        if task_name == "summarization":
            input_text = prefix + item.get("document", "")
            target_text = item.get("summary", "")
        elif task_name == "open_qa":
            input_text = prefix + f"question: {item.get('question', '')} context: {item.get('context', '')}"
            target_text = item.get("answer", "")
        elif task_name == "close_qa":
            input_text = prefix + f"question: {item.get('question', '')} context: {item.get('context', '')}"
            target_text = item.get("answer_text", "")
        elif task_name == "classification":
            input_text = prefix + item.get("text", "")
            target_text = item.get("label", "")
        elif task_name == "creative_writing":
            input_text = prefix + item.get("prompt", "")
            target_text = item.get("generated_text", "")
        elif task_name == "brainstorming":
            input_text = prefix + item.get("topic", "")
            target_text = "\n".join(item.get("ideas", []))
        elif task_name == "multiple_choice_qa":
            choices_str = " | ".join(item.get("choices", []))
            input_text = prefix + f"question: {item.get('question', '')} choices: {choices_str} context: {item.get('context', '')}"
            answer_idx = item.get("answer_index")
            if answer_idx is not None and item.get("choices"):
                target_text = item["choices"][answer_idx]

        if input_text and target_text:
            # Add domain and task for potential filtering later
            processed.append({
                "domain": domain_name,
                "task": task_name,
                "input_text": input_text,
                "target_text": target_text
            })
    return processed


if __name__ == '__main__':
    # Example of accessing data
    print("--- Medical Domain Dataset Samples ---")
    print(f"\nSummarization Example:\n{medical_domain_data['summarization'][0]}")
    print(f"\nOpen QA Example:\n{medical_domain_data['open_qa'][0]}")
    print(f"\nClose QA Example:\n{medical_domain_data['close_qa'][0]}")
    print(f"\nClassification Example:\n{medical_domain_data['classification'][0]}")
    print(f"\nCreative Writing Example:\n{medical_domain_data['creative_writing'][0]}")
    print(f"\nBrainstorming Example:\n{medical_domain_data['brainstorming'][0]}")
    print(f"\nMultiple Choice QA Example:\n{medical_domain_data['multiple_choice_qa'][0]}")

    # --- Generate Parquet File ---
    print("\n--- Generating Parquet file ---")
    all_records = []
    domain_name = "medical"
    for task, data in medical_domain_data.items():
        all_records.extend(preprocess_data_for_parquet(domain_name, task, data))

    if all_records:
        df = pd.DataFrame(all_records)
        output_filename = "medical_data.parquet"
        # Define output directory relative to the script's location
        script_dir = os.path.dirname(__file__)
        output_dir = os.path.abspath(os.path.join(script_dir, '..', 'DataOutput')) # Go up one level, then into DataOutput
        # Create DataOutput directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        try:
            df.to_parquet(output_path, index=False)
            print(f"Successfully generated {output_filename} at {output_path}")
            print(f"DataFrame Info:\n{df.info()}")
            print(f"\nFirst 5 rows:\n{df.head().to_string()}")
        except Exception as e:
            print(f"Error saving Parquet file: {e}")
            print("Please ensure 'pandas' and 'pyarrow' are installed (`pip install pandas pyarrow`)")
    else:
        print("No records processed, Parquet file not generated.")
