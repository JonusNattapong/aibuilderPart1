# -*- coding: utf-8 -*-
"""
Sample dataset for the Medical domain covering various NLP tasks.
"""

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
