# -*- coding: utf-8 -*-
import uuid

cot_reasoning_data = [
    # --- 1. Chain of Thought (CoT) ---
    {
        "id": str(uuid.uuid4()),
        "task": "Question Answering (Math)",
        "reasoning_type_tested": "CoT",
        "input_data": {
            "question": "ร้านค้ามีแอปเปิ้ล 5 ลัง แต่ละลังมี 12 ผล ขายไปแล้ว 20 ผล เหลือแอปเปิ้ลกี่ผล?",
            "context": None # Context might not be needed if the question is self-contained
        },
        "expected_output": {
            "reasoning_steps": [
                "1. คำนวณจำนวนแอปเปิ้ลทั้งหมด: 5 ลัง * 12 ผล/ลัง = 60 ผล",
                "2. คำนวณจำนวนแอปเปิ้ลที่เหลือ: 60 ผล - 20 ผล = 40 ผล"
            ],
            "final_answer": "40 ผล"
        }
    },
    {
        "id": str(uuid.uuid4()),
        "task": "Question Answering (Logic)",
        "reasoning_type_tested": "CoT",
        "input_data": {
            "question": "ถ้าฝนตกแล้วพื้นจะเปียก วันนี้พื้นเปียก สรุปได้หรือไม่ว่าฝนตก?",
            "context": None
        },
        "expected_output": {
            "reasoning_steps": [
                "1. เงื่อนไขคือ 'ถ้าฝนตก -> พื้นเปียก'",
                "2. ข้อมูลคือ 'พื้นเปียก'",
                "3. การที่พื้นเปียกไม่ได้หมายความว่าฝนตกเสมอไป อาจเกิดจากสาเหตุอื่น (เช่น รดน้ำต้นไม้)",
                "4. ดังนั้น สรุปไม่ได้แน่นอนว่าฝนตก"
            ],
            "final_answer": "สรุปไม่ได้แน่นอนว่าฝนตก เพราะอาจมีสาเหตุอื่นที่ทำให้พื้นเปียก"
        }
    },
    # Add Text Classification and Token Classification examples previously under "Other Task Examples"
    {
        "id": str(uuid.uuid4()),
        "task": "Text Classification",
        "reasoning_type_tested": "CoT", # Can use CoT to explain classification
        "input_data": {
            "text": "ได้รับสินค้าแล้ว แต่ขนาดไม่ตรงกับที่สั่ง ต้องการทำเรื่องเปลี่ยนคืน",
            "categories": ["สอบถามข้อมูล", "แจ้งปัญหา/ร้องเรียน", "เสนอแนะ", "อื่นๆ"]
        },
        "expected_output": {
            "reasoning_steps": [
                "1. ข้อความกล่าวถึง 'ได้รับสินค้าแล้ว'",
                "2. ระบุปัญหา 'ขนาดไม่ตรง'",
                "3. แสดงความต้องการ 'เปลี่ยนคืน'",
                "4. เนื้อหาเป็นการแจ้งปัญหาและต้องการดำเนินการแก้ไข",
            ],
            "final_answer": "แจ้งปัญหา/ร้องเรียน"
        }
    },
    {
        "id": str(uuid.uuid4()),
        "task": "Token Classification (NER)",
        "reasoning_type_tested": "CoT", # Can use CoT to justify tagging
        "input_data": {
            "text": "คุณสมชายเดินทางไปประชุมที่กรุงเทพฯ เมื่อวานนี้",
        },
        "expected_output": {
            "reasoning_steps": [
                "'คุณสมชาย' เป็นชื่อคน -> PERSON",
                "'กรุงเทพฯ' เป็นชื่อสถานที่ -> LOCATION",
                "'เมื่อวานนี้' เป็นการอ้างอิงถึงเวลา -> DATE"
            ],
            "tokens_and_tags": [
                ("คุณสมชาย", "B-PERSON"),
                ("เดินทางไปประชุมที่", "O"),
                ("กรุงเทพฯ", "B-LOCATION"),
                ("เมื่อวานนี้", "B-DATE")
            ]
        }
    },
    {
        "id": str(uuid.uuid4()),
        "task": "Multiple Choice",
        "reasoning_type_tested": "CoT",
        "input_data": {
            "question": "ข้อใดคือเมืองหลวงของประเทศไทย?",
            "choices": ["เชียงใหม่", "กรุงเทพมหานคร", "ภูเก็ต", "ขอนแก่น"],
            "context": None
        },
        "expected_output": {
            "reasoning_steps": [
                "1. คำถามถามถึงเมืองหลวงของประเทศไทย",
                "2. จากความรู้ทั่วไป เมืองหลวงคือกรุงเทพมหานคร",
                "3. ตรวจสอบตัวเลือก: เชียงใหม่ (ไม่ใช่), กรุงเทพมหานคร (ใช่), ภูเก็ต (ไม่ใช่), ขอนแก่น (ไม่ใช่)",
            ],
            "final_answer_index": 1,
            "final_answer_text": "กรุงเทพมหานคร"
        }
    },
]
