# -*- coding: utf-8 -*-
import uuid

toolformer_reasoning_data = [
    # --- 5. Toolformer / Tool-augmented LLMs ---
    {
        "id": str(uuid.uuid4()),
        "task": "Question Answering (Calculation)",
        "reasoning_type_tested": "Toolformer",
        "input_data": {
            "question": "พื้นที่วงกลมรัศมี 7 เซนติเมตร เป็นเท่าใด (ใช้ pi ≈ 3.14)?",
        },
        "expected_output": {
            "reasoning_process_simulation": [
                {"thought": "ต้องการคำนวณพื้นที่วงกลม สูตรคือ πr²"},
                {"action": "Calculator(3.14 * 7 * 7)"},
                {"observation": "153.86"},
                {"thought": "ได้ผลลัพธ์แล้ว"},
            ],
            "final_answer": "153.86 ตารางเซนติเมตร"
        }
    },
    {
        "id": str(uuid.uuid4()),
        "task": "Translation",
        "reasoning_type_tested": "Toolformer",
        "input_data": {
            "text": "สวัสดีครับ",
            "target_language": "English"
        },
        "expected_output": {
            "reasoning_process_simulation": [
                {"thought": "ต้องการแปล 'สวัสดีครับ' เป็นภาษาอังกฤษ"},
                {"action": "Translate('สวัสดีครับ', target='en')"},
                {"observation": "Hello / Hi"},
                {"thought": "ได้คำแปลแล้ว"},
            ],
            "final_answer": "Hello" # Or Hi, depending on context/formality preference
        }
    },
]
