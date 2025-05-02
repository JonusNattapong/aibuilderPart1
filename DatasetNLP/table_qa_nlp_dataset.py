# -*- coding: utf-8 -*-
"""
Example dataset for Table Question Answering.
Answering questions based on information presented in a table.
"""

table_qa_data = [
    {
        "table": {
            "header": ["ชื่อนักเรียน", "วิชา", "คะแนน"],
            "rows": [
                ["สมชาย", "คณิตศาสตร์", 85],
                ["สมหญิง", "คณิตศาสตร์", 92],
                ["สมชาย", "วิทยาศาสตร์", 78],
                ["สมหญิง", "วิทยาศาสตร์", 88],
                ["วิชัย", "คณิตศาสตร์", 75]
            ]
        },
        "question": "สมหญิงได้คะแนนวิชาวิทยาศาสตร์เท่าไหร่?",
        "answer_coordinates": [[3, 2]], # Row index 3, Column index 2 (88)
        "answer_text": "88"
    },
    {
        "table": {
            "header": ["จังหวัด", "ภาค", "ประชากร (ล้านคน)"],
            "rows": [
                ["กรุงเทพมหานคร", "กลาง", 10.5],
                ["เชียงใหม่", "เหนือ", 1.8],
                ["สงขลา", "ใต้", 1.4],
                ["ขอนแก่น", "ตะวันออกเฉียงเหนือ", 1.8]
            ]
        },
        "question": "จังหวัดใดอยู่ในภาคใต้?",
        "answer_coordinates": [[2, 0]], # Row index 2, Column index 0 (สงขลา)
        "answer_text": "สงขลา"
    },
    {
        "table": {
            "header": ["สินค้า", "ราคา (บาท)", "จำนวนคงเหลือ"],
            "rows": [
                ["ปากกา", 15, 100],
                ["ดินสอ", 10, 150],
                ["ยางลบ", 5, 200],
                ["ไม้บรรทัด", 20, 50]
            ]
        },
        "question": "สินค้าใดมีราคาแพงที่สุด?",
        "answer_coordinates": [[3, 0]], # Row index 3, Column index 0 (ไม้บรรทัด)
        "answer_text": "ไม้บรรทัด"
    }
]

# Example usage:
# import pandas as pd
# table_df = pd.DataFrame(table_qa_data[0]['table']['rows'], columns=table_qa_data[0]['table']['header'])
# print("Table:\n", table_df)
# print("Question:", table_qa_data[0]['question'])
# print("Answer:", table_qa_data[0]['answer_text'])
