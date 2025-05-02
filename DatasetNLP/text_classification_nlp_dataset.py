# -*- coding: utf-8 -*-
"""
Example dataset for Text Classification task.
Assigns a label (e.g., category) to a given text.
"""

text_classification_data = [
    # หมวดหมู่: ข่าวการเมือง (politics), ข่าวกีฬา (sports), ข่าวบันเทิง (entertainment), ข่าวเทคโนโลยี (technology)
    {"text": "นายกรัฐมนตรีแถลงนโยบายใหม่เน้นกระตุ้นเศรษฐกิจฐานราก", "label": "politics"},
    {"text": "ทีมชาติไทยชนะการแข่งขันฟุตบอลนัดล่าสุดด้วยสกอร์ 2-1", "label": "sports"},
    {"text": "เปิดตัวสมาร์ทโฟนรุ่นใหม่ล่าสุดพร้อมกล้องความละเอียดสูง", "label": "technology"},
    {"text": "นักแสดงชื่อดังเข้ารับรางวัลภาพยนตร์ยอดเยี่ยมแห่งปี", "label": "entertainment"},
    {"text": "กกต. เตรียมจัดการเลือกตั้งซ่อมในเดือนหน้า", "label": "politics"},
    {"text": "ผลการแข่งขันเทนนิสแกรนด์สแลมรอบชิงชนะเลิศ", "label": "sports"},
    {"text": "ภาพยนตร์เรื่องใหม่ทำรายได้ทะลุร้อยล้านบาท", "label": "entertainment"},
    {"text": "บริษัทเทคโนโลยีเปิดตัว AI ช่วยวิเคราะห์ข้อมูลทางการแพทย์", "label": "technology"},
]

# Example usage:
# import pandas as pd
# df = pd.DataFrame(text_classification_data)
# print(df.head())
# df.to_csv("text_classification_data.csv", index=False, encoding='utf-8-sig')
