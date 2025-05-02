# -*- coding: utf-8 -*-
"""
Example dataset for Zero-Shot Text Classification.
Classifying text into categories the model hasn't explicitly seen during training,
using candidate labels provided at inference time.
"""

zero_shot_data = [
    {
        "sequence": "รัฐบาลประกาศขึ้นอัตราดอกเบี้ยนโยบายเพื่อควบคุมเงินเฟ้อ",
        "candidate_labels": ["เศรษฐกิจ", "การเมือง", "สังคม", "กีฬา"],
        "expected_label": "เศรษฐกิจ" # For evaluation/reference
    },
    {
        "sequence": "นักแสดงนำชายได้รับรางวัลตุ๊กตาทองคำจากบทบาทล่าสุด",
        "candidate_labels": ["ภาพยนตร์", "ดนตรี", "ละครเวที", "ข่าว"],
        "expected_label": "ภาพยนตร์"
    },
    {
        "sequence": "เปิดตัวแอปพลิเคชันใหม่สำหรับเรียนภาษาต่างประเทศด้วย AI",
        "candidate_labels": ["การศึกษา", "เทคโนโลยี", "สุขภาพ", "การเดินทาง"],
        "expected_label": "เทคโนโลยี" # Could also be การศึกษา depending on focus
    },
     {
        "sequence": "วิธีทำแกงเขียวหวานไก่ให้อร่อยกลมกล่อม",
        "candidate_labels": ["อาหาร", "ท่องเที่ยว", "แฟชั่น", "ยานยนต์"],
        "expected_label": "อาหาร"
    }
]

# Note: For zero-shot, the 'dataset' is often just the sequences.
# Candidate labels are provided during inference.
# Example usage:
# print(zero_shot_data[0]['sequence'])
# print("Candidate Labels:", zero_shot_data[0]['candidate_labels'])
