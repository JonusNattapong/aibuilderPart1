# -*- coding: utf-8 -*-
"""
Example dataset for Token Classification (e.g., Named Entity Recognition - NER).
Assigns a label to each token (word/subword) in a sentence.
Labels: O (Outside), B-PER (Beginning of Person), I-PER (Inside of Person),
        B-ORG (Beginning of Organization), I-ORG (Inside of Organization),
        B-LOC (Beginning of Location), I-LOC (Inside of Location)
"""

token_classification_data = [
    {
        "tokens": ["คุณ", "สมชาย", "ทำงาน", "ที่", "บริษัท", "ก้าวหน้า", "จำกัด", "ใน", "กรุงเทพมหานคร"],
        "ner_tags": ["O", "B-PER", "O", "O", "B-ORG", "I-ORG", "I-ORG", "O", "B-LOC"]
    },
    {
        "tokens": ["นายก", "เศรษฐา", "ทวีสิน", "เดินทาง", "ไป", "ประเทศ", "ญี่ปุ่น"],
        "ner_tags": ["O", "B-PER", "I-PER", "O", "O", "B-LOC", "I-LOC"]
    },
    {
        "tokens": ["ธนาคาร", "กรุงไทย", "ประกาศ", "ผลประกอบการ", "ไตรมาส", "ล่าสุด"],
        "ner_tags": ["B-ORG", "I-ORG", "O", "O", "O", "O"]
    },
    {
        "tokens": ["ฉัน", "อยาก", "ไป", "เที่ยว", "เชียงใหม่", "กับ", "มานี"],
        "ner_tags": ["O", "O", "O", "O", "B-LOC", "O", "B-PER"]
    }
]

# Note: This format is simplified. Real-world datasets often use BIO or BILOU schemes
# and might be pre-tokenized or require tokenization aligned with labels.
# Example usage:
# print(token_classification_data[0])
