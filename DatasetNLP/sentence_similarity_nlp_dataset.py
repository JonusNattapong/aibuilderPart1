# -*- coding: utf-8 -*-
"""
Example dataset for Sentence Similarity (or Semantic Textual Similarity - STS).
Pairs of sentences with a similarity score (e.g., 0-1 or 0-5).
Higher score means more similar meaning.
"""

sentence_similarity_data = [
    # Score range 0.0 (no similarity) to 1.0 (identical meaning)
    {"sentence1": "รถยนต์คันนี้สีแดง", "sentence2": "รถคันนี้มีสีแดง", "similarity_score": 1.0},
    {"sentence1": "วันนี้อากาศดีมาก", "sentence2": "อากาศวันนี้แจ่มใส", "similarity_score": 0.9},
    {"sentence1": "เขากำลังกินข้าว", "sentence2": "เขากำลังทานอาหาร", "similarity_score": 0.8},
    {"sentence1": "แมวกำลังนอนหลับ", "sentence2": "สุนัขกำลังวิ่งเล่น", "similarity_score": 0.1},
    {"sentence1": "วิธีเดินทางไปสนามบิน", "sentence2": "ร้านอาหารอร่อยในกรุงเทพ", "similarity_score": 0.0},
    {"sentence1": "การลงทุนมีความเสี่ยง", "sentence2": "ควรศึกษาข้อมูลก่อนลงทุน", "similarity_score": 0.6},
    {"sentence1": "ตั๋วเครื่องบินไปญี่ปุ่นราคาเท่าไหร่", "sentence2": "ราคาตั๋วเครื่องบินไปเกาหลี", "similarity_score": 0.5}, # Similar topic, different specifics
]

# Example usage:
# import pandas as pd
# df = pd.DataFrame(sentence_similarity_data)
# print(df.head())
