# -*- coding: utf-8 -*-
"""
Example dataset for Machine Translation.
Pairs of sentences in source and target languages.
"""

translation_data = [
    # Thai (th) to English (en)
    {"th": "สวัสดีตอนเช้า", "en": "Good morning"},
    {"th": "ขอบคุณมากครับ", "en": "Thank you very much"},
    {"th": "แมวตัวนี้น่ารักมาก", "en": "This cat is very cute"},
    {"th": "ปัญญาประดิษฐ์กำลังเปลี่ยนแปลงโลก", "en": "Artificial intelligence is changing the world"},

    # English (en) to Thai (th)
    {"en": "Where is the nearest hospital?", "th": "โรงพยาบาลที่ใกล้ที่สุดอยู่ที่ไหน"},
    {"en": "I would like to order Pad Thai.", "th": "ฉันต้องการสั่งผัดไทย"},
    {"en": "The weather is nice today.", "th": "วันนี้อากาศดี"},
    {"en": "Machine learning is a subset of AI.", "th": "การเรียนรู้ของเครื่องเป็นส่วนหนึ่งของ AI"}
]

# Example usage:
# import pandas as pd
# df = pd.DataFrame(translation_data)
# print(df.head())
