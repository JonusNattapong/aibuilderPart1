# -*- coding: utf-8 -*-
"""
Example dataset for general Text-to-Text Generation tasks.
Input text is transformed into an output text based on a task prefix or instruction.
Examples cover various tasks like simplification, style transfer, correction etc.
"""

text2text_data = [
    # Task: Simplify
    {"input_text": "simplify: ปรากฏการณ์โลกร้อนส่งผลกระทบต่อระบบนิเวศน์อย่างมีนัยสำคัญ",
     "target_text": "โลกร้อนทำให้ธรรมชาติเปลี่ยนไปมาก"},
    # Task: Correct Grammar
    {"input_text": "correct: ฉันไปตลาดเมื่อวานนี้ซื้อผลไม้เยอะแยะ",
     "target_text": "เมื่อวานนี้ฉันไปตลาด ซื้อผลไม้มาเยอะแยะ"},
    # Task: Formal to Informal
    {"input_text": "to_informal: กรุณารอสักครู่ เจ้าหน้าที่กำลังดำเนินการ",
     "target_text": "รอแป๊บนะ กำลังทำให้"},
    # Task: Question Generation from Statement
    {"input_text": "generate_question: กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย",
     "target_text": "เมืองหลวงของประเทศไทยคืออะไร?"},
     # Task: Keyword Extraction
    {"input_text": "extract_keywords: เรียนรู้เทคนิคการถ่ายภาพด้วยกล้องดิจิทัลสำหรับมือใหม่",
     "target_text": "ถ่ายภาพ, กล้องดิจิทัล, มือใหม่"}
]

# Example usage:
# import pandas as pd
# df = pd.DataFrame(text2text_data)
# print(df.head())
