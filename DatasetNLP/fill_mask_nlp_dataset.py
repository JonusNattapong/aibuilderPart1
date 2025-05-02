# -*- coding: utf-8 -*-
"""
Example sentences for the Fill-Mask task.
The model predicts the word(s) that should fill the masked position(s).
The mask token depends on the tokenizer (e.g., <mask>, [MASK]).
"""

fill_mask_sentences = [
    "กรุงเทพมหานครเป็น <mask> ของประเทศไทย", # Expected: เมืองหลวง
    "ฉันชอบดื่มกาแฟในตอน <mask>", # Expected: เช้า
    "แมวเป็นสัตว์เลี้ยงที่ <mask> มาก", # Expected: น่ารัก
    "การออกกำลังกายทำให้สุขภาพ <mask>", # Expected: แข็งแรง, ดี
    "วันนี้อากาศ <mask> แดดแรง", # Expected: ร้อน
    "เขาอ่าน <mask> ก่อนนอนทุกคืน" # Expected: หนังสือ
]

# Example usage:
# print(fill_mask_sentences)
# A model would predict likely words to replace '<mask>'.
