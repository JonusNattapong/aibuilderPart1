# -*- coding: utf-8 -*-
"""
Example dataset for Question Answering (Extractive).
Finding the answer to a question within a given context document.
"""

qa_data = [
    {
        "id": "qa_nlp_001",
        "context": "ปัญญาประดิษฐ์ (AI) เป็นสาขาหนึ่งของวิทยาการคอมพิวเตอร์ที่มุ่งเน้นการสร้างเครื่องจักรที่สามารถทำงานที่ต้องใช้สติปัญญาของมนุษย์ เช่น การเรียนรู้ การแก้ปัญหา และการตัดสินใจ AI ถูกนำไปใช้ในหลากหลายอุตสาหกรรม รวมถึงการแพทย์ การเงิน และการขนส่ง",
        "question": "AI ย่อมาจากอะไร?",
        "answers": {
            "text": ["ปัญญาประดิษฐ์"],
            "answer_start": [0]
        }
    },
    {
        "id": "qa_nlp_002",
        "context": "ประเทศไทยมีสภาพภูมิอากาศแบบร้อนชื้น แบ่งออกเป็น 3 ฤดูหลัก ได้แก่ ฤดูร้อน ฤดูฝน และฤดูหนาว ฤดูร้อนเริ่มตั้งแต่กลางเดือนกุมภาพันธ์ถึงกลางเดือนพฤษภาคม",
        "question": "ฤดูร้อนในประเทศไทยเริ่มเมื่อไหร่?",
        "answers": {
            "text": ["กลางเดือนกุมภาพันธ์"],
            "answer_start": [106]
        }
    },
    {
        "id": "qa_nlp_003",
        "context": "แมวเป็นสัตว์เลี้ยงลูกด้วยนมที่ได้รับความนิยมอย่างสูง มีหลากหลายสายพันธุ์ทั่วโลก พวกมันเป็นสัตว์นักล่าโดยธรรมชาติ และมีสัญชาตญาณในการปีนป่ายและกระโดด",
        "question": "แมวเป็นสัตว์ประเภทใด?",
        "answers": {
            "text": ["สัตว์เลี้ยงลูกด้วยนม"],
            "answer_start": [10]
        }
    }
]

# Example usage:
# import pandas as pd
# df = pd.json_normalize(qa_data, 'answers', ['id', 'context', 'question'], record_prefix='answer_')
# print(df.head())
