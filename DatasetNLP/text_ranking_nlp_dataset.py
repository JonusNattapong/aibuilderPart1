# -*- coding: utf-8 -*-
"""
Example dataset for Text Ranking (or Passage Ranking).
Given a query, rank a list of passages based on relevance.
"""

text_ranking_data = [
    {
        "query": "ประโยชน์ของ AI ในการแพทย์",
        "positive_passage": "AI ช่วยแพทย์วินิจฉัยโรคจากภาพถ่ายทางการแพทย์ได้แม่นยำขึ้น และช่วยพัฒนายาใหม่ๆ", # Highly relevant
        "negative_passages": [
            "AI ถูกนำไปใช้ในรถยนต์ไร้คนขับเพื่อเพิ่มความปลอดภัย", # Irrelevant
            "การแพทย์แผนไทยใช้สมุนไพรในการรักษาโรค", # Irrelevant
            "ความสำคัญของการตรวจสุขภาพประจำปี" # Slightly related topic (health) but not AI
        ]
    },
    {
        "query": "วิธีลดน้ำหนักอย่างปลอดภัย",
        "positive_passage": "การลดน้ำหนักที่ดีควรควบคุมอาหารควบคู่กับการออกกำลังกายสม่ำเสมอ และปรึกษาผู้เชี่ยวชาญ", # Highly relevant
        "negative_passages": [
            "สูตรเค้กช็อกโกแลตทำง่ายๆ", # Irrelevant
            "เทรนด์แฟชั่นล่าสุดสำหรับฤดูร้อน", # Irrelevant
            "การอดอาหารเพื่อลดน้ำหนักอย่างรวดเร็วอาจส่งผลเสียต่อสุขภาพ" # Related but focuses on negative aspect/alternative
        ]
    },
    {
        "query": "สถานที่ท่องเที่ยวในเชียงใหม่",
        "positive_passage": "ดอยสุเทพเป็นวัดสำคัญและจุดชมวิวเมืองเชียงใหม่ที่นักท่องเที่ยวนิยมไป", # Highly relevant
        "negative_passages": [
            "ตลาดน้ำดำเนินสะดวกอยู่ที่จังหวัดราชบุรี", # Irrelevant (wrong location)
            "ประวัติศาสตร์ของอาณาจักรล้านนา", # Related background but not a specific attraction
            "โรงแรมหรูในกรุงเทพมหานคร" # Irrelevant
        ]
    }
]

# Note: Real-world ranking datasets often have graded relevance (e.g., highly relevant, relevant, irrelevant)
# or just pairs of (query, relevant_passage, irrelevant_passage).
# Example usage:
# print("Query:", text_ranking_data[0]['query'])
# print("Relevant:", text_ranking_data[0]['positive_passage'])
# print("Irrelevant:", text_ranking_data[0]['negative_passages'][0])
