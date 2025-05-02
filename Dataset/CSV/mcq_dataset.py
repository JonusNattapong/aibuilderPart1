import csv
import uuid
import os
import random

# ตรวจสอบและสร้างไดเรกทอรี DataOutput หากยังไม่มี
output_dir = 'DataOutput'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ตัวอย่างข้อมูล MCQ (Context, Question, Choices, Answer Label)
mcq_data = [
    # Politics
    {
        "context": "การเลือกตั้งทั่วไปในประเทศไทยครั้งล่าสุดจัดขึ้นเมื่อวันที่ 14 พฤษภาคม พ.ศ. 2566 เป็นการเลือกตั้งสมาชิกสภาผู้แทนราษฎรจำนวน 500 คน",
        "question": "การเลือกตั้งทั่วไปครั้งล่าสุดในประเทศไทยจัดขึ้นเมื่อใด?",
        "choices": ["14 พฤษภาคม 2562", "14 พฤษภาคม 2566", "24 มีนาคม 2562", "24 มีนาคม 2566"],
        "answer_label": "14 พฤษภาคม 2566"
    },
    {
        "context": "คณะรัฐมนตรีไทยประกอบด้วยนายกรัฐมนตรี 1 คน และรัฐมนตรีอื่นอีกไม่เกิน 35 คน ซึ่งพระมหากษัตริย์ทรงแต่งตั้งตามคำแนะนำของนายกรัฐมนตรี",
        "question": "คณะรัฐมนตรีไทยมีรัฐมนตรีได้ไม่เกินกี่คน (ไม่รวมนายกรัฐมนตรี)?",
        "choices": ["30 คน", "35 คน", "40 คน", "45 คน"],
        "answer_label": "35 คน"
    },
    # Science
    {
        "context": "ดาวเคราะห์ในระบบสุริยะที่อยู่ใกล้ดวงอาทิตย์มากที่สุดคือดาวพุธ รองลงมาคือดาวศุกร์ โลก และดาวอังคาร ตามลำดับ",
        "question": "ดาวเคราะห์ดวงใดอยู่ลำดับที่ 3 จากดวงอาทิตย์?",
        "choices": ["ดาวพุธ", "ดาวศุกร์", "โลก", "ดาวอังคาร"],
        "answer_label": "โลก"
    },
    {
        "context": "น้ำบริสุทธิ์มีสูตรทางเคมีคือ H₂O ประกอบด้วยไฮโดรเจน 2 อะตอม และออกซิเจน 1 อะตอม",
        "question": "ธาตุใดเป็นส่วนประกอบของโมเลกุลน้ำ?",
        "choices": ["คาร์บอน", "ไนโตรเจน", "ออกซิเจน", "คลอรีน"],
        "answer_label": "ออกซิเจน"
    },
    # Sports
    {
        "context": "การแข่งขันฟุตบอลโลก (FIFA World Cup) จัดขึ้นทุกๆ 4 ปี โดยครั้งล่าสุดจัดขึ้นในปี 2022 ที่ประเทศกาตาร์ และทีมชาติอาร์เจนตินาเป็นผู้ชนะเลิศ",
        "question": "ทีมชาติใดเป็นผู้ชนะเลิศฟุตบอลโลกปี 2022?",
        "choices": ["ฝรั่งเศส", "บราซิล", "เยอรมนี", "อาร์เจนตินา"],
        "answer_label": "อาร์เจนตินา"
    },
    {
        "context": "กีฬาโอลิมปิกฤดูร้อนครั้งต่อไป (ครั้งที่ 33) จะจัดขึ้นในปี 2024 ที่กรุงปารีส ประเทศฝรั่งเศส",
        "question": "กีฬาโอลิมปิกฤดูร้อนปี 2024 จะจัดขึ้นที่เมืองใด?",
        "choices": ["ลอสแอนเจลิส", "โตเกียว", "ปารีส", "ลอนดอน"],
        "answer_label": "ปารีส"
    }
    # เพิ่มเติมตัวอย่างตามต้องการ
]

# สร้างรายการข้อมูลพร้อม ID และจัดรูปแบบ choices
rows = []
max_choices = 4 # กำหนดจำนวน choice สูงสุดสำหรับ header
header = ['id', 'context', 'question'] + [f'choice_{i}' for i in range(max_choices)] + ['answer_label']

for item in mcq_data:
    row_data = [
        str(uuid.uuid4()),
        item.get("context", ""), # ใส่ context หรือสตริงว่างถ้าไม่มี
        item["question"]
    ]

    choices = item["choices"]
    # สับเปลี่ยนตำแหน่ง choice เพื่อให้คำตอบไม่ได้อยู่ตำแหน่งเดิมเสมอ
    random.shuffle(choices)

    # เติม choice ให้ครบตาม max_choices
    row_data.extend(choices)
    row_data.extend([""] * (max_choices - len(choices))) # เติมสตริงว่างถ้า choice ไม่ครบ

    row_data.append(item["answer_label"])
    rows.append(row_data)


# บันทึกเป็นไฟล์ CSV ในโฟลเดอร์ DataOutput
output_file = os.path.join(output_dir, 'thai_dataset_mcq.csv')
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"Created {output_file}")
