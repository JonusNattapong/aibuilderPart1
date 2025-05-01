import csv
import uuid
import os
import json # ใช้สำหรับแปลง list ของ entities เป็น string

# ตรวจสอบและสร้างไดเรกทอรี DataOutput หากยังไม่มี
output_dir = 'DataOutput'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ตัวอย่างข้อมูล NER (Sentence, Entities)
# Entities format: [(entity_text, entity_label, start_char, end_char), ...]
ner_data = [
    {
        "sentence": "นายสมศักดิ์เดินทางไปเชียงใหม่เมื่อวันที่ 10 ตุลาคม",
        "entities": [("นายสมศักดิ์", "PERSON", 0, 10), ("เชียงใหม่", "LOCATION", 20, 28), ("วันที่ 10 ตุลาคม", "DATE", 37, 51)]
    },
    {
        "sentence": "บริษัท Google ประกาศเปิดตัวผลิตภัณฑ์ใหม่ในงาน Google I/O",
        "entities": [("Google", "ORGANIZATION", 7, 13), ("Google I/O", "EVENT", 45, 55)] # ใช้ EVENT หรือ MISC ก็ได้
    },
    {
        "sentence": "ลิซ่า Blackpink เกิดที่จังหวัดบุรีรัมย์ ประเทศไทย",
        "entities": [("ลิซ่า Blackpink", "PERSON", 0, 15), ("บุรีรัมย์", "LOCATION", 26, 33), ("ประเทศไทย", "LOCATION", 42, 51)]
    },
    {
        "sentence": "การประชุมสุดยอดผู้นำเอเปคจัดขึ้นที่กรุงเทพฯ ในปี 2565",
        "entities": [("เอเปค", "ORGANIZATION", 24, 28), ("กรุงเทพฯ", "LOCATION", 41, 48), ("ปี 2565", "DATE", 53, 60)]
    },
    {
        "sentence": "ธนาคารแห่งประเทศไทยคาดการณ์ GDP ปีหน้าจะโต 3.5%",
        "entities": [("ธนาคารแห่งประเทศไทย", "ORGANIZATION", 0, 20), ("ปีหน้า", "DATE", 31, 36), ("3.5%", "PERCENT", 44, 48)] # เพิ่ม PERCENT เป็นตัวอย่าง
    },
    {
        "sentence": "คุณสุนทรทำงานที่ Microsoft ตั้งแต่ปี 2020",
        "entities": [("คุณสุนทร", "PERSON", 0, 8), ("Microsoft", "ORGANIZATION", 18, 27), ("ปี 2020", "DATE", 37, 44)]
    }
    # เพิ่มเติมตัวอย่างตามต้องการ
]

# สร้างรายการข้อมูลพร้อม ID และแปลง entities เป็น JSON string
rows = []
for item in ner_data:
    # แปลง list ของ tuples เป็น JSON string
    entities_json = json.dumps(item["entities"], ensure_ascii=False)
    rows.append([str(uuid.uuid4()), item["sentence"], entities_json])

# บันทึกเป็นไฟล์ CSV ในโฟลเดอร์ DataOutput
output_file = os.path.join(output_dir, 'thai_dataset_ner.csv')
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'sentence', 'entities_json'])
    writer.writerows(rows)

print(f"Created {output_file}")
