import csv
import uuid
import os

# ตรวจสอบและสร้างไดเรกทอรี DataOutput หากยังไม่มี
output_dir = 'DataOutput'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ตัวอย่างข้อมูล Paraphrase Identification (Sentence1, Sentence2, Label)
# Label: 1 = Paraphrase, 0 = Not Paraphrase
paraphrase_data = [
    # Travel
    {"sentence1": "ฉันวางแผนจะไปเที่ยวเชียงใหม่ช่วงฤดูหนาวนี้", "sentence2": "ทริปเชียงใหม่หน้าหนาวนี้อยู่ในแผนการเดินทางของฉัน", "label": 1},
    {"sentence1": "โรงแรมนี้อยู่ใกล้ชายหาดมาก เดินไปแค่ 5 นาที", "sentence2": "ที่พักแห่งนี้ตั้งอยู่ติดริมทะเลเลย", "label": 1},
    {"sentence1": "ตั๋วเครื่องบินไปญี่ปุ่นราคาค่อนข้างสูง", "sentence2": "ค่าตั๋วเครื่องบินไปเกาหลีไม่แพงมาก", "label": 0},
    {"sentence1": "การเดินทางด้วยรถไฟความเร็วสูงสะดวกและรวดเร็ว", "sentence2": "รถไฟความเร็วสูงเป็นตัวเลือกการเดินทางที่สบายและใช้เวลาน้อย", "label": 1},
    {"sentence1": "ฉันชอบไปเที่ยวภูเขามากกว่าทะเล", "sentence2": "ฉันไม่ชอบไปเที่ยวทะเล", "label": 0},
    # Food
    {"sentence1": "ร้านอาหารนี้มีเมนูเด็ดคือต้มยำกุ้ง", "sentence2": "ต้มยำกุ้งเป็นอาหารแนะนำของร้านนี้", "label": 1},
    {"sentence1": "ส้มตำร้านนี้รสชาติจัดจ้านมาก", "sentence2": "ร้านนี้ทำส้มตำได้อร่อย เผ็ดกำลังดี", "label": 1},
    {"sentence1": "ฉันไม่ชอบกินผักชี", "sentence2": "ฉันชอบกินอาหารรสเผ็ด", "label": 0},
    {"sentence1": "กาแฟแก้วนี้เข้มข้นและหอมมาก", "sentence2": "รสชาติกาแฟแก้วนี้เข้มและมีกลิ่นหอม", "label": 1},
    {"sentence1": "อาหารไทยส่วนใหญ่มีรสชาติเผ็ดร้อน", "sentence2": "อาหารญี่ปุ่นมักจะมีรสชาติจืด", "label": 0},
    # Education
    {"sentence1": "นักเรียนควรทบทวนบทเรียนอย่างสม่ำเสมอ", "sentence2": "การทบทวนบทเรียนเป็นประจำสำคัญสำหรับนักเรียน", "label": 1},
    {"sentence1": "มหาวิทยาลัยแห่งนี้มีชื่อเสียงด้านวิศวกรรมศาสตร์", "sentence2": "คณะวิศวกรรมศาสตร์ของมหาวิทยาลัยนี้มีชื่อเสียงโด่งดัง", "label": 1},
    {"sentence1": "การเรียนภาษาอังกฤษต้องฝึกฝนทุกวัน", "sentence2": "การเรียนคณิตศาสตร์ต้องทำโจทย์บ่อยๆ", "label": 0},
    {"sentence1": "เขาได้รับทุนการศึกษาไปเรียนต่อต่างประเทศ", "sentence2": "ทุนการศึกษาทำให้เขาได้ไปศึกษาต่อในต่างแดน", "label": 1},
    {"sentence1": "ห้องสมุดเป็นแหล่งค้นคว้าหาความรู้ที่ดี", "sentence2": "โรงอาหารเป็นที่สำหรับรับประทานอาหาร", "label": 0},
    # เพิ่มเติมตัวอย่างตามต้องการ
]

# สร้างรายการข้อมูลพร้อม ID
rows = []
for item in paraphrase_data:
    rows.append([str(uuid.uuid4()), item["sentence1"], item["sentence2"], item["label"]])

# บันทึกเป็นไฟล์ CSV ในโฟลเดอร์ DataOutput
output_file = os.path.join(output_dir, 'thai_dataset_paraphrase_identification.csv')
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'sentence1', 'sentence2', 'label'])
    writer.writerows(rows)

print(f"Created {output_file}")
