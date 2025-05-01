import csv
import uuid
import os

# ตรวจสอบและสร้างไดเรกทอรี DataOutput หากยังไม่มี
output_dir = 'DataOutput'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ตัวอย่างข้อมูล Translation (Thai to English)
translation_data = [
    {"thai_text": "สวัสดีตอนเช้า", "english_text": "Good morning"},
    {"thai_text": "ขอบคุณมากครับ", "english_text": "Thank you very much"},
    {"thai_text": "ขอโทษค่ะ ฉันมาสาย", "english_text": "Sorry, I'm late"},
    {"thai_text": "วันนี้อากาศดีจังเลยนะ", "english_text": "The weather is nice today, isn't it?"},
    {"thai_text": "คุณชื่ออะไร", "english_text": "What is your name?"},
    {"thai_text": "ฉันหิวข้าวแล้ว", "english_text": "I'm hungry"},
    {"thai_text": "ห้องน้ำอยู่ที่ไหน", "english_text": "Where is the restroom?"},
    {"thai_text": "ราคาเท่าไหร่ครับ", "english_text": "How much does it cost?"},
    {"thai_text": "ฉันไม่เข้าใจ", "english_text": "I don't understand"},
    {"thai_text": "ช่วยพูดช้าๆ หน่อยได้ไหมคะ", "english_text": "Could you speak more slowly, please?"},
    {"thai_text": "ยินดีที่ได้รู้จัก", "english_text": "Nice to meet you"},
    {"thai_text": "ลาก่อน แล้วพบกันใหม่", "english_text": "Goodbye, see you later"},
    {"thai_text": "ฉันรักคุณ", "english_text": "I love you"},
    {"thai_text": "นี่คือหนังสือของฉัน", "english_text": "This is my book"},
    {"thai_text": "เขาเป็นนักเรียน", "english_text": "He is a student"},
    {"thai_text": "พวกเรากำลังจะไปตลาด", "english_text": "We are going to the market"},
    {"thai_text": "แมวกำลังนอนหลับอยู่บนโซฟา", "english_text": "The cat is sleeping on the sofa"},
    {"thai_text": "กรุงเทพเป็นเมืองหลวงของประเทศไทย", "english_text": "Bangkok is the capital of Thailand"},
    {"thai_text": "ฉันชอบกินอาหารไทย โดยเฉพาะต้มยำกุ้ง", "english_text": "I like eating Thai food, especially Tom Yum Goong"},
    {"thai_text": "การเรียนภาษาใหม่ต้องใช้เวลาและความอดทน", "english_text": "Learning a new language takes time and patience"},
    # เพิ่มเติมตัวอย่างตามต้องการ
]

# สร้างรายการข้อมูลพร้อม ID
rows = []
for item in translation_data:
    rows.append([str(uuid.uuid4()), item["thai_text"], item["english_text"]])

# บันทึกเป็นไฟล์ CSV ในโฟลเดอร์ DataOutput
output_file = os.path.join(output_dir, 'thai_dataset_translation_th_en.csv')
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'thai_text', 'english_text'])
    writer.writerows(rows)

print(f"Created {output_file}")
