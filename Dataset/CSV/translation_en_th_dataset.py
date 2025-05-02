import csv
import uuid
import os

# ตรวจสอบและสร้างไดเรกทอรี DataOutput หากยังไม่มี
output_dir = 'DataOutput'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ตัวอย่างข้อมูล Translation (English to Thai)
translation_data = [
    {"english_text": "Good afternoon", "thai_text": "สวัสดีตอนบ่าย"},
    {"english_text": "You're welcome", "thai_text": "ไม่เป็นไร / ด้วยความยินดี"},
    {"english_text": "Excuse me, where can I find the train station?", "thai_text": "ขอโทษครับ/ค่ะ สถานีรถไฟอยู่ที่ไหน"},
    {"english_text": "It's raining heavily outside", "thai_text": "ข้างนอกฝนตกหนักมาก"},
    {"english_text": "How old are you?", "thai_text": "คุณอายุเท่าไหร่"},
    {"english_text": "I'm thirsty", "thai_text": "ฉันหิวน้ำ"},
    {"english_text": "Can I have the menu, please?", "thai_text": "ขอเมนูหน่อยได้ไหมครับ/คะ"},
    {"english_text": "Do you accept credit cards?", "thai_text": "คุณรับบัตรเครดิตไหม"},
    {"english_text": "I need help", "thai_text": "ฉันต้องการความช่วยเหลือ"},
    {"english_text": "Can you repeat that, please?", "thai_text": "ช่วยพูดอีกครั้งได้ไหมครับ/คะ"},
    {"english_text": "Have a nice day", "thai_text": "ขอให้เป็นวันที่ดี"},
    {"english_text": "Take care", "thai_text": "ดูแลตัวเองด้วยนะ"},
    {"english_text": "What time is it?", "thai_text": "ตอนนี้กี่โมงแล้ว"},
    {"english_text": "This is my friend, Somsak", "thai_text": "นี่คือเพื่อนของฉัน สมศักดิ์"},
    {"english_text": "She works as a doctor", "thai_text": "เธอทำงานเป็นหมอ"},
    {"english_text": "They are playing football in the park", "thai_text": "พวกเขากำลังเล่นฟุตบอลอยู่ในสวนสาธารณะ"},
    {"english_text": "The dog is barking loudly", "thai_text": "สุนัขกำลังเห่าเสียงดัง"},
    {"english_text": "Chiang Mai is a popular tourist destination in Thailand", "thai_text": "เชียงใหม่เป็นสถานที่ท่องเที่ยวยอดนิยมในประเทศไทย"},
    {"english_text": "I enjoy listening to music in my free time", "thai_text": "ฉันชอบฟังเพลงในเวลาว่าง"},
    {"english_text": "Traveling broadens the mind", "thai_text": "การเดินทางช่วยเปิดโลกทัศน์"},
    # เพิ่มเติมตัวอย่างตามต้องการ
]

# สร้างรายการข้อมูลพร้อม ID
rows = []
for item in translation_data:
    rows.append([str(uuid.uuid4()), item["english_text"], item["thai_text"]])

# บันทึกเป็นไฟล์ CSV ในโฟลเดอร์ DataOutput
output_file = os.path.join(output_dir, 'thai_dataset_translation_en_th.csv')
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'english_text', 'thai_text'])
    writer.writerows(rows)

print(f"Created {output_file}")
