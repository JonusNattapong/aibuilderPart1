import csv
import uuid
import os

# ตรวจสอบและสร้างไดเรกทอรี DataOutput หากยังไม่มี
output_dir = 'DataOutput'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ตัวอย่างข้อมูล Style Transfer (Formal, Informal)
style_transfer_data = [
    {"formal_text": "กรุณารอสักครู่", "informal_text": "รอแป๊บนึงนะ"},
    {"formal_text": "ขออภัยในความไม่สะดวก", "informal_text": "ขอโทษทีนะที่ทำให้ไม่สะดวก"},
    {"formal_text": "ท่านต้องการความช่วยเหลือเพิ่มเติมหรือไม่", "informal_text": "อยากให้ช่วยอะไรอีกไหม"},
    {"formal_text": "โปรดติดต่อกลับในภายหลัง", "informal_text": "เดี๋ยวค่อยโทรมาใหม่นะ"},
    {"formal_text": "ข้าพเจ้าไม่เห็นด้วยกับข้อเสนอดังกล่าว", "informal_text": "ฉันว่าข้อเสนอนี้มันไม่โอเคอะ"},
    {"formal_text": "การประชุมจะเริ่มในอีก 10 นาที", "informal_text": "อีก 10 นาทีประชุมนะ"},
    {"formal_text": "กรุณาแสดงบัตรประจำตัวประชาชน", "informal_text": "ขอดูบัตรประชาชนหน่อย"},
    {"formal_text": "ขอบคุณสำหรับข้อมูล", "informal_text": "ขอบใจนะสำหรับข้อมูล"},
    {"formal_text": "รับประทานอาหารให้อร่อยนะครับ", "informal_text": "กินข้าวให้อร่อยนะ"},
    {"formal_text": "ขณะนี้ระบบไม่สามารถใช้งานได้ชั่วคราว", "informal_text": "ตอนนี้ระบบล่ม ใช้ไม่ได้แป๊บนึง"},
    {"formal_text": "โปรดระบุชื่อและนามสกุลของท่าน", "informal_text": "บอกชื่อกับนามสกุลมาหน่อย"},
    {"formal_text": "ข้าพเจ้ามีความประสงค์จะลาป่วยในวันนี้", "informal_text": "วันนี้ฉันขอลาป่วยนะ"},
    {"formal_text": "กรุณากรอกแบบฟอร์มนี้ให้ครบถ้วน", "informal_text": "กรอกฟอร์มนี้ให้ครบด้วย"},
    {"formal_text": "ท่านสามารถติดต่อได้ที่หมายเลขโทรศัพท์...", "informal_text": "โทรไปเบอร์นี้ได้เลย..."},
    {"formal_text": "ขอเรียนเชิญเข้าร่วมงานเลี้ยง", "informal_text": "มางานเลี้ยงกันนะ"},
    {"formal_text": "โปรดรักษาความสะอาด", "informal_text": "ช่วยกันรักษาความสะอาดหน่อย"},
    {"formal_text": "ข้าพเจ้าขอแสดงความยินดีด้วย", "informal_text": "ยินดีด้วยนะ"},
    {"formal_text": "กรุณาเข้าแถวตามลำดับ", "informal_text": "ต่อแถวด้วยจ้า"},
    {"formal_text": "ท่านมีความคิดเห็นอย่างไรบ้าง", "informal_text": "คิดว่าไงบ้างอะ"},
    {"formal_text": "ขอให้เดินทางโดยสวัสดิภาพ", "informal_text": "เดินทางดีๆ นะ"},
    # เพิ่มเติมตัวอย่างตามต้องการ
]

# สร้างรายการข้อมูลพร้อม ID
rows = []
for item in style_transfer_data:
    rows.append([str(uuid.uuid4()), item["formal_text"], item["informal_text"]])

# บันทึกเป็นไฟล์ CSV ในโฟลเดอร์ DataOutput
output_file = os.path.join(output_dir, 'thai_dataset_style_transfer.csv')
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'formal_text', 'informal_text'])
    writer.writerows(rows)

print(f"Created {output_file}")
