import csv
import uuid
import os

# ตรวจสอบและสร้างไดเรกทอรี DataOutput หากยังไม่มี
output_dir = 'DataOutput'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ตัวอย่างข้อมูล Content Moderation (Text, Label)
# Label: 0 = Appropriate, 1 = Inappropriate (General)
moderation_data = [
    # Appropriate Examples
    {"text": "แบ่งปันสูตรทำเค้กช็อกโกแลตง่ายๆ ค่ะ", "label": 0},
    {"text": "ใครมีหนังสือดีๆ แนะนำบ้างครับช่วงนี้", "label": 0},
    {"text": "วันนี้อากาศดีจัง ไปเดินเล่นสวนสาธารณะดีกว่า", "label": 0},
    {"text": "ขอคำแนะนำเรื่องการเลือกซื้อโน้ตบุ๊กหน่อยครับ", "label": 0},
    {"text": "ภาพถ่ายดอกไม้สวยๆ จากสวนหลังบ้าน", "label": 0},
    {"text": "ขอบคุณสำหรับข้อมูลดีๆ นะคะ", "label": 0},
    {"text": "ร่วมแสดงความยินดีกับบัณฑิตใหม่ทุกคนครับ", "label": 0},
    {"text": "มีใครเคยไปเที่ยวญี่ปุ่นช่วงฤดูใบไม้ร่วงบ้าง", "label": 0},
    {"text": "อัปเดตข่าวสารเทคโนโลยีล่าสุด", "label": 0},
    {"text": "ขอพลังใจให้ทุกคนที่กำลังท้อแท้", "label": 0},

    # Inappropriate Examples (General - Avoid overly explicit content)
    {"text": "ไอ้พวก... มันน่าโดนกระทืบจริงๆ", "label": 1}, # Hate speech / Violence
    {"text": "ขายยาลดความอ้วน เห็นผลใน 3 วัน ไม่โยโย่ สนใจทักแชท", "label": 1}, # Spam / Misleading
    {"text": "ใครมีคลิปหลุดดาราคนนั้นบ้าง ขอหน่อย", "label": 1}, # Requesting inappropriate content
    {"text": "เว็บพนันออนไลน์ที่ดีที่สุด สมัครเลย!", "label": 1}, # Promoting gambling
    {"text": "ด่ากันไปมาด้วยคำหยาบคาย", "label": 1}, # Profanity / Harassment
    {"text": "ปล่อยเงินกู้นอกระบบ ดอกเบี้ยถูก ติดต่อไลน์ไอดี...", "label": 1}, # Illegal activity
    {"text": "เหยียดเชื้อชาติและศาสนาอย่างรุนแรง", "label": 1}, # Hate speech
    {"text": "โพสต์ภาพอุบัติเหตุที่น่าสยดสยองโดยไม่มีการเซ็นเซอร์", "label": 1}, # Graphic violence
    {"text": "แอบอ้างเป็นบุคคลอื่นเพื่อหลอกลวงข้อมูลส่วนตัว", "label": 1}, # Impersonation / Phishing
    {"text": "เผยแพร่ข้อมูลส่วนตัวของผู้อื่นโดยไม่ได้รับอนุญาต", "label": 1}, # Doxing / Privacy violation
    # เพิ่มเติมตัวอย่างตามต้องการ (ระมัดระวังเรื่องเนื้อหา)
]

# สร้างรายการข้อมูลพร้อม ID
rows = []
for item in moderation_data:
    rows.append([str(uuid.uuid4()), item["text"], item["label"]])

# บันทึกเป็นไฟล์ CSV ในโฟลเดอร์ DataOutput
output_file = os.path.join(output_dir, 'thai_dataset_content_moderation.csv')
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print(f"Created {output_file}")
