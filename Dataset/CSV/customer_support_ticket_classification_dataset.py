import csv
import uuid
import os

# ตรวจสอบและสร้างไดเรกทอรี DataOutput หากยังไม่มี
output_dir = 'DataOutput'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ตัวอย่างข้อมูล Customer Support Ticket Classification (Ticket Text, Label)
ticket_data = {
    "shipping_issue": [
        "ยังไม่ได้รับสินค้าเลยค่ะ สั่งไปตั้งแต่อาทิตย์ที่แล้ว",
        "สถานะการจัดส่งไม่อัปเดตเลย ของถึงไหนแล้วครับ",
        "ได้รับสินค้าไม่ครบตามจำนวนที่สั่ง",
        "สินค้าที่ได้รับเสียหายระหว่างการขนส่ง",
        "ต้องการเปลี่ยนที่อยู่ในการจัดส่ง",
        "ค่าจัดส่งแพงเกินไป มีตัวเลือกอื่นไหม",
        "พนักงานส่งของโทรมาแต่ไม่รอ รับสายไม่ทัน",
        "เลขพัสดุที่ให้มาตรวจสอบไม่ได้",
        "สินค้าถูกตีกลับไปยังผู้ส่ง ต้องทำอย่างไร",
        "ระยะเวลาในการจัดส่งนานกว่าที่แจ้งไว้มาก",
    ],
    "payment_problem": [
        "จ่ายเงินไปแล้ว แต่ระบบแจ้งว่ายังไม่ชำระเงิน",
        "บัตรเครดิตถูกตัดเงินซ้ำซ้อน",
        "ไม่สามารถใช้โค้ดส่วนลดได้ ระบบแจ้งว่าโค้ดไม่ถูกต้อง",
        "ต้องการขอใบกำกับภาษี",
        "โอนเงินผิดบัญชี ต้องทำอย่างไร",
        "ระบบไม่รองรับการชำระเงินผ่านช่องทางนี้",
        "ยอดเงินที่ต้องชำระไม่ถูกต้อง",
        "ต้องการยกเลิกคำสั่งซื้อและขอคืนเงิน",
        "ไม่ได้รับอีเมลยืนยันการชำระเงิน",
        "มีปัญหาในการผ่อนชำระสินค้า",
    ],
    "service_request": [
        "ต้องการสอบถามข้อมูลเพิ่มเติมเกี่ยวกับสินค้า",
        "สนใจสมัครเป็นตัวแทนจำหน่าย",
        "ต้องการเปลี่ยนรหัสผ่านเข้าระบบ",
        "ขอคำแนะนำในการใช้งานผลิตภัณฑ์",
        "ต้องการแจ้งเปลี่ยนแปลงข้อมูลส่วนตัว",
        "สอบถามเกี่ยวกับโปรโมชั่นปัจจุบัน",
        "ต้องการยกเลิกการรับข่าวสารทางอีเมล",
        "สนใจเข้าร่วมโปรแกรมสะสมแต้ม",
        "ขอใบเสนอราคาสำหรับองค์กร",
        "ต้องการติดต่อฝ่ายเทคนิคเพื่อแจ้งปัญหาการใช้งาน",
    ],
    "product_inquiry": [
        "สินค้าชิ้นนี้มีสีอะไรบ้าง",
        "ขนาดของสินค้าเท่าไหร่ครับ",
        "วัสดุที่ใช้ผลิตคืออะไร",
        "สินค้ารับประกันกี่ปี",
        "มีคู่มือการใช้งานภาษาไทยไหม",
        "สินค้าชิ้นนี้กันน้ำได้หรือไม่",
        "แบตเตอรี่ใช้งานได้นานแค่ไหน",
        "สินค้านี้เหมาะสำหรับเด็กอายุเท่าไหร่",
        "มีอะไหล่สำรองขายหรือไม่",
        "สินค้าผลิตที่ประเทศอะไร",
    ]
    # เพิ่มเติมตัวอย่างตามต้องการ
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in ticket_data.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV ในโฟลเดอร์ DataOutput
output_file = os.path.join(output_dir, 'thai_dataset_customer_support_ticket_classification.csv')
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'ticket_text', 'label'])
    writer.writerows(rows)

print(f"Created {output_file}")
