import csv
import uuid
import os

# ตรวจสอบและสร้างไดเรกทอรี DataOutput หากยังไม่มี
output_dir = 'DataOutput'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ตัวอย่างข้อมูล Conversation Simulation (Dialogue Turns)
# อาจจะเก็บเป็นคู่ User-Agent หรือเก็บเป็นลำดับการสนทนา
# แบบง่าย: เก็บเป็นข้อความสนทนาต่อเนื่อง
conversation_data = [
    # Example 1: Ordering Coffee
    "User: สวัสดีครับ ขอสั่งกาแฟหน่อยครับ\nAgent: สวัสดีค่ะ รับเป็นกาแฟอะไรดีคะ\nUser: เอาลาเต้ร้อนแก้วนึงครับ ไม่หวานเลย\nAgent: ได้ค่ะ ลาเต้ร้อน ไม่หวาน รอสักครู่นะคะ",
    # Example 2: Asking for Directions
    "User: ขอโทษนะครับ ไม่ทราบว่าไปสถานีรถไฟฟ้าที่ใกล้ที่สุดทางไหนครับ\nAgent: อ๋อ เดินตรงไปทางนี้ประมาณ 200 เมตร แล้วเลี้ยวขวาตรงสี่แยกค่ะ สถานีจะอยู่ทางซ้ายมือ\nUser: ขอบคุณมากครับ\nAgent: ยินดีค่ะ",
    # Example 3: Booking Appointment
    "User: ต้องการนัดหมายเพื่อปรึกษาเรื่องการลงทุนครับ\nAgent: ได้ค่ะ ไม่ทราบว่าสะดวกช่วงวันไหน เวลาใดคะ\nUser: ขอเป็นวันพุธหน้าช่วงบ่ายครับ\nAgent: วันพุธช่วงบ่าย มีคิวว่างเวลา 14.00 น. สะดวกไหมคะ\nUser: สะดวกครับ\nAgent: เรียบร้อยค่ะ ดิฉันทำการนัดหมายให้แล้วนะคะ",
    # Example 4: Simple Chat
    "User: วันนี้อากาศดีจังเลยนะ\nAgent: ใช่ค่ะ ท้องฟ้าแจ่มใสดี เหมาะกับการออกไปข้างนอก\nUser: ว่าจะไปเดินเล่นที่สวนสาธารณะสักหน่อย\nAgent: เป็นความคิดที่ดีเลยค่ะ ขอให้มีความสุขกับการเดินเล่นนะคะ",
    # Example 5: Quiz Game Turn
    "Agent: คำถามข้อต่อไป เมืองหลวงของประเทศฝรั่งเศสคือเมืองใด?\nUser: ปารีส\nAgent: ถูกต้องค่ะ! คุณได้รับ 10 คะแนน",
    # เพิ่มเติมตัวอย่างตามต้องการ
]

# สร้างรายการข้อมูลพร้อม ID
rows = []
for dialogue in conversation_data:
    rows.append([str(uuid.uuid4()), dialogue])

# บันทึกเป็นไฟล์ CSV ในโฟลเดอร์ DataOutput
output_file = os.path.join(output_dir, 'thai_dataset_conversation_simulation.csv')
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'dialogue'])
    writer.writerows(rows)

print(f"Created {output_file}")
