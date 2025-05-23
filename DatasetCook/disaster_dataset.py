import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ disaster
categories = {
    "disaster": [
        "วิธีรับมือเมื่อเกิดน้ำท่วมบ้าน ควรทำอย่างไร",
        "การเตรียมตัวหนีไฟป่าและป้องกันหมอกควัน",
        "ข้อควรปฏิบัติเมื่อเกิดแผ่นดินไหว",
        "การเตรียมพร้อมรับมือพายุฤดูร้อนและลมกระโชกแรง",
        "วิธีเอาตัวรอดจากสึนามิ",
        "การป้องกันและระงับอัคคีภัยในอาคารบ้านเรือน",
        "ภัยแล้ง ผลกระทบและการเตรียมรับมือ",
        "ข้อควรปฏิบัติเมื่อเกิดดินโคลนถล่ม",
        "การเตรียมถุงยังชีพฉุกเฉิน ควรมีอะไรบ้าง",
        "วิธีตรวจสอบความเสี่ยงภัยพิบัติในพื้นที่ของคุณ",
        "การอพยพเมื่อเกิดภัยพิบัติ ควรไปที่ไหน อย่างไร",
        "การปฐมพยาบาลเบื้องต้นสำหรับผู้ประสบภัย",
        "วิธีติดต่อขอความช่วยเหลือในสถานการณ์ฉุกเฉิน",
        "การป้องกันโรคระบาดที่มาพร้อมกับน้ำท่วม",
        "ข้อควรระวังเกี่ยวกับไฟฟ้าดูด ไฟฟ้ารั่ว ช่วงน้ำท่วม",
        "การดูแลสุขภาพจิตของผู้ประสบภัยพิบัติ",
        "วิธีสร้างบ้านให้ทนทานต่อแผ่นดินไหว",
        "การติดตามข่าวสารและประกาศเตือนภัยจากหน่วยงานราชการ",
        "บทบาทของอาสาสมัครในการช่วยเหลือผู้ประสบภัย",
        "การป้องกันฟ้าผ่าในช่วงพายุฝนฟ้าคะนอง",
        "วิธีรับมือกับคลื่นความร้อน (Heatwave)",
        "การเตรียมพร้อมสำหรับภัยหนาวในพื้นที่สูง",
        "ข้อควรปฏิบัติเมื่อติดอยู่ในอาคารที่เกิดไฟไหม้",
        "การช่วยเหลือสัตว์เลี้ยงในสถานการณ์ภัยพิบัติ",
        "วิธีทำความสะอาดบ้านหลังน้ำลด",
        "การป้องกันและรับมือกับพายุหมุนเขตร้อน (ไต้ฝุ่น, ไซโคลน)",
        "การเตรียมแหล่งน้ำสำรองในภาวะภัยแล้ง",
        "ข้อควรปฏิบัติเมื่อขับรถขณะเกิดพายุฝน",
        "การสร้างหลุมหลบภัยชั่วคราว",
        "วิธีตรวจสอบความเสียหายของโครงสร้างอาคารหลังแผ่นดินไหว",
        "การป้องกันอันตรายจากสัตว์มีพิษที่มากับน้ำท่วม",
        "การวางแผนเส้นทางอพยพของครอบครัว",
        "วิธีสื่อสารกับครอบครัวเมื่อสัญญาณโทรศัพท์ล่ม",
        "การเตรียมเอกสารสำคัญให้พร้อมสำหรับการอพยพ",
        "การป้องกันการเกิดไฟป่า",
        "วิธีรับมือกับสถานการณ์ฝุ่น PM 2.5 รุนแรง",
        "การเตรียมอาหารและน้ำดื่มสำรองสำหรับภาวะฉุกเฉิน",
        "ข้อควรปฏิบัติเมื่ออยู่ในพื้นที่เสี่ยงสึนามิ",
        "การช่วยเหลือผู้สูงอายุและผู้พิการในสถานการณ์ภัยพิบัติ",
        "วิธีป้องกันบ้านจากลมพายุ",
        "การเรียนรู้สัญญาณเตือนภัยทางธรรมชาติ",
        "การฟื้นฟูสภาพจิตใจหลังประสบภัยพิบัติ",
        "วิธีป้องกันการเกิดดินถล่มในพื้นที่ลาดชัน",
        "การเตรียมอุปกรณ์ดับเพลิงเบื้องต้นในบ้าน",
        "ข้อควรปฏิบัติเมื่อต้องอพยพไปศูนย์พักพิงชั่วคราว",
        "การป้องกันโรคติดต่อที่เกิดจากสุขอนามัยไม่ดีหลังภัยพิบัติ",
        "วิธีตรวจสอบข่าวปลอม (Fake News) ในช่วงเกิดภัยพิบัติ",
        "การเตรียมพร้อมรับมือภูเขาไฟระเบิด (ในพื้นที่เสี่ยง)",
        "การให้ความช่วยเหลือเพื่อนบ้านในยามเกิดภัย",
        "บทเรียนจากภัยพิบัติในอดีตและการปรับตัว"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_disaster.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created DataOutput/thai_dataset_disaster.csv")
