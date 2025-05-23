import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ automotive
categories = {
    "automotive": [
        "รีวิวรถยนต์ไฟฟ้ารุ่นใหม่ล่าสุด",
        "เทคโนโลยีรถยนต์ไร้คนขับ (Autonomous Driving)",
        "การดูแลรักษารถยนต์เบื้องต้น",
        "เปรียบเทียบรถยนต์ SUV ยอดนิยม",
        "งานมอเตอร์โชว์ปีนี้มีอะไรน่าสนใจ",
        "เทรนด์รถยนต์ในอนาคต",
        "การเลือกซื้อประกันภัยรถยนต์",
        "ระบบความปลอดภัยในรถยนต์สมัยใหม่",
        "การแต่งรถยนต์แบบต่างๆ",
        "รถยนต์ไฮบริดทำงานอย่างไร",
        "การเลือกน้ำมันเครื่องให้เหมาะสมกับรถ",
        "ตลาดรถยนต์มือสองน่าสนใจไหม",
        "การทดสอบการชนและความปลอดภัยของรถยนต์",
        "เทคโนโลยีแบตเตอรี่สำหรับรถยนต์ไฟฟ้า",
        "การดูแลยางรถยนต์และการเติมลม",
        "รถยนต์พลังงานไฮโดรเจน",
        "การออกแบบรถยนต์และหลักอากาศพลศาสตร์",
        "ระบบ Infotainment ในรถยนต์",
        "การขอสินเชื่อรถยนต์",
        "รถกระบะรุ่นไหนน่าใช้ที่สุด",
        "การดูแลรักษาสีรถยนต์",
        "เทคโนโลยี Connected Car",
        "การเลือกฟิล์มกรองแสงรถยนต์",
        "รถยนต์ Eco Car ประหยัดน้ำมัน",
        "การดูแลระบบเบรกและช่วงล่าง",
        "รถยนต์สปอร์ตสมรรถนะสูง",
        "การติดตั้งแก๊ส LPG/NGV ในรถยนต์",
        "เทคโนโลยีการชาร์จรถยนต์ไฟฟ้า",
        "การดูแลรักษาแบตเตอรี่รถยนต์",
        "รถยนต์ MPV สำหรับครอบครัว",
        "การเลือกซื้อรถยนต์คันแรก",
        "การดูแลระบบปรับอากาศในรถยนต์",
        "เทคโนโลยีช่วยเหลือการขับขี่ (ADAS)",
        "การต่อภาษีรถยนต์และ พ.ร.บ.",
        "รถยนต์คลาสสิกและการสะสม",
        "การดูแลภายในห้องโดยสารรถยนต์",
        "เทคโนโลยี V2X (Vehicle-to-Everything)",
        "การเลือกซื้ออุปกรณ์เสริมรถยนต์",
        "รถยนต์หรูและ Supercar",
        "การดูแลระบบไฟส่องสว่างรถยนต์",
        "เทคโนโลยีการผลิตรถยนต์",
        "การเลือกศูนย์บริการรถยนต์",
        "รถยนต์ไฟฟ้าดัดแปลง",
        "การดูแลรักษาหม้อน้ำรถยนต์",
        "เทคโนโลยีการจอดรถอัตโนมัติ",
        "การเลือกซื้อกล้องติดหน้ารถ",
        "รถยนต์สำหรับผู้สูงอายุและผู้พิการ",
        "การดูแลรักษาเกียร์อัตโนมัติและเกียร์ธรรมดา",
        "เทคโนโลยีการแสดงผลบนกระจกหน้า (Head-up Display)",
        "การเลือกซื้อจักรยานยนต์"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_automotive.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created DataOutput/thai_dataset_automotive.csv")
