import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ science
categories = {
    "science": [
        "การค้นพบคลื่นความโน้มถ่วง",
        "ทฤษฎีสัมพัทธภาพของไอน์สไตน์",
        "กลศาสตร์ควอนตัมเบื้องต้น",
        "การทำงานของหลุมดำ",
        "การกำเนิดของเอกภพและทฤษฎีบิกแบง",
        "การสำรวจดาวอังคารและค้นหาสิ่งมีชีวิต",
        "โครงสร้างของ DNA และพันธุวิศวกรรม",
        "วิวัฒนาการของสิ่งมีชีวิตตามแนวคิดของดาร์วิน",
        "การทำงานของเซลล์ประสาทและสมอง",
        "ปรากฏการณ์โลกร้อนและวิทยาศาสตร์ภูมิอากาศ",
        "การค้นพบอนุภาคฮิกส์โบซอน",
        "เทคโนโลยีนาโนและการประยุกต์ใช้",
        "การทำงานของระบบภูมิคุ้มกัน",
        "วิทยาศาสตร์ทางทะเลและการเปลี่ยนแปลงของมหาสมุทร",
        "การค้นพบดาวเคราะห์นอกระบบสุริยะ",
        "พลังงานนิวเคลียร์ฟิวชัน",
        "ความก้าวหน้าทางด้านวัสดุศาสตร์",
        "การศึกษาเรื่องจุลินทรีย์และบทบาทในระบบนิเวศ",
        "วิทยาศาสตร์การอาหารและโภชนาการ",
        "การทำงานของยีนและการแสดงออก",
        "การศึกษาเรื่องแผ่นดินไหวและภูเขาไฟระเบิด",
        "ความลับของสสารมืดและพลังงานมืด",
        "เทคโนโลยีชีวภาพและการประยุกต์ใช้ทางการแพทย์",
        "การทำงานของเลเซอร์",
        "วิทยาศาสตร์เกี่ยวกับพฤติกรรมสัตว์",
        "การศึกษาเรื่องการนอนหลับ",
        "ความก้าวหน้าในการรักษาโรคมะเร็ง",
        "การทำงานของกล้องโทรทรรศน์อวกาศ",
        "วิทยาศาสตร์ข้อมูล (Data Science)",
        "การศึกษาเรื่องไวรัสและโรคระบาด",
        "การสังเคราะห์แสงของพืช",
        "การทำงานของระบบต่อมไร้ท่อและฮอร์โมน",
        "วิทยาศาสตร์โลกและธรณีวิทยา",
        "การพัฒนาปัญญาประดิษฐ์ (AI)",
        "การศึกษาเรื่องความจำและการเรียนรู้",
        "การทำงานของระบบไหลเวียนโลหิต",
        "วิทยาศาสตร์อวกาศและการเดินทางในอวกาศ",
        "การพัฒนาวัคซีน",
        "การศึกษาเรื่องเซลล์ต้นกำเนิด (Stem Cell)",
        "การทำงานของระบบย่อยอาหาร",
        "วิทยาศาสตร์สิ่งแวดล้อมและการอนุรักษ์",
        "การพัฒนาหุ่นยนต์",
        "การศึกษาเรื่องแสงและทัศนศาสตร์",
        "การทำงานของระบบหายใจ",
        "วิทยาศาสตร์นิติเวช",
        "การพัฒนาแบตเตอรี่ประสิทธิภาพสูง",
        "การศึกษาเรื่องพันธุกรรมมนุษย์",
        "การทำงานของระบบขับถ่าย",
        "วิทยาศาสตร์คอมพิวเตอร์และอัลกอริทึม",
        "การศึกษาเรื่องความเจ็บปวด"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_science.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created thai_dataset_science.csv")
