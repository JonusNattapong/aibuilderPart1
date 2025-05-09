import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ legal_rights
categories = {
    "legal_rights": [
        "โดนโกงจากการซื้อของออนไลน์ ต้องแจ้งความที่ไหน อย่างไร?",
        "สิทธิการลาคลอดตามกฎหมายแรงงาน ได้กี่วัน ได้รับค่าจ้างหรือไม่?",
        "ผู้บริโภคมีสิทธิอะไรบ้างเมื่อซื้อสินค้าหรือบริการ",
        "ขั้นตอนการร้องเรียน สคบ. เมื่อไม่ได้รับความเป็นธรรม",
        "กฎหมาย PDPA คุ้มครองข้อมูลส่วนบุคคลของเราอย่างไร",
        "สิทธิของผู้ต้องหาในคดีอาญา มีอะไรบ้าง",
        "การกู้ยืมเงินนอกระบบ ผิดกฎหมายหรือไม่ ต้องทำอย่างไร",
        "กฎหมายเกี่ยวกับมรดก การทำพินัยกรรม และการแบ่งทรัพย์สิน",
        "สิทธิการลาป่วย ลากิจ ตามกฎหมายแรงงาน",
        "การถูกเลิกจ้างอย่างไม่เป็นธรรม สามารถฟ้องร้องได้หรือไม่",
        "กฎหมายลิขสิทธิ์ การนำผลงานผู้อื่นมาใช้โดยไม่ได้รับอนุญาต",
        "สิทธิในการเข้าถึงข้อมูลข่าวสารของราชการ",
        "การถูกละเมิดสิทธิส่วนบุคคลบนโลกออนไลน์ ทำอะไรได้บ้าง",
        "กฎหมายเกี่ยวกับการเช่าซื้อบ้านและคอนโด",
        "สิทธิของผู้พิการตามกฎหมาย",
        "การฟ้องร้องคดีผู้บริโภค ขั้นตอนและค่าใช้จ่าย",
        "กฎหมายเกี่ยวกับการหย่าร้าง การแบ่งสินสมรส และสิทธิเลี้ยงดูบุตร",
        "สิทธิในการได้รับค่าชดเชยเมื่อถูกเลิกจ้าง",
        "การถูกหมิ่นประมาททางออนไลน์ สามารถดำเนินคดีได้อย่างไร",
        "กฎหมายเกี่ยวกับสัญญาจ้างงานและข้อตกลงต่างๆ",
        "สิทธิในการรวมตัวและเจรจาต่อรองของแรงงาน",
        "การถูกคุกคามทางเพศในที่ทำงาน กฎหมายคุ้มครองอย่างไร",
        "กฎหมายเกี่ยวกับหนี้บัตรเครดิตและการทวงถามหนี้",
        "สิทธิในการได้รับบริการสาธารณสุขที่มีคุณภาพ",
        "การถูกหลอกลวงให้ลงทุน (แชร์ลูกโซ่) แจ้งความได้ที่ไหน",
        "กฎหมายเกี่ยวกับการคุ้มครองเด็กและเยาวชน",
        "สิทธิในการเปลี่ยนหรือคืนสินค้าตามกฎหมาย",
        "การถูกเลือกปฏิบัติอย่างไม่เป็นธรรมเนื่องจากเพศ อายุ ศาสนา",
        "กฎหมายเกี่ยวกับ พ.ร.บ. คอมพิวเตอร์ฯ",
        "สิทธิในการได้รับความช่วยเหลือทางกฎหมายจากทนายความ",
        "การถูกละเมิดเครื่องหมายการค้า",
        "กฎหมายเกี่ยวกับการจราจรทางบก ข้อหาและค่าปรับ",
        "สิทธิในการประกันตัวเมื่อถูกจับกุม",
        "การถูกละเมิดสัญญาเช่าที่พักอาศัย",
        "กฎหมายเกี่ยวกับการคุ้มครองผู้สูงอายุ",
        "สิทธิในการร้องเรียนหน่วยงานรัฐที่ไม่โปร่งใส",
        "การถูกหลอกลวงจากแก๊งคอลเซ็นเตอร์",
        "กฎหมายเกี่ยวกับการรับบุตรบุญธรรม",
        "สิทธิในการได้รับค่าจ้างขั้นต่ำและค่าล่วงเวลา",
        "การถูกละเมิดข้อมูลส่วนบุคคลตาม PDPA",
        "กฎหมายเกี่ยวกับการกู้ยืมเงิน กยศ.",
        "สิทธิในการเข้าถึงกระบวนการยุติธรรมอย่างเท่าเทียม",
        "การถูกละเมิดลิขสิทธิ์ซอฟต์แวร์",
        "กฎหมายเกี่ยวกับการประกันสังคม",
        "สิทธิในการได้รับข้อมูลเกี่ยวกับสินค้าและบริการอย่างถูกต้อง",
        "การถูกใส่ร้ายป้ายสี ทำให้เสียชื่อเสียง",
        "กฎหมายเกี่ยวกับการคุ้มครองสัตว์",
        "สิทธิในการแสดงความคิดเห็นและเสรีภาพสื่อ",
        "การถูกหลอกลวงในการทำธุรกรรมทางการเงิน",
        "กฎหมายเกี่ยวกับภาษีเงินได้บุคคลธรรมดา"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_legal_rights.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created DataOutput/DataOutput/thai_dataset_legal_rights.csv")
