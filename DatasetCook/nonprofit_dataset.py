import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ nonprofit
categories = {
    "nonprofit": [
        "โครงการช่วยเหลือเด็กกำพร้าและด้อยโอกาสทางการศึกษา",
        "เปิดรับบริจาคเงินและสิ่งของช่วยเหลือผู้ประสบภัยน้ำท่วม",
        "มูลนิธิช่วยชีวิตสัตว์ป่า เชิญร่วมเป็นอาสาสมัคร",
        "กิจกรรมปลูกป่าชายเลนเพื่อฟื้นฟูระบบนิเวศ",
        "องค์กรภาคประชาสังคมรณรงค์ต่อต้านความรุนแรงในครอบครัว",
        "โครงการอาหารกลางวันสำหรับเด็กนักเรียนในพื้นที่ห่างไกล",
        "ระดมทุนสร้างโรงพยาบาลสนามเพื่อรองรับผู้ป่วยโควิด-19",
        "มูลนิธิกระจกเงา เปิดรับบริจาคเสื้อผ้าและของใช้",
        "กิจกรรมเก็บขยะชายหาด รักษาสิ่งแวดล้อมทางทะเล",
        "องค์กรช่วยเหลือผู้ลี้ภัยและผู้พลัดถิ่น",
        "โครงการพัฒนาอาชีพสำหรับผู้พิการ",
        "เปิดรับบริจาคโลหิต ช่วยต่อชีวิตเพื่อนมนุษย์",
        "มูลนิธิเพื่อสุนัขในซอย ต้องการอาสาสมัครดูแลสุนัขจรจัด",
        "กิจกรรมสอนหนังสือเด็กในชุมชนแออัด",
        "องค์กรส่งเสริมสิทธิมนุษยชนและประชาธิปไตย",
        "โครงการสร้างแหล่งน้ำสะอาดให้ชุมชนในถิ่นทุรกันดาร",
        "ระดมทุนช่วยเหลือผู้ป่วยโรคมะเร็งที่ขาดแคลนทุนทรัพย์",
        "มูลนิธิสืบนาคะเสถียร ปกป้องผืนป่าและสัตว์ป่า",
        "กิจกรรมเยี่ยมเยียนและให้กำลังใจผู้สูงอายุในบ้านพักคนชรา",
        "องค์กรพัฒนาเอกชนทำงานด้านการศึกษาทางเลือก",
        "โครงการช่วยเหลือเกษตรกรที่ได้รับผลกระทบจากภัยแล้ง",
        "เปิดรับบริจาคอุปกรณ์การเรียนให้โรงเรียนชายขอบ",
        "มูลนิธิรักษ์ไทย สนับสนุนการพัฒนาชุมชนอย่างยั่งยืน",
        "กิจกรรมค่ายอาสาพัฒนาชนบท",
        "องค์กรต่อต้านการค้ามนุษย์",
        "โครงการช่วยเหลือผู้ประสบภัยหนาวบนดอย",
        "ระดมทุนจัดซื้ออุปกรณ์ทางการแพทย์ให้โรงพยาบาล",
        "มูลนิธิเด็กโสสะแห่งประเทศไทย สร้างครอบครัวทดแทนถาวร",
        "กิจกรรมรณรงค์ลดใช้พลาสติก",
        "องค์กรช่วยเหลือแรงงานข้ามชาติ",
        "โครงการฝึกอาชีพให้แม่เลี้ยงเดี่ยว",
        "เปิดรับบริจาคหนังสือเข้าห้องสมุดโรงเรียน",
        "มูลนิธิป่อเต็กตึ๊ง ช่วยเหลือผู้ประสบภัยและผู้ยากไร้",
        "กิจกรรมซ่อมแซมบ้านให้ผู้ยากไร้",
        "องค์กรส่งเสริมความเท่าเทียมทางเพศ",
        "โครงการช่วยเหลือผู้ป่วยติดเตียง",
        "ระดมทุนช่วยเหลือค่าอาหารและยารักษาโรคให้สัตว์จรจัด",
        "มูลนิธิยุวพัฒน์ มอบทุนการศึกษาแก่เยาวชน",
        "กิจกรรมรณรงค์ให้ความรู้เรื่องสุขภาพจิต",
        "องค์กรตรวจสอบการทุจริตคอร์รัปชัน",
        "โครงการพัฒนาคุณภาพชีวิตผู้สูงอายุ",
        "เปิดรับบริจาคคอมพิวเตอร์มือสองเพื่อการศึกษา",
        "มูลนิธิสายใจไทย ในพระบรมราชูปถัมภ์",
        "กิจกรรมอาสาพาน้องเที่ยว",
        "องค์กรคุ้มครองผู้บริโภค",
        "โครงการช่วยเหลือผู้ประสบภัยจากเหตุการณ์ไฟไหม้",
        "ระดมทุนจัดซื้อรถวีลแชร์ให้ผู้พิการ",
        "มูลนิธิสร้างรอยยิ้ม ผ่าตัดช่วยเหลือเด็กปากแหว่งเพดานโหว่",
        "กิจกรรมรณรงค์ต่อต้านยาเสพติด",
        "องค์กรส่งเสริมศิลปวัฒนธรรมท้องถิ่น"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_nonprofit.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created DataOutput/thai_dataset_nonprofit.csv")
