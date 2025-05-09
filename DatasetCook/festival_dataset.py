import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ festival
categories = {
    "festival": [
        "สงกรานต์ปีนี้เล่นน้ำที่ไหนดี",
        "ประเพณีลอยกระทง สืบสานวัฒนธรรมไทย",
        "ฉลองปีใหม่ เคาท์ดาวน์กับครอบครัว",
        "เทศกาลฮาโลวีน แต่งตัวเป็นผี",
        "วันวาเลนไทน์ มอบดอกไม้ให้คนรัก",
        "เทศกาลกินเจ งดเนื้อสัตว์",
        "วันคริสต์มาส แลกของขวัญ",
        "ประเพณีแห่เทียนพรรษาที่อุบลราชธานี",
        "เทศกาลตรุษจีน รับอั่งเปา",
        "วันแม่แห่งชาติ บอกรักแม่",
        "วันพ่อแห่งชาติ ทำกิจกรรมกับพ่อ",
        "เทศกาลไหว้พระจันทร์ กินขนมไหว้พระจันทร์",
        "ประเพณีบุญบั้งไฟที่ยโสธร",
        "งานเทศกาลดนตรี Wonderfruit",
        "เทศกาลลอยโคมยี่เป็งที่เชียงใหม่",
        "วันเด็กแห่งชาติ พาเด็กๆ ไปเที่ยว",
        "เทศกาล Oktoberfest ที่เยอรมนี",
        "ประเพณีวิ่งควายที่ชลบุรี",
        "เทศกาล Holi สาดสีที่อินเดีย",
        "วันครูแห่งชาติ ไหว้ครู",
        "เทศกาลดอกไม้ไฟนานาชาติ",
        "ประเพณีตักบาตรเทโว",
        "เทศกาล St. Patrick's Day",
        "วันสงกรานต์แบบ New Normal",
        "การเฉลิมฉลองเทศกาลต่างๆ ทั่วโลก",
        "ประเพณีกวนข้าวทิพย์",
        "เทศกาลภาพยนตร์นานาชาติ",
        "วันแรงงานแห่งชาติ",
        "เทศกาล Mardi Gras ที่นิวออร์ลีนส์",
        "ประเพณีแห่นางแมวขอฝน",
        "เทศกาลบอลลูนนานาชาติ",
        "วันปิยมหาราช",
        "เทศกาลทานาบาตะที่ญี่ปุ่น",
        "ประเพณีรับบัวที่สมุทรปราการ",
        "เทศกาลแข่งเรือยาว",
        "วันรัฐธรรมนูญ",
        "เทศกาล Cherry Blossom ที่ญี่ปุ่นและเกาหลี",
        "ประเพณีไหลเรือไฟที่นครพนม",
        "เทศกาลหน้ากากนานาชาติ",
        "วันจักรี",
        "เทศกาลโคมไฟไต้หวัน",
        "ประเพณีสารทเดือนสิบ",
        "เทศกาลดนตรี Big Mountain",
        "วันฉัตรมงคล",
        "เทศกาลน้ำแข็งที่ฮาร์บิน",
        "ประเพณีกำฟ้าของชาวไทยพวน",
        "เทศกาลอาหารนานาชาติ",
        "วันอาสาฬหบูชาและวันเข้าพรรษา",
        "เทศกาล La Tomatina ปามะเขือเทศที่สเปน",
        "ประเพณีผีตาโขนที่เลย"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_festival.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created DataOutput/thai_dataset_festival.csv")
