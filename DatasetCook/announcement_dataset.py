import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ announcement
categories = {
    "announcement": [
        "ประกาศปิดปรับปรุงระบบชั่วคราว วันที่ 15 มิถุนายน เวลา 00:00 - 03:00 น.",
        "ประกาศผลสอบคัดเลือกพนักงานตำแหน่งการตลาด",
        "แจ้งย้ายสำนักงานใหญ่ไปยังอาคาร XYZ ตั้งแต่วันที่ 1 กรกฎาคม เป็นต้นไป",
        "ประกาศรายชื่อผู้โชคดีได้รับรางวัลจากกิจกรรม...",
        "แจ้งเปลี่ยนแปลงเวลาทำการสาขา...",
        "ประกาศหยุดทำการเนื่องในวันหยุดนักขัตฤกษ์",
        "แจ้งเตือนภัยพายุฤดูร้อน โปรดระมัดระวัง",
        "ประกาศรับสมัครนักศึกษาใหม่ ปีการศึกษา 2567",
        "แจ้งกำหนดการฉีดวัคซีนป้องกันโรค...",
        "ประกาศผลการประกวดออกแบบโลโก้",
        "แจ้งยกเลิกเที่ยวบิน/เที่ยวรถ...",
        "ประกาศรายชื่อผู้ผ่านการคัดเลือกรอบแรก",
        "แจ้งปิดปรับปรุงพื้นที่ส่วนกลางของคอนโด",
        "ประกาศเชิญประชุมผู้ถือหุ้นประจำปี",
        "แจ้งเปลี่ยนแปลงอัตราค่าบริการ",
        "ประกาศรายชื่อผู้ได้รับทุนการศึกษา",
        "แจ้งปิดรับสมัครงานตำแหน่ง...",
        "ประกาศผลการแข่งขันกีฬาภายใน",
        "แจ้งกำหนดการซ้อมรับปริญญา",
        "ประกาศรายชื่อผู้มีสิทธิ์เข้าสอบสัมภาษณ์",
        "แจ้งปิดถนนเพื่อจัดกิจกรรม...",
        "ประกาศผลการเลือกตั้งคณะกรรมการ...",
        "แจ้งเปลี่ยนแปลงเงื่อนไขการให้บริการ",
        "ประกาศรายชื่อผู้ได้รับรางวัลพนักงานดีเด่น",
        "แจ้งปิดปรับปรุงเว็บไซต์ชั่วคราว",
        "ประกาศผลการดำเนินงานไตรมาสล่าสุด",
        "แจ้งกำหนดการอบรมพนักงานใหม่",
        "ประกาศรายชื่อผู้ผ่านการทดลองงาน",
        "แจ้งปิดปรับปรุงสระว่ายน้ำ",
        "ประกาศผลการจับสลากรางวัล...",
        "แจ้งเปลี่ยนแปลงเลขที่บัญชีธนาคารสำหรับชำระค่าบริการ",
        "ประกาศรายชื่อผู้สำเร็จการศึกษา",
        "แจ้งปิดปรับปรุงลิฟต์โดยสาร",
        "ประกาศผลการประมูล...",
        "แจ้งกำหนดการตรวจสุขภาพประจำปี",
        "ประกาศรายชื่อผู้ได้รับคัดเลือกเข้าร่วมโครงการ...",
        "แจ้งปิดปรับปรุงห้องสมุด",
        "ประกาศผลการสำรวจความพึงพอใจ",
        "แจ้งกำหนดการจ่ายเงินปันผล",
        "ประกาศรายชื่อผู้ได้รับแต่งตั้งตำแหน่งใหม่",
        "แจ้งปิดปรับปรุงระบบไฟฟ้า",
        "ประกาศผลการพิจารณาข้อเสนอโครงการ",
        "แจ้งกำหนดการสัมมนาวิชาการ",
        "ประกาศรายชื่อผู้ได้รับโล่เกียรติคุณ",
        "แจ้งปิดปรับปรุงห้องออกกำลังกาย",
        "ประกาศผลการคัดเลือกนักกีฬาตัวแทน...",
        "แจ้งกำหนดการประชุมใหญ่สามัญประจำปี",
        "ประกาศรายชื่อผู้ได้รับสิทธิ์ซื้อสินค้า...",
        "แจ้งปิดปรับปรุงระบบน้ำประปา",
        "ประกาศผลการตัดสินโครงงานวิทยาศาสตร์"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_announcement.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created DataOutput/thai_dataset_announcement.csv")
