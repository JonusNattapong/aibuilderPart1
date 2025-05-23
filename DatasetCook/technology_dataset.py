import csv
import uuid


# สร้างข้อความใหม่สำหรับหมวด technology และ entertainment
new_categories.update({
    "technology": [
        "เทคโนโลยี 5G ทำให้การสื่อสารรวดเร็วขึ้น",
        "AI เริ่มมีบทบาทในชีวิตประจำวันมากขึ้น",
        "มือถือรุ่นใหม่มีฟีเจอร์กล้องที่ล้ำสมัย",
        "หุ่นยนต์เริ่มเข้ามาแทนแรงงานมนุษย์ในหลายอุตสาหกรรม",
        "การใช้ Cloud Storage ทำให้การจัดเก็บข้อมูลสะดวกขึ้น",
        "การเข้ารหัสข้อมูลมีความสำคัญต่อความปลอดภัย",
        "เรียนรู้การเขียนโค้ดช่วยเปิดโอกาสในอาชีพใหม่ๆ",
        "มีแอปแนะนำสุขภาพที่น่าใช้ไหม",
        "การพัฒนาแอปมือถือต้องใช้ภาษาอะไรบ้าง",
        "เทคโนโลยี Smart Home ทำให้ชีวิตง่ายขึ้น",
        "ใครเคยใช้สมาร์ทวอทช์แล้วรู้สึกว่าเปลี่ยนชีวิตไหม",
        "เทคโนโลยี VR ใช้ในเกมและการศึกษาได้ดี",
        "แบตเตอรี่โทรศัพท์จะพัฒนาให้ทนขึ้นอีกไหม",
        "เทคโนโลยี AI ในการแปลภาษาเริ่มแม่นยำขึ้นเรื่อย ๆ",
        "ใครเคยใช้แว่น AR ในการทำงานบ้าง",
        "แพลตฟอร์ม E-learning ช่วยให้เรียนรู้ได้ทุกที่",
        "Cybersecurity เป็นสิ่งที่ทุกคนควรให้ความสำคัญ",
        "รถยนต์ไฟฟ้ากำลังกลายเป็นกระแสหลัก",
        "ใครใช้ Smart TV แล้วติดใจบ้าง",
        "เทคโนโลยีการสแกนใบหน้ากำลังถูกใช้อย่างแพร่หลาย",
        "บล็อกเชนไม่ได้ใช้แค่กับคริปโตเท่านั้น",
        "ใครเคยลองเขียนโปรแกรมด้วย Python แล้วบ้าง",
        "มีเครื่องมือช่วยทำงานอัตโนมัติอะไรบ้าง",
        "เทคโนโลยีช่วยให้การแพทย์แม่นยำมากขึ้น",
        "ใช้ Google Assistant หรือ Siri กันบ้างไหม",
        "การเก็บข้อมูล Big Data ช่วยในการวิเคราะห์ธุรกิจ",
        "อุปกรณ์ IoT ทำให้บ้านของเราฉลาดขึ้น",
        "มีเว็บเรียนเขียนโปรแกรมฟรีอะไรแนะนำไหม",
        "รู้ไหมว่า AI ถูกใช้ในการวิเคราะห์ตลาดหุ้นด้วย",
        "การอัปเดตระบบปฏิบัติการช่วยเพิ่มความปลอดภัย",
        "ใครเคยลองใช้แอปสร้างภาพด้วย AI บ้าง",
        "เทคโนโลยี Deepfake ทำให้เราต้องตั้งคำถามกับสิ่งที่เห็น",
        "เกมในอนาคตจะใช้ AI เพื่อปรับความยากตามผู้เล่น",
        "ใครเคยทำงานผ่านแพลตฟอร์มออนไลน์บ้าง",
        "การทำงานจากบ้านต้องใช้อุปกรณ์อะไรบ้าง",
        "เทคโนโลยีช่วยให้การเดินทางปลอดภัยมากขึ้น",
        "แนะนำอุปกรณ์เสริมคอมพิวเตอร์สำหรับสายทำงาน",
        "ใครใช้บริการคลาวด์ของไทยบ้าง",
        "เรียนรู้การใช้ Excel ขั้นสูงมีประโยชน์อย่างไร",
        "ใครเคยลองใช้ ChatGPT ทำงานบ้าง",
        "การใช้แอปบริหารเวลาเพิ่มประสิทธิภาพการทำงาน",
        "เทคโนโลยีการพิมพ์ 3 มิติช่วยในอุตสาหกรรมการแพทย์",
        "เคยลองเขียนบล็อกด้วย Markdown หรือยัง",
        "ใครเคยสร้างเว็บไซต์ด้วยตัวเองบ้าง",
        "มีแอปไหนช่วยจัดการการเงินส่วนตัวได้ดี",
        "ใช้ VPN มีประโยชน์ในเรื่องความปลอดภัยยังไง",
        "รู้ไหมว่ามี AI ที่สามารถแต่งเพลงได้แล้ว",
        "เทคโนโลยีใหม่ ๆ ช่วยเพิ่มโอกาสทางธุรกิจ",
        "ใครใช้สมาร์ทโฟนทำงานแทนคอมพิวเตอร์บ้าง"
    ]
})

# เตรียมข้อมูลเพิ่ม
rows = []
for label, texts in new_categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกลง CSV
with open('DataOutput/thai_technology_dataset.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

