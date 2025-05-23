import csv
import uuid
import os

# ตรวจสอบและสร้างไดเรกทอรี DataOutput หากยังไม่มี
output_dir = 'DataOutput'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ตัวอย่างข้อมูล Emotion Detection
emotion_data = {
    "joy": [ # ดีใจ
        "เย้! สอบผ่านแล้ว ดีใจที่สุดเลย!",
        "ได้รับของขวัญเซอร์ไพรส์ ถูกใจมาก ขอบคุณนะ",
        "วันนี้เป็นวันที่มีความสุขมากจริงๆ",
        "ยิ้มไม่หุบเลย ได้เจอเพื่อนเก่าโดยบังเอิญ",
        "ตื่นเต้นจัง พรุ่งนี้จะได้ไปเที่ยวแล้ว!",
        "ภูมิใจในตัวเองมากที่ทำสำเร็จ",
        "หัวเราะจนท้องแข็งเลย เรื่องนี้ตลกมาก",
        "รู้สึกอบอุ่นใจที่ได้อยู่กับครอบครัว",
        "ดีใจที่ได้ช่วยเหลือคนอื่น",
        "มีความสุขกับสิ่งเล็กๆ น้อยๆ รอบตัว",
    ],
    "sadness": [ # เศร้า
        "เสียใจจังที่ทำผิดพลาดไป",
        "รู้สึกเหงามากวันนี้ ไม่มีใครคุยด้วยเลย",
        "น้ำตาจะไหล คิดถึงบ้านจัง",
        "ผิดหวังกับผลลัพธ์ ไม่เป็นอย่างที่คิดไว้เลย",
        "เศร้าใจที่ต้องบอกลา",
        "รู้สึกหดหู่ ไม่รู้จะทำยังไงต่อไป",
        "ใจสลายเลยที่ได้ยินข่าวร้าย",
        "ท้อแท้จัง เหมือนไม่มีใครเข้าใจเราเลย",
        "เสียดายโอกาสที่หลุดลอยไป",
        "วันนี้มันดูเงียบเหงาผิดปกติ",
    ],
    "anger": [ # โกรธ
        "โมโหมาก! ทำไมถึงทำแบบนี้!",
        "หงุดหงิดจริงๆ พูดไม่รู้เรื่องเลย!",
        "ทนไม่ไหวแล้วนะ อย่ามาหาเรื่องกัน!",
        "โกรธจนตัวสั่น ไม่คิดว่าจะกล้าพูดแบบนี้",
        "รำคาญที่สุด! หยุดทำเสียงดังสักที!",
        "ไม่พอใจอย่างแรงที่โดนเอาเปรียบ",
        "ทำไมถึงไม่มีความรับผิดชอบเลย!",
        "อย่ามายุ่งเรื่องของฉัน!",
        "เหลืออดแล้วนะ พูดดีๆ ไม่ฟังใช่ไหม!",
        "ประสาทจะกินกับเรื่องวุ่นวายพวกนี้!",
    ],
    "love": [ # หลงใหล / รัก
        "ฉันรักคุณมากที่สุดในโลก",
        "คิดถึงเธอจัง อยากเจอหน้าทุกวันเลย",
        "อยู่กับคุณแล้วรู้สึกปลอดภัยและมีความสุข",
        "คุณคือคนสำคัญที่สุดในชีวิตของฉัน",
        "รอยยิ้มของคุณทำให้ใจฉันละลาย",
        "อยากใช้เวลาอยู่กับคุณตลอดไป",
        "ขอบคุณที่เข้ามาในชีวิตนะ",
        "คุณทำให้ฉันเป็นคนที่ดีขึ้น",
        "ตกหลุมรักเธอตั้งแต่แรกเจอ",
        "ไม่ว่าคุณจะเป็นยังไง ฉันก็รักคุณเสมอ",
    ],
    # อาจเพิ่ม Neutral หรือ Surprise ได้ถ้าต้องการ
    # "neutral": [
    #     "วันนี้วันอังคาร",
    #     "กาแฟแก้วนี้ราคา 50 บาท",
    # ],
    # "surprise": [
    #     "อ้าว! มาได้ยังไงเนี่ย ไม่เห็นบอกกันก่อนเลย",
    #     "จริงเหรอ! ไม่น่าเชื่อเลยว่าจะเกิดขึ้นได้",
    # ]
    # เพิ่มเติมตัวอย่างตามต้องการ
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in emotion_data.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV ในโฟลเดอร์ DataOutput
output_file = os.path.join(output_dir, 'thai_dataset_emotion_detection.csv')
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'emotion'])
    writer.writerows(rows)

print(f"Created {output_file}")
