# -*- coding: utf-8 -*-
"""
Example dataset for Text Summarization.
Pairs of longer documents and their shorter summaries.
"""

summarization_data = [
    {
        "document": "การประชุมสุดยอดผู้นำเอเปคครั้งล่าสุดได้เสร็จสิ้นลงแล้ว โดยมีประเด็นสำคัญคือการหารือเกี่ยวกับความร่วมมือทางเศรษฐกิจในภูมิภาคเอเชียแปซิฟิก การส่งเสริมการค้าดิจิทัล และการรับมือกับการเปลี่ยนแปลงสภาพภูมิอากาศ ผู้นำประเทศสมาชิกได้ลงนามในปฏิญญาร่วมกันเพื่อยืนยันเจตนารมณ์ในการทำงานร่วมกันเพื่อเป้าหมายดังกล่าว",
        "summary": "การประชุมเอเปคสรุปผลหารือความร่วมมือเศรษฐกิจ การค้าดิจิทัล และการรับมือโลกร้อน ผู้นำลงนามปฏิญญาร่วม"
    },
    {
        "document": "งานวิจัยชิ้นใหม่ที่ตีพิมพ์ในวารสาร Nature Communications พบว่าการนอนหลับไม่เพียงพอมีความสัมพันธ์กับการทำงานของสมองที่ลดลง โดยเฉพาะในส่วนที่เกี่ยวข้องกับความจำและการตัดสินใจ ทีมวิจัยได้ทำการทดลองกับอาสาสมัครจำนวน 100 คน โดยแบ่งเป็นกลุ่มที่นอนหลับปกติและกลุ่มที่นอนน้อยกว่า 6 ชั่วโมงต่อคืน ผลการทดสอบพบว่ากลุ่มที่นอนน้อยมีประสิทธิภาพในการทำแบบทดสอบความจำต่ำกว่าอย่างมีนัยสำคัญ",
        "summary": "งานวิจัยชี้การนอนน้อยกว่า 6 ชั่วโมงส่งผลเสียต่อความจำและการตัดสินใจ"
    },
    {
        "document": "ตลาดหลักทรัพย์แห่งประเทศไทย (ตลท.) รายงานว่าดัชนีหุ้นไทยปรับตัวลดลงในช่วงเช้าวันนี้ ตามทิศทางตลาดหุ้นต่างประเทศ เนื่องจากความกังวลเกี่ยวกับภาวะเงินเฟ้อและการปรับขึ้นอัตราดอกเบี้ยของธนาคารกลางสหรัฐฯ อย่างไรก็ตาม นักวิเคราะห์คาดว่าตลาดหุ้นไทยยังมีปัจจัยพื้นฐานที่แข็งแกร่งและอาจฟื้นตัวได้ในช่วงบ่าย",
        "summary": "หุ้นไทยเช้าปรับลงตามตลาดต่างประเทศ กังวลเงินเฟ้อและดอกเบี้ยสหรัฐฯ นักวิเคราะห์คาดอาจฟื้นตัวช่วงบ่าย"
    }
]

# Example usage:
# import pandas as pd
# df = pd.DataFrame(summarization_data)
# print(df.head())
