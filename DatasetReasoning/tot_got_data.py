# -*- coding: utf-8 -*-
import uuid

tot_got_reasoning_data = [
    # --- 2. Tree of Thoughts (ToT) / Graph of Thoughts (GoT) ---
    # ToT/GoT often involve exploring multiple paths. The dataset might store the prompt and expected final outcome,
    # or potentially log the exploration paths if the testing framework supports it.
    {
        "id": str(uuid.uuid4()),
        "task": "Text Generation (Planning)",
        "reasoning_type_tested": "ToT/GoT",
        "input_data": {
            "prompt": "วางแผนการเดินทาง 3 วัน 2 คืนไปเชียงใหม่สำหรับครอบครัว 4 คน เน้นธรรมชาติและวัฒนธรรม งบประมาณจำกัด",
        },
        "expected_output": {
            # The expected output could be a well-structured plan, implying exploration of options (places, activities, costs)
            "final_plan": """
            วันที่ 1:
            - เช้า: เดินทางถึงเชียงใหม่ เข้าที่พัก (เลือกโฮมสเตย์นอกเมือง)
            - บ่าย: เที่ยวดอยสุเทพ ไหว้พระธาตุ
            - เย็น: เดินถนนคนเดินท่าแพ (หาของกินราคาประหยัด)
            วันที่ 2:
            - เช้า: เดินทางไปดอยอินทนนท์ ชมธรรมชาติ เส้นทางศึกษาธรรมชาติกิ่วแม่ปาน
            - บ่าย: แวะน้ำตกวชิรธาร
            - เย็น: กลับที่พัก ทำอาหารทานเอง (ประหยัดงบ)
            วันที่ 3:
            - เช้า: เยี่ยมชมหมู่บ้านหัตถกรรม (บ่อสร้าง/สันกำแพง)
            - บ่าย: เดินทางกลับ
            """,
            "evaluation_criteria": ["Feasibility", "Budget Adherence", "Coverage of Nature/Culture", "Logical Flow"]
        }
    },
]
