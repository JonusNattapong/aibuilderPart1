# -*- coding: utf-8 -*-
import uuid

react_reasoning_data = [
    # --- 3. ReAct (Reasoning + Acting) ---
    # ReAct examples often require interaction or tool use simulation.
    {
        "id": str(uuid.uuid4()),
        "task": "Question Answering (Info Lookup)",
        "reasoning_type_tested": "ReAct",
        "input_data": {
            "question": "นายกรัฐมนตรีคนปัจจุบันของประเทศไทยคือใคร และดำรงตำแหน่งตั้งแต่เมื่อไหร่?",
        },
        "expected_output": {
            "reasoning_process_simulation": [
                {"thought": "ต้องการข้อมูลนายกฯ คนปัจจุบันและวันเริ่มดำรงตำแหน่ง"},
                {"action": "Search('นายกรัฐมนตรีไทยคนปัจจุบัน')"},
                {"observation": "ผลการค้นหา: เศรษฐา ทวีสิน"},
                {"action": "Search('เศรษฐา ทวีสิน เริ่มดำรงตำแหน่งนายก')"},
                {"observation": "ผลการค้นหา: 22 สิงหาคม 2566"},
                {"thought": "ได้คำตอบครบถ้วนแล้ว"},
            ],
            "final_answer": "เศรษฐา ทวีสิน ดำรงตำแหน่งตั้งแต่วันที่ 22 สิงหาคม 2566"
        }
    },
]
