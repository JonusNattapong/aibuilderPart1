# -*- coding: utf-8 -*-
import uuid

par_reasoning_data = [
    # --- 6. Program-aided Reasoning (PaR) ---
    {
        "id": str(uuid.uuid4()),
        "task": "Question Answering (Math/Code)",
        "reasoning_type_tested": "PaR",
        "input_data": {
            "question": "หาผลรวมของเลขคู่ตั้งแต่ 1 ถึง 100",
        },
        "expected_output": {
            "reasoning_steps": [
                "1. ต้องการหาผลรวมของ 2, 4, 6, ..., 100",
                "2. สามารถเขียนโปรแกรมเพื่อคำนวณได้"
            ],
            "generated_code": """
python
total = 0
for i in range(1, 101):
  if i % 2 == 0:
    total += i
print(total)
""",
            "code_execution_result": "2550",
            "final_answer": "2550"
        }
    },
]
