# -*- coding: utf-8 -*-
"""
Sample dataset for the Chemistry domain covering various NLP tasks.
Can also be run to generate chemistry_data.parquet in the DataOutput directory.
Requires pandas and pyarrow: pip install pandas pyarrow
"""
import os
import pandas as pd

# --- 1. Summarization (Chemical Process / Concept) ---
chemistry_summarization_data = [
    {
        "document": "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that, through cellular respiration, can later be released to fuel the organisms' activities. This chemical energy is stored in carbohydrate molecules, such as sugars and starches, which are synthesized from carbon dioxide and water – hence the name photosynthesis, from the Greek φῶς, phos, 'light', and σύνθεσις, synthesis, 'putting together'. In most cases, oxygen is also released as a waste product.",
        "summary": "Photosynthesis is the process where plants convert light energy, carbon dioxide, and water into chemical energy (stored as carbohydrates like sugars) to fuel their activities, releasing oxygen as a byproduct."
    },
    # ... more summarization examples
]

# --- 2. Open QA (Definitions / Explanations) ---
chemistry_open_qa_data = [
    {
        "context": "An acid is a molecule or ion capable of donating a proton (hydrogen ion H+) (a Brønsted–Lowry acid), or, alternatively, capable of forming a covalent bond with an electron pair (a Lewis acid). Acids have a pH less than 7, taste sour (though tasting is not advised), and react with bases and certain metals.",
        "question": "What is the definition of an acid according to Brønsted–Lowry theory?",
        "answer": "According to Brønsted–Lowry theory, an acid is a molecule or ion that can donate a proton (H+)."
    },
    # ... more open QA examples
]

# --- 3. Close QA (Specific Facts / Properties) ---
chemistry_close_qa_data = [
    {
        "context": "Water (H₂O) is a polar molecule due to the uneven distribution of electron density. Oxygen is more electronegative than hydrogen, pulling the shared electrons closer, which results in a partial negative charge on the oxygen atom and partial positive charges on the hydrogen atoms. The molecule has a bent geometry.",
        "question": "What is the chemical formula for water?",
        "answer_text": "H₂O"
    },
    # ... more close QA examples
]

# --- 4. Classification (e.g., Reaction Type, Substance Type) ---
chemistry_classification_data = [
    {"text": "The reaction involves the combination of an acid and a base to form salt and water.", "label": "neutralization_reaction"},
    {"text": "A substance composed of only one type of atom, listed on the periodic table.", "label": "element"},
    {"text": "Process where a substance gains electrons.", "label": "reduction_reaction"},
    {"text": "A homogeneous mixture of two or more substances.", "label": "solution"},
    # ... more classification examples
]

# --- 5. Creative Writing (Procedure Description / Hypothetical Scenario) ---
chemistry_creative_writing_data = [
    {
        "prompt": "Describe the steps for performing a simple titration experiment.",
        "generated_text": "To perform a titration: 1. Rinse and fill a burette with the titrant solution of known concentration. 2. Accurately measure a volume of the analyte solution (unknown concentration) into an Erlenmeyer flask. 3. Add a few drops of an appropriate indicator to the analyte. 4. Slowly add the titrant from the burette to the analyte while swirling the flask, until the indicator shows a permanent color change (endpoint). 5. Record the volume of titrant used."
    },
    {
        "prompt": "Explain what might happen if you mix sodium metal with water.",
        "generated_text": "Mixing sodium metal with water results in a vigorous and potentially explosive exothermic reaction. Sodium reacts rapidly with water to produce sodium hydroxide (NaOH) and hydrogen gas (H₂). The heat generated can ignite the hydrogen gas, causing fire or explosion. 2Na(s) + 2H₂O(l) → 2NaOH(aq) + H₂(g)"
    },
    # ... more creative writing examples
]

# --- 6. Brainstorming (Experiment Ideas / Safety Measures) ---
chemistry_brainstorming_data = [
    {
        "topic": "Safety precautions when working with strong acids",
        "ideas": [
            "Always wear appropriate personal protective equipment (PPE): safety goggles, lab coat, gloves.",
            "Work in a well-ventilated area or fume hood.",
            "Always add acid to water slowly, never water to acid, to control heat generation.",
            "Have neutralizing agents (like sodium bicarbonate) readily available for spills.",
            "Know the location and use of safety showers and eyewash stations."
        ]
    },
    # ... more brainstorming examples
]

# --- 7. Multiple Choice QA (Chemical Concepts) ---
chemistry_mcq_data = [
    {
        "context": "The pH scale measures how acidic or basic a substance is. It ranges from 0 to 14. A pH of 7 is neutral. A pH less than 7 indicates acidity, while a pH greater than 7 indicates basicity (alkalinity).",
        "question": "A solution with a pH of 3 is considered:",
        "choices": ["Neutral", "Basic", "Acidic", "Salty"],
        "answer_index": 2 # Index of "Acidic"
    },
    # ... more multiple choice QA examples
]

# Combine all data
chemistry_domain_data = {
    "summarization": chemistry_summarization_data,
    "open_qa": chemistry_open_qa_data,
    "close_qa": chemistry_close_qa_data,
    "classification": chemistry_classification_data,
    "creative_writing": chemistry_creative_writing_data,
    "brainstorming": chemistry_brainstorming_data,
    "multiple_choice_qa": chemistry_mcq_data,
}

# Function to preprocess data into a standard format for Parquet (copied for simplicity)
def preprocess_data_for_parquet(domain_name, task_name, data_list):
    processed = []
    for item in data_list:
        input_text = ""
        target_text = ""
        prefix = f"{domain_name} {task_name}: "

        if task_name == "summarization":
            input_text = prefix + item.get("document", "")
            target_text = item.get("summary", "")
        elif task_name == "open_qa":
            input_text = prefix + f"question: {item.get('question', '')} context: {item.get('context', '')}"
            target_text = item.get("answer", "")
        elif task_name == "close_qa":
            input_text = prefix + f"question: {item.get('question', '')} context: {item.get('context', '')}"
            target_text = item.get("answer_text", "")
        elif task_name == "classification":
            input_text = prefix + item.get("text", "")
            target_text = item.get("label", "")
        elif task_name == "creative_writing":
            input_text = prefix + item.get("prompt", "")
            target_text = item.get("generated_text", "")
        elif task_name == "brainstorming":
            input_text = prefix + item.get("topic", "")
            target_text = "\n".join(item.get("ideas", []))
        elif task_name == "multiple_choice_qa":
            choices_str = " | ".join(item.get("choices", []))
            input_text = prefix + f"question: {item.get('question', '')} choices: {choices_str} context: {item.get('context', '')}"
            answer_idx = item.get("answer_index")
            if answer_idx is not None and item.get("choices"):
                target_text = item["choices"][answer_idx]

        if input_text and target_text:
            processed.append({
                "domain": domain_name,
                "task": task_name,
                "input_text": input_text,
                "target_text": target_text
            })
    return processed


if __name__ == '__main__':
    print("--- Chemistry Domain Dataset Samples ---")
    print(f"\nSummarization Example:\n{chemistry_domain_data['summarization'][0]}")
    print(f"\nOpen QA Example:\n{chemistry_domain_data['open_qa'][0]}")
    print(f"\nClose QA Example:\n{chemistry_domain_data['close_qa'][0]}")
    print(f"\nClassification Example:\n{chemistry_domain_data['classification'][0]}")
    print(f"\nCreative Writing Example:\n{chemistry_domain_data['creative_writing'][0]}")
    print(f"\nBrainstorming Example:\n{chemistry_domain_data['brainstorming'][0]}")
    print(f"\nMultiple Choice QA Example:\n{chemistry_domain_data['multiple_choice_qa'][0]}")

    # --- Generate Parquet File ---
    print("\n--- Generating Parquet file ---")
    all_records = []
    domain_name = "chemistry"
    for task, data in chemistry_domain_data.items():
        all_records.extend(preprocess_data_for_parquet(domain_name, task, data))

    if all_records:
        df = pd.DataFrame(all_records)
        output_filename = "chemistry_data.parquet"
        # Define output directory relative to the script's location
        script_dir = os.path.dirname(__file__)
        # Go up one level from Parquet, then into DataOutput
        output_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'DataOutput'))
        # Create DataOutput directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        try:
            df.to_parquet(output_path, index=False)
            print(f"Successfully generated {output_filename} at {output_path}")
            print(f"DataFrame Info:\n{df.info()}")
            print(f"\nFirst 5 rows:\n{df.head().to_string()}")
        except Exception as e:
            print(f"Error saving Parquet file: {e}")
            print("Please ensure 'pandas' and 'pyarrow' are installed (`pip install pandas pyarrow`)")
    else:
        print("No records processed, Parquet file not generated.")
