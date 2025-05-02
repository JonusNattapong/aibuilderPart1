# -*- coding: utf-8 -*-
"""
Sample dataset for the Biology domain covering various NLP tasks.
Can also be run to generate biology_data.parquet in the DataOutput directory.
Requires pandas and pyarrow: pip install pandas pyarrow
"""
import os
import pandas as pd

# --- 1. Summarization (Biological Process / Concept) ---
biology_summarization_data = [
    {
        "document": "Cellular respiration is a set of metabolic reactions and processes that take place in the cells of organisms to convert chemical energy from nutrients into adenosine triphosphate (ATP), and then release waste products. The reactions involved in respiration are catabolic reactions, which break large molecules into smaller ones, releasing energy. The main stages are glycolysis, the Krebs cycle (citric acid cycle), and oxidative phosphorylation (electron transport chain).",
        "summary": "Cellular respiration is the cellular process of converting nutrient energy into ATP (energy currency) through catabolic reactions. Key stages include glycolysis, the Krebs cycle, and oxidative phosphorylation, releasing waste products."
    },
    # ... more summarization examples
]

# --- 2. Open QA (Definitions / Explanations) ---
biology_open_qa_data = [
    {
        "context": "DNA (Deoxyribonucleic acid) is a molecule composed of two polynucleotide chains that coil around each other to form a double helix carrying genetic instructions for the development, functioning, growth and reproduction of all known organisms and many viruses. DNA and RNA are nucleic acids. Alongside proteins, lipids and complex carbohydrates (polysaccharides), nucleic acids are one of the four major types of macromolecules that are essential for all known forms of life.",
        "question": "What is the primary function of DNA?",
        "answer": "DNA's primary function is to carry the genetic instructions necessary for the development, functioning, growth, and reproduction of living organisms and many viruses."
    },
    # ... more open QA examples
]

# --- 3. Close QA (Specific Facts / Terminology) ---
biology_close_qa_data = [
    {
        "context": "The mitochondria are often called the 'powerhouses' of the cell. They generate most of the cell's supply of adenosine triphosphate (ATP), used as a source of chemical energy. Mitochondria are found in nearly all eukaryotic cells.",
        "question": "What organelle is known as the 'powerhouse' of the cell?",
        "answer_text": "mitochondria"
    },
    # ... more close QA examples
]

# --- 4. Classification (e.g., Kingdom, Process Type, Cell Type) ---
biology_classification_data = [
    {"text": "A multicellular organism that obtains nutrition by absorbing dissolved molecules, typically by decomposing organic material.", "label": "Fungi"},
    {"text": "The process by which green plants use sunlight, water, and carbon dioxide to create their own food.", "label": "Photosynthesis"},
    {"text": "A type of cell that lacks a nucleus and other membrane-bound organelles.", "label": "Prokaryotic Cell"},
    {"text": "The division of a single cell into two genetically identical daughter cells.", "label": "Mitosis"},
    # ... more classification examples
]

# --- 5. Creative Writing (Process Description / Hypothetical Scenario) ---
biology_creative_writing_data = [
    {
        "prompt": "Describe the journey of a red blood cell through the circulatory system.",
        "generated_text": "Starting in the lungs, I pick up precious oxygen, turning bright red. Pumped by the heart, I race through arteries, branching into smaller vessels until I reach the capillaries. Here, I squeeze through, delivering oxygen to needy tissues and picking up carbon dioxide waste. Now darker, I travel back through veins towards the heart, ready for another trip to the lungs to refuel."
    },
    {
        "prompt": "Explain the concept of natural selection using a hypothetical example.",
        "generated_text": "Imagine a population of rabbits in a snowy environment. Most have brown fur, but a rare mutation causes some to have white fur. Brown rabbits are easily spotted by predators against the snow, while white rabbits blend in. Over time, more white rabbits survive and reproduce, passing on the white fur trait. Eventually, the population becomes predominantly white-furred due to this 'survival of the fittest' – that's natural selection."
    },
    # ... more creative writing examples
]

# --- 6. Brainstorming (Research Questions / Conservation Ideas) ---
biology_brainstorming_data = [
    {
        "topic": "Research questions about the impact of microplastics on marine life",
        "ideas": [
            "How do microplastics accumulate in different trophic levels?",
            "What are the physiological effects of microplastic ingestion on fish?",
            "Do microplastics affect the reproductive success of marine invertebrates?",
            "Can microplastics act as vectors for harmful chemicals or pathogens?",
            "What are the long-term ecosystem-level consequences of microplastic pollution?"
        ]
    },
    # ... more brainstorming examples
]

# --- 7. Multiple Choice QA (Biological Concepts) ---
biology_mcq_data = [
    {
        "context": "Genes are segments of DNA that contain the instructions for building specific proteins or functional RNA molecules. These proteins perform most life functions and make up the majority of cellular structures.",
        "question": "What do genes primarily contain instructions for building?",
        "choices": ["Carbohydrates", "Lipids", "Proteins", "Minerals"],
        "answer_index": 2 # Index of "Proteins"
    },
    # ... more multiple choice QA examples
]

# Combine all data
biology_domain_data = {
    "summarization": biology_summarization_data,
    "open_qa": biology_open_qa_data,
    "close_qa": biology_close_qa_data,
    "classification": biology_classification_data,
    "creative_writing": biology_creative_writing_data,
    "brainstorming": biology_brainstorming_data,
    "multiple_choice_qa": biology_mcq_data,
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
    print("--- Biology Domain Dataset Samples ---")
    print(f"\nSummarization Example:\n{biology_domain_data['summarization'][0]}")
    print(f"\nOpen QA Example:\n{biology_domain_data['open_qa'][0]}")
    print(f"\nClose QA Example:\n{biology_domain_data['close_qa'][0]}")
    print(f"\nClassification Example:\n{biology_domain_data['classification'][0]}")
    print(f"\nCreative Writing Example:\n{biology_domain_data['creative_writing'][0]}")
    print(f"\nBrainstorming Example:\n{biology_domain_data['brainstorming'][0]}")
    print(f"\nMultiple Choice QA Example:\n{biology_domain_data['multiple_choice_qa'][0]}")

    # --- Generate Parquet File ---
    print("\n--- Generating Parquet file ---")
    all_records = []
    domain_name = "biology"
    for task, data in biology_domain_data.items():
        all_records.extend(preprocess_data_for_parquet(domain_name, task, data))

    if all_records:
        df = pd.DataFrame(all_records)
        output_filename = "biology_data.parquet"
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
