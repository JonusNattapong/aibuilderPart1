# -*- coding: utf-8 -*-
"""
Sample dataset for the Climate domain covering various NLP tasks.
Can also be run to generate climate_data.parquet in the DataOutput directory.
Requires pandas and pyarrow: pip install pandas pyarrow
"""
import os
import pandas as pd

# --- 1. Summarization (Climate Report / Concept) ---
climate_summarization_data = [
    {
        "document": "Global warming is the long-term heating of Earth's climate system observed since the pre-industrial period (between 1850 and 1900) due to human activities, primarily fossil fuel burning, which increases heat-trapping greenhouse gas levels in Earth's atmosphere. It is most commonly measured as the average increase in Earth's global surface temperature. Consequences include rising sea levels, changes in precipitation patterns, and more frequent extreme weather events.",
        "summary": "Global warming refers to the long-term heating of Earth's climate since the pre-industrial era, caused mainly by human activities like burning fossil fuels, which release greenhouse gases. It leads to rising sea levels, altered precipitation, and more extreme weather."
    },
    # ... more summarization examples
]

# --- 2. Open QA (Causes / Effects / Solutions) ---
climate_open_qa_data = [
    {
        "context": "Renewable energy sources, such as solar, wind, hydro, geothermal, and biomass, are alternatives to fossil fuels that help reduce greenhouse gas emissions. Transitioning to renewable energy is a key strategy for mitigating climate change. Energy efficiency improvements also play a crucial role.",
        "question": "What are some examples of renewable energy sources that help combat climate change?",
        "answer": "Examples of renewable energy sources include solar power, wind power, hydropower, geothermal energy, and biomass energy. They help reduce greenhouse gas emissions compared to fossil fuels."
    },
    # ... more open QA examples
]

# --- 3. Close QA (Specific Data / Policy Detail) ---
climate_close_qa_data = [
    {
        "context": "The Paris Agreement, adopted in 2015, is a legally binding international treaty on climate change. Its goal is to limit global warming to well below 2, preferably to 1.5 degrees Celsius, compared to pre-industrial levels. Countries submit nationally determined contributions (NDCs) outlining their climate action plans.",
        "question": "What is the main temperature goal of the Paris Agreement?",
        "answer_text": "limit global warming to well below 2, preferably to 1.5 degrees Celsius, compared to pre-industrial levels"
    },
    # ... more close QA examples
]

# --- 4. Classification (e.g., Impact Type, Mitigation Strategy, Policy) ---
climate_classification_data = [
    {"text": "Increased frequency and intensity of hurricanes in the Atlantic.", "label": "extreme_weather_impact"},
    {"text": "Implementing carbon capture and storage technology at power plants.", "label": "mitigation_technology"},
    {"text": "Government subsidies for electric vehicle purchases.", "label": "climate_policy"},
    {"text": "Melting glaciers contributing to sea-level rise.", "label": "sea_level_rise_impact"},
    # ... more classification examples
]

# --- 5. Creative Writing (Future Scenario / Call to Action) ---
climate_creative_writing_data = [
    {
        "prompt": "Write a short paragraph describing a city in 2050 that has successfully adapted to climate change.",
        "generated_text": "In 2050, the city thrives beneath expansive green roofs and vertical gardens that cool the air. Solar panels adorn nearly every surface, powering silent electric transport gliding along dedicated lanes. Restored wetlands surrounding the city act as natural flood barriers, while advanced water recycling systems ensure resilience against drought. Life continues, harmonized with nature's new realities."
    },
    {
        "prompt": "Write a brief call to action urging individuals to reduce their carbon footprint.",
        "generated_text": "Our planet needs us now! Every action counts in the fight against climate change. Choose sustainable transport, reduce waste, conserve energy, and make conscious consumption choices. Let's work together to build a healthier, more sustainable future for all. Start today!"
    },
    # ... more creative writing examples
]

# --- 6. Brainstorming (Policy Ideas / Adaptation Measures) ---
climate_brainstorming_data = [
    {
        "topic": "Ideas for reducing plastic waste in communities",
        "ideas": [
            "Promote reusable bags, bottles, and containers",
            "Implement community composting programs",
            "Support local refill stations for household products",
            "Organize regular cleanup events",
            "Advocate for policies limiting single-use plastics"
        ]
    },
    # ... more brainstorming examples
]

# --- 7. Multiple Choice QA (Climate Science Terms) ---
climate_mcq_data = [
    {
        "context": "Greenhouse gases (GHGs) are gases in Earth's atmosphere that trap heat. They let sunlight pass through the atmosphere but prevent the heat that the sunlight brings from leaving the atmosphere. The main GHGs are water vapor, carbon dioxide (CO₂), methane (CH₄), nitrous oxide (N₂O), and ozone (O₃).",
        "question": "Which of the following is considered a major greenhouse gas?",
        "choices": ["Nitrogen (N₂)", "Oxygen (O₂)", "Carbon Dioxide (CO₂)", "Argon (Ar)"],
        "answer_index": 2 # Index of "Carbon Dioxide (CO₂)"
    },
    # ... more multiple choice QA examples
]

# Combine all data
climate_domain_data = {
    "summarization": climate_summarization_data,
    "open_qa": climate_open_qa_data,
    "close_qa": climate_close_qa_data,
    "classification": climate_classification_data,
    "creative_writing": climate_creative_writing_data,
    "brainstorming": climate_brainstorming_data,
    "multiple_choice_qa": climate_mcq_data,
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
    print("--- Climate Domain Dataset Samples ---")
    print(f"\nSummarization Example:\n{climate_domain_data['summarization'][0]}")
    print(f"\nOpen QA Example:\n{climate_domain_data['open_qa'][0]}")
    print(f"\nClose QA Example:\n{climate_domain_data['close_qa'][0]}")
    print(f"\nClassification Example:\n{climate_domain_data['classification'][0]}")
    print(f"\nCreative Writing Example:\n{climate_domain_data['creative_writing'][0]}")
    print(f"\nBrainstorming Example:\n{climate_domain_data['brainstorming'][0]}")
    print(f"\nMultiple Choice QA Example:\n{climate_domain_data['multiple_choice_qa'][0]}")

    # --- Generate Parquet File ---
    print("\n--- Generating Parquet file ---")
    all_records = []
    domain_name = "climate"
    for task, data in climate_domain_data.items():
        all_records.extend(preprocess_data_for_parquet(domain_name, task, data))

    if all_records:
        df = pd.DataFrame(all_records)
        output_filename = "climate_data.parquet"
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
