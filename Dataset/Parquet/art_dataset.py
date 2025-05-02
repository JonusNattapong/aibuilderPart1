# -*- coding: utf-8 -*-
"""
Sample dataset for the Art domain covering various NLP tasks.
Can also be run to generate art_data.parquet in the DataOutput directory.
Requires pandas and pyarrow: pip install pandas pyarrow
"""
import os
import pandas as pd

# --- 1. Summarization (Artist Bio / Art Movement) ---
art_summarization_data = [
    {
        "document": "Vincent van Gogh (1853-1890) was a Dutch Post-Impressionist painter who posthumously became one of the most famous and influential figures in Western art history. In a decade, he created about 2,100 artworks, including around 860 oil paintings, most of them in the last two years of his life. His works include landscapes, still lifes, portraits, and self-portraits, characterized by bold colors and dramatic, impulsive and expressive brushwork.",
        "summary": "Vincent van Gogh, a Dutch Post-Impressionist painter (1853-1890), became highly influential after his death. He produced around 2,100 artworks, mostly oils, in his last decade, known for bold colors and expressive brushwork in landscapes, portraits, and still lifes."
    },
    # ... more summarization examples
]

# --- 2. Open QA (Art History / Concepts) ---
art_open_qa_data = [
    {
        "context": "Impressionism was a 19th-century art movement characterized by relatively small, thin, yet visible brush strokes, open composition, emphasis on accurate depiction of light in its changing qualities (often accentuating the effects of the passage of time), ordinary subject matter, and inclusion of movement as a crucial element of human perception and experience.",
        "question": "What are the key characteristics of the Impressionist art movement?",
        "answer": "Impressionism is known for visible brush strokes, open composition, focus on capturing changing light, depicting ordinary subjects, and conveying movement."
    },
    # ... more open QA examples
]

# --- 3. Close QA (Specific Artwork/Artist Fact) ---
art_close_qa_data = [
    {
        "context": "Leonardo da Vinci's Mona Lisa, housed in the Louvre Museum in Paris, is arguably the most famous painting in the world. It is a half-length portrait painting considered an archetypal masterpiece of the Italian Renaissance.",
        "question": "Which museum houses the Mona Lisa?",
        "answer_text": "the Louvre Museum in Paris"
    },
    # ... more close QA examples
]

# --- 4. Classification (e.g., Art Period, Genre, Medium) ---
art_classification_data = [
    {"text": "A painting featuring dramatic light and shadow, intense emotion, typical of the Baroque period.", "label": "Baroque"},
    {"text": "Sculpture made from carved marble depicting a human figure.", "label": "Sculpture"},
    {"text": "Abstract artwork with geometric shapes and primary colors.", "label": "Abstract Art"},
    {"text": "A landscape painting done with watercolors.", "label": "Watercolor"},
    # ... more classification examples
]

# --- 5. Creative Writing (Art Description / Interpretation) ---
art_creative_writing_data = [
    {
        "prompt": "Write a short description of Claude Monet's 'Water Lilies' series.",
        "generated_text": "Monet's 'Water Lilies' series captures the serene beauty of his garden pond at Giverny. The paintings dissolve form into shimmering reflections of light and color, focusing on the interplay between water, lilies, and the changing atmosphere, creating an immersive and tranquil experience."
    },
    {
        "prompt": "Imagine you are an art critic reviewing a contemporary abstract sculpture. Write a brief interpretation.",
        "generated_text": "The sculpture's sharp angles and contrasting textures evoke a sense of urban tension. Its fragmented form invites viewers to contemplate the fractured nature of modern experience, challenging traditional notions of beauty and coherence."
    },
    # ... more creative writing examples
]

# --- 6. Brainstorming (Art Project Ideas) ---
art_brainstorming_data = [
    {
        "topic": "Ideas for a community mural project",
        "ideas": [
            "Theme based on local history or landmarks",
            "Involve local schools and artists",
            "Use eco-friendly paints",
            "Incorporate interactive elements (e.g., QR codes)",
            "Hold workshops for community participation in design/painting"
        ]
    },
    # ... more brainstorming examples
]

# --- 7. Multiple Choice QA (Art Terms) ---
art_mcq_data = [
    {
        "context": "Chiaroscuro is an Italian artistic term used to describe the dramatic effect of contrasting areas of light and dark in an artwork, particularly paintings. It is used to create a sense of volume, three-dimensionality, and drama.",
        "question": "What artistic technique uses strong contrasts between light and dark?",
        "choices": ["Impasto", "Sfumato", "Chiaroscuro", "Foreshortening"],
        "answer_index": 2 # Index of "Chiaroscuro"
    },
    # ... more multiple choice QA examples
]

# Combine all data
art_domain_data = {
    "summarization": art_summarization_data,
    "open_qa": art_open_qa_data,
    "close_qa": art_close_qa_data,
    "classification": art_classification_data,
    "creative_writing": art_creative_writing_data,
    "brainstorming": art_brainstorming_data,
    "multiple_choice_qa": art_mcq_data,
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
    print("--- Art Domain Dataset Samples ---")
    print(f"\nSummarization Example:\n{art_domain_data['summarization'][0]}")
    print(f"\nOpen QA Example:\n{art_domain_data['open_qa'][0]}")
    print(f"\nClose QA Example:\n{art_domain_data['close_qa'][0]}")
    print(f"\nClassification Example:\n{art_domain_data['classification'][0]}")
    print(f"\nCreative Writing Example:\n{art_domain_data['creative_writing'][0]}")
    print(f"\nBrainstorming Example:\n{art_domain_data['brainstorming'][0]}")
    print(f"\nMultiple Choice QA Example:\n{art_domain_data['multiple_choice_qa'][0]}")

    # --- Generate Parquet File ---
    print("\n--- Generating Parquet file ---")
    all_records = []
    domain_name = "art"
    for task, data in art_domain_data.items():
        all_records.extend(preprocess_data_for_parquet(domain_name, task, data))

    if all_records:
        df = pd.DataFrame(all_records)
        output_filename = "art_data.parquet"
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
