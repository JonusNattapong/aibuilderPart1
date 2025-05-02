# -*- coding: utf-8 -*-
"""
Sample dataset for the Music domain covering various NLP tasks.
Can also be run to generate music_data.parquet in the DataOutput directory.
Requires pandas and pyarrow: pip install pandas pyarrow
"""
import os
import pandas as pd

# --- 1. Summarization (Genre / Artist Bio) ---
music_summarization_data = [
    {
        "document": "Jazz is a music genre that originated in the African-American communities of New Orleans, Louisiana, in the late 19th and early 20th centuries, with its roots in blues and ragtime. Jazz is characterized by swing and blue notes, complex chords, call and response vocals, polyrhythms and improvisation.",
        "summary": "Jazz, originating from African-American communities in New Orleans around the turn of the 20th century, evolved from blues and ragtime. It's known for swing, blue notes, complex chords, call and response, polyrhythms, and improvisation."
    },
    # ... more summarization examples
]

# --- 2. Open QA (Music Theory / History) ---
music_open_qa_data = [
    {
        "context": "A time signature in music notation indicates how many beats (pulses) are contained in each measure (bar), and which note value is equivalent to a beat. For example, 4/4 time (common time) means there are four beats per measure, and a quarter note gets one beat.",
        "question": "What does the top number in a time signature represent?",
        "answer": "The top number in a time signature indicates the number of beats in each measure."
    },
    # ... more open QA examples
]

# --- 3. Close QA (Specific Song/Artist Fact) ---
music_close_qa_data = [
    {
        "context": "The Beatles were an English rock band formed in Liverpool in 1960, comprising John Lennon, Paul McCartney, George Harrison and Ringo Starr. They are regarded as the most influential band of all time.",
        "question": "In which city was The Beatles formed?",
        "answer_text": "Liverpool"
    },
    # ... more close QA examples
]

# --- 4. Classification (e.g., Genre, Mood, Instrument) ---
music_classification_data = [
    {"text": "Fast tempo electronic music with a repetitive beat, often played in clubs.", "label": "Techno"},
    {"text": "A sad, slow song featuring acoustic guitar and harmonica.", "label": "Folk Ballad"},
    {"text": "Upbeat, energetic rock song with prominent electric guitar riffs and drums.", "label": "Rock"},
    {"text": "Calm, relaxing instrumental piece featuring piano and strings.", "label": "Ambient"},
    # ... more classification examples
]

# --- 5. Creative Writing (Song Lyrics / Music Review) ---
music_creative_writing_data = [
    {
        "prompt": "Write a few lines of lyrics for a blues song about losing a job.",
        "generated_text": "Woke up this mornin', sun refused to shine\nGot that letter yesterday, said my job ain't mine\nEmpty pockets, heavy heart, don't know where to start\nYeah, the blues got a hold on me, tearin' me apart."
    },
    {
        "prompt": "Write a short review of a live concert.",
        "generated_text": "The energy in the arena was electric from the first note. The band delivered a powerhouse performance, blending classic hits with new material seamlessly. The lead singer's vocals were stunning, and the light show added another layer of spectacle. An unforgettable night of pure rock and roll."
    },
    # ... more creative writing examples
]

# --- 6. Brainstorming (Songwriting Ideas / Band Names) ---
music_brainstorming_data = [
    {
        "topic": "Ideas for song themes about city life",
        "ideas": [
            "The loneliness of crowds",
            "Finding beauty in urban decay",
            "The rhythm of the subway",
            "Late-night adventures",
            "Dreams and struggles in the concrete jungle"
        ]
    },
    # ... more brainstorming examples
]

# --- 7. Multiple Choice QA (Music Terms) ---
music_mcq_data = [
    {
        "context": "Tempo refers to the speed or pace of a given piece of music. It is typically measured in beats per minute (BPM). Common tempo markings include Largo (very slow), Adagio (slow), Andante (walking pace), Moderato (moderate), Allegro (fast), and Presto (very fast).",
        "question": "Which tempo marking indicates a fast speed?",
        "choices": ["Adagio", "Andante", "Moderato", "Allegro"],
        "answer_index": 3 # Index of "Allegro"
    },
    # ... more multiple choice QA examples
]

# Combine all data
music_domain_data = {
    "summarization": music_summarization_data,
    "open_qa": music_open_qa_data,
    "close_qa": music_close_qa_data,
    "classification": music_classification_data,
    "creative_writing": music_creative_writing_data,
    "brainstorming": music_brainstorming_data,
    "multiple_choice_qa": music_mcq_data,
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
    print("--- Music Domain Dataset Samples ---")
    print(f"\nSummarization Example:\n{music_domain_data['summarization'][0]}")
    print(f"\nOpen QA Example:\n{music_domain_data['open_qa'][0]}")
    print(f"\nClose QA Example:\n{music_domain_data['close_qa'][0]}")
    print(f"\nClassification Example:\n{music_domain_data['classification'][0]}")
    print(f"\nCreative Writing Example:\n{music_domain_data['creative_writing'][0]}")
    print(f"\nBrainstorming Example:\n{music_domain_data['brainstorming'][0]}")
    print(f"\nMultiple Choice QA Example:\n{music_domain_data['multiple_choice_qa'][0]}")

    # --- Generate Parquet File ---
    print("\n--- Generating Parquet file ---")
    all_records = []
    domain_name = "music"
    for task, data in music_domain_data.items():
        all_records.extend(preprocess_data_for_parquet(domain_name, task, data))

    if all_records:
        df = pd.DataFrame(all_records)
        output_filename = "music_data.parquet"
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
