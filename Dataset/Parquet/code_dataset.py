# -*- coding: utf-8 -*-
"""
Sample dataset for the Code/Programming domain covering various NLP tasks.
Can also be run to generate code_data.parquet in the DataOutput directory.
Requires pandas and pyarrow: pip install pandas pyarrow
"""
import os
import pandas as pd

# --- 1. Summarization (Code Explanation) ---
code_summarization_data = [
    {
        "document": """
def factorial(n):
    \"\"\"Calculates the factorial of a non-negative integer.\"\"\"
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    elif n == 0 or n == 1:
        return 1
    else:
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result
""",
        "summary": "Python function `factorial(n)` calculates the factorial of `n`. It raises ValueError for negative input, returns 1 for 0 or 1, and iteratively multiplies numbers from 2 to `n` otherwise."
    },
    # ... more summarization examples
]

# --- 2. Open QA (Conceptual Questions) ---
code_open_qa_data = [
    {
        "context": "Object-Oriented Programming (OOP) is a programming paradigm based on the concept of 'objects', which can contain data in the form of fields (often known as attributes or properties) and code, in the form of procedures (often known as methods). Key principles include encapsulation, inheritance, and polymorphism.",
        "question": "What are the main principles of Object-Oriented Programming?",
        "answer": "The main principles of OOP are encapsulation (bundling data and methods), inheritance (allowing classes to inherit properties from others), and polymorphism (allowing objects to be treated as instances of their parent class)."
    },
    # ... more open QA examples
]

# --- 3. Close QA (Code Specific) ---
code_close_qa_data = [
    {
        "context": """
import requests

def get_website_status(url):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return "Online"
        else:
            return f"Error: Status Code {response.status_code}"
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

status = get_website_status("https://example.com")
print(status)
""",
        "question": "Which library is used to make HTTP requests in the Python code?",
        "answer_text": "requests"
    },
    # ... more close QA examples
]

# --- 4. Classification (e.g., Bug Report, Feature Request, Question) ---
code_classification_data = [
    {"text": "The application crashes when I click the save button after editing.", "label": "bug_report"},
    {"text": "It would be great if we could export the data to CSV format.", "label": "feature_request"},
    {"text": "How do I configure the database connection string?", "label": "question"},
    {"text": "TypeError: 'NoneType' object is not iterable on line 55", "label": "bug_report"},
    # ... more classification examples
]

# --- 5. Creative Writing (Code Generation/Explanation) ---
code_creative_writing_data = [
    {
        "prompt": "Write a simple Python function to check if a number is even.",
        "generated_text": """
def is_even(number):
    \"\"\"Checks if a given integer is even.\"\"\"
    return number % 2 == 0
"""
    },
    {
        "prompt": "Explain the difference between a list and a tuple in Python.",
        "generated_text": "In Python, both lists and tuples are used to store collections of items. The main difference is mutability: lists are mutable (can be changed after creation, e.g., adding/removing elements), while tuples are immutable (cannot be changed after creation). Lists use square brackets `[]` and tuples use parentheses `()`."
    },
    # ... more creative writing examples
]

# --- 6. Brainstorming (Algorithm Ideas) ---
code_brainstorming_data = [
    {
        "topic": "Ideas for optimizing a web scraping process",
        "ideas": [
            "Use asynchronous requests (e.g., asyncio with aiohttp)",
            "Implement caching for frequently accessed pages",
            "Use efficient parsing libraries (e.g., lxml)",
            "Distribute scraping tasks across multiple workers/machines",
            "Respect robots.txt and add delays to avoid overloading the server"
        ]
    },
    # ... more brainstorming examples
]

# --- 7. Multiple Choice QA (Language Concepts) ---
code_mcq_data = [
    {
        "context": "In Python, the `pass` statement is a null operation — when it is executed, nothing happens. It is useful as a placeholder when a statement is required syntactically, but no code needs to be executed.",
        "question": "What does the `pass` statement do in Python?",
        "choices": ["Exits the loop", "Raises an error", "Does nothing", "Prints 'pass'"],
        "answer_index": 2 # Index of "Does nothing"
    },
    # ... more multiple choice QA examples
]

# Combine all data
code_domain_data = {
    "summarization": code_summarization_data,
    "open_qa": code_open_qa_data,
    "close_qa": code_close_qa_data,
    "classification": code_classification_data,
    "creative_writing": code_creative_writing_data,
    "brainstorming": code_brainstorming_data,
    "multiple_choice_qa": code_mcq_data,
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
    print("--- Code Domain Dataset Samples ---")
    print(f"\nSummarization Example:\n{code_domain_data['summarization'][0]}")
    print(f"\nOpen QA Example:\n{code_domain_data['open_qa'][0]}")
    print(f"\nClose QA Example:\n{code_domain_data['close_qa'][0]}")
    print(f"\nClassification Example:\n{code_domain_data['classification'][0]}")
    print(f"\nCreative Writing Example:\n{code_domain_data['creative_writing'][0]}")
    print(f"\nBrainstorming Example:\n{code_domain_data['brainstorming'][0]}")
    print(f"\nMultiple Choice QA Example:\n{code_domain_data['multiple_choice_qa'][0]}")

    # --- Generate Parquet File ---
    print("\n--- Generating Parquet file ---")
    all_records = []
    domain_name = "code"
    for task, data in code_domain_data.items():
        all_records.extend(preprocess_data_for_parquet(domain_name, task, data))

    if all_records:
        df = pd.DataFrame(all_records)
        output_filename = "code_data.parquet"
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
