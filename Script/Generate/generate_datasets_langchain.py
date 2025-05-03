import os
import json
import time
from dotenv import load_dotenv # Added

# Load environment variables from .env file
load_dotenv() # Added

from langchain_huggingface import HuggingFaceEndpoint
# Import configuration
from config_generate import (
    MODEL_ID, OUTPUT_DIR, NUM_SAMPLES_PER_TASK,
    CLASSIFICATION_TOPICS, CLASSIFICATION_CATEGORIES,
    QA_TOPICS,
    TABLE_QA_TOPICS,
    ZERO_SHOT_TOPICS, ZERO_SHOT_POTENTIAL_LABELS,
    NER_TOPICS,
    TRANSLATION_TOPICS,
    SUMMARIZATION_TOPICS,
    SENTENCE_SIMILARITY_TOPICS,
    TEXT_GEN_TOPICS,
    STYLE_TRANSFER_TOPICS,
    FILL_MASK_TOPICS,
    TEXT_RANKING_TOPICS,
    CODE_GENERATION_TOPICS, # New import
    REASONING_COT_TOPICS    # New import
)
# Import generation functions from separate files
from gen_text_classification import generate_text_classification_data
from gen_qa import generate_qa_data
from gen_table_qa import generate_table_qa_data
from gen_zero_shot import generate_zero_shot_data
from gen_ner import generate_ner_data
from gen_translation import generate_translation_data
from gen_summarization import generate_summarization_data
from gen_sentence_similarity import generate_sentence_similarity_data
from gen_text_generation import generate_text_generation_data
from gen_style_transfer import generate_style_transfer_data
from gen_fill_mask import generate_fill_mask_data
from gen_text_ranking import generate_text_ranking_data
from gen_code_generation import generate_code_generation_data # New import
from gen_reasoning_cot import generate_reasoning_cot_data     # New import

# --- LLM Setup ---
def setup_llm(model_id=MODEL_ID, temperature=0.7, max_new_tokens=512):
    """Initializes the Hugging Face LLM."""
    try:
        # Using HuggingFaceEndpoint which is recommended for inference endpoints
        llm = HuggingFaceEndpoint(
            repo_id=model_id,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            # model_kwargs={"token": os.environ.get("HUGGINGFACEHUB_API_TOKEN")} # Pass token if needed, often handled by env var
        )
        print(f"LLM '{model_id}' initialized successfully.")
        return llm
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        print("Please ensure the HUGGINGFACEHUB_API_TOKEN environment variable is set correctly.")
        return None

# --- Prompt Templates ---
# (Removed - Now in individual gen_*.py files)

# --- Helper Functions ---
# (Removed - Now in gen_utils.py)

# --- Data Generation Functions ---
# (Removed - Now in individual gen_*.py files)


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting dataset generation process...")

    # Ensure output directory exists (relative to script execution location)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize LLM
    llm = setup_llm() # Uses MODEL_ID from config

    if llm:
        # --- Generate Text Classification Data ---
        generate_text_classification_data(llm,
                                          num_samples=NUM_SAMPLES_PER_TASK,
                                          topics=CLASSIFICATION_TOPICS,
                                          categories=CLASSIFICATION_CATEGORIES)

        # --- Generate Question Answering Data ---
        generate_qa_data(llm,
                         num_samples=NUM_SAMPLES_PER_TASK,
                         topics=QA_TOPICS)

        # --- Generate Table Question Answering Data ---
        generate_table_qa_data(llm,
                               num_samples=NUM_SAMPLES_PER_TASK,
                               topics=TABLE_QA_TOPICS)

        # --- Generate Zero-Shot Classification Data ---
        generate_zero_shot_data(llm,
                                num_samples=NUM_SAMPLES_PER_TASK,
                                topics=ZERO_SHOT_TOPICS,
                                potential_labels=ZERO_SHOT_POTENTIAL_LABELS)

        # --- Generate NER Data ---
        generate_ner_data(llm,
                          num_samples=NUM_SAMPLES_PER_TASK,
                          topics=NER_TOPICS)

        # --- Generate Translation Data ---
        generate_translation_data(llm,
                                  num_samples=NUM_SAMPLES_PER_TASK,
                                  topics=TRANSLATION_TOPICS)

        # --- Generate Summarization Data ---
        generate_summarization_data(llm,
                                    num_samples=NUM_SAMPLES_PER_TASK,
                                    topics=SUMMARIZATION_TOPICS)

        # --- Generate Sentence Similarity Data ---
        generate_sentence_similarity_data(llm,
                                          num_samples=NUM_SAMPLES_PER_TASK,
                                          topics=SENTENCE_SIMILARITY_TOPICS)

        # --- Generate Text Generation Data ---
        generate_text_generation_data(llm,
                                      num_samples=NUM_SAMPLES_PER_TASK,
                                      topics=TEXT_GEN_TOPICS)

        # --- Generate Style Transfer Data ---
        generate_style_transfer_data(llm,
                                     num_samples=NUM_SAMPLES_PER_TASK,
                                     topics=STYLE_TRANSFER_TOPICS)

        # --- Generate Fill-Mask Data ---
        generate_fill_mask_data(llm,
                                num_samples=NUM_SAMPLES_PER_TASK,
                                topics=FILL_MASK_TOPICS)

        # --- Generate Text Ranking Data ---
        generate_text_ranking_data(llm,
                                   num_samples=NUM_SAMPLES_PER_TASK,
                                   topics=TEXT_RANKING_TOPICS)

        # --- Generate Code Generation Data ---
        generate_code_generation_data(llm,
                                      num_samples=NUM_SAMPLES_PER_TASK,
                                      topics=CODE_GENERATION_TOPICS)

        # --- Generate Reasoning (CoT) Data ---
        generate_reasoning_cot_data(llm,
                                    num_samples=NUM_SAMPLES_PER_TASK,
                                    topics=REASONING_COT_TOPICS)


        print("\nDataset generation process finished.")
    else:
        print("\nDataset generation process failed due to LLM initialization error.")
