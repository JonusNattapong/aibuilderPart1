# Set this flag to True to use PyThaiNLP for all translations (much faster, less accurate for medical context)
USE_PYTHAINLP_ONLY = True
import os
import re
import sys
import gc
import torch
from tqdm import tqdm
from contextlib import nullcontext
from datasets import load_from_disk

# Install required packages
print("Installing dependencies...")
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain", "pythainlp", "python-crfsuite", "tqdm", "transformers"])

from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pythainlp import word_tokenize
from pythainlp.util import normalize
from transformers.pipelines import pipeline
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from medical_terms import (
    replace_all_terms,
    escape_regex_chars,
    MEDICAL_TERMS,
    ANATOMICAL_TERMS,
    SYMPTOM_TERMS,
    COMPOUND_ANATOMICAL
)

# Define paths using raw strings
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_PATH, 'DatasetDownload', 'FreedomIntelligence_medical-o1-reasoning-SFT')
MODEL_PATH = os.path.join(BASE_PATH, 'ModelUse', 'nllb-200-3.3B')
SOURCE_LANG = "English"
TARGET_LANG = "Thai"

# Typhoon-7B-Instruct prompt template for medical translation
PROMPT_TEMPLATE = """You are a professional medical translator. Translate the following English medical text to Thai with high accuracy, preserving all medical terminology and context.

English:
{input}

Thai:"""

def load_langchain_pipeline():
    """Load the Typhoon-7B-Instruct model and create a LangChain pipeline."""
    print("Loading Typhoon-7B-Instruct model for LangChain...")
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )

        hf_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            device=0 if torch.cuda.is_available() else -1
        )

        llm = HuggingFacePipeline(pipeline=hf_pipe)
        prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["input"])
        chain = LLMChain(llm=llm, prompt=prompt)
        print("LangChain pipeline loaded successfully.")
        return chain
    except Exception as e:
        print(f"Error loading LangChain pipeline: {e}")
        return None

def clean_thai_text(text: str) -> str:
    """Clean and normalize Thai text using PyThaiNLP."""
    if not text or not text.strip():
        return ""
    try:
        # Remove translation artifacts
        text = re.sub(r'แปล.*?:|Translate.*?:|Thai:', '', text)
        # Basic Thai text cleaning
        text = normalize(text)
        tokens = word_tokenize(text)
        text = ' '.join(tokens) if tokens else text
        # Clean up spacing
        text = re.sub(r'\s*\n\s*', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([.,!?])\s*', r'\1 ', text)
        return text.strip()
    except Exception as e:
        print(f"Thai text processing failed: {e}")
        return text.strip()

def translate_with_langchain(text: str, chain) -> str:
    """Translate text using LangChain and Typhoon-7B-Instruct prompt."""
    if not text or not text.strip():
        return ""
    try:
        # Split text into manageable chunks
        sentences = re.split(r'([.!?]+)', text.strip())
        max_chunk_size = 80
        chunks = []
        current_chunk = ""
        for i in range(0, len(sentences), 2):
            sent = sentences[i].strip()
            if not sent:
                continue
            if i + 1 < len(sentences):
                sent += sentences[i + 1]
            if len(current_chunk) + len(sent) < max_chunk_size:
                current_chunk += sent
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sent
        if current_chunk:
            chunks.append(current_chunk)
        translated_parts = []
        for chunk in chunks:
            if not chunk.strip():
                continue
            try:
                result = chain.run(input=chunk)
                # Extract only the Thai translation (after "Thai:")
                if "Thai:" in result:
                    translation = result.split("Thai:")[-1].strip()
                else:
                    translation = result.strip()
                translation = clean_thai_text(translation)
            except Exception as e:
                print(f"Translation error: {e}")
                translation = "เกิดข้อผิดพลาดในการแปล"
            # Apply medical term replacements
            translation = replace_all_terms(translation)
            translated_parts.append(translation)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return ' '.join(translated_parts)
    except Exception as e:
        print(f"Translation error: {e}")
        return "เกิดข้อผิดพลาดในการแปล"

def main():
    # --- PyThaiNLP translation utility ---
    try:
        from pythainlp.translate import Translate
        pythainlp_translator = Translate(src_lang="en", target_lang="th")
    except Exception as e:
        pythainlp_translator = None

    def translate_with_pythainlp(text: str) -> str:
        """Translate English to Thai using PyThaiNLP's built-in translation."""
        if not pythainlp_translator:
            return "PyThaiNLP translation not available"
        try:
            return pythainlp_translator.translate(text)
        except Exception as e:
            return f"PyThaiNLP translation error: {e}"

    # Load LangChain pipeline
    chain = load_langchain_pipeline()
    if not chain:
        print("Failed to load LangChain pipeline. Exiting.")
        return

    print("Loading dataset...")
    try:
        dataset = load_from_disk(DATASET_PATH)['train']
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("\nTranslating 5 examples:")
    for idx in tqdm(range(5), desc="Translating examples"):
        try:
            item = dataset[idx]
            print(f"\n=== Example {idx + 1} ===")

            # Process question
            question = item["Question"]
            print("\nQuestion (EN):")
            print(question)

            # Highlight medical terms
            highlighted_question = question
            all_terms = {**COMPOUND_ANATOMICAL, **ANATOMICAL_TERMS, **SYMPTOM_TERMS, **MEDICAL_TERMS}
            for term in sorted(all_terms.keys(), key=len, reverse=True):
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                if pattern.search(highlighted_question):
                    highlighted_question = pattern.sub(f"[{term}]", highlighted_question)
            print("\nQuestion (EN) with highlighted terms:")
            print(highlighted_question)

            # Translate question
            print("\nQuestion (TH):")
            if USE_PYTHAINLP_ONLY:
                question_th = translate_with_pythainlp(question)
            else:
                question_th = translate_with_langchain(question, chain)
            if question_th:
                print(question_th)
            else:
                print("Translation failed")

            # Process response
            response = item["Response"]
            print("\nResponse (EN):")
            print(response)

            # Highlight medical terms in response
            highlighted_response = response
            for term in sorted(all_terms.keys(), key=len, reverse=True):
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                if pattern.search(highlighted_response):
                    highlighted_response = pattern.sub(f"[{term}]", highlighted_response)
            print("\nResponse (EN) with highlighted terms:")
            print(highlighted_response)

            # Translate response
            print("\nResponse (TH):")
            if USE_PYTHAINLP_ONLY:
                response_th = translate_with_pythainlp(response)
            else:
                response_th = translate_with_langchain(response, chain)
            if response_th:
                print(response_th)
            else:
                print("Translation failed")

            print("\n" + "="*50)

        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            continue

if __name__ == "__main__":
    main()
# (Removed duplicate PyThaiNLP translation utility at end of file)