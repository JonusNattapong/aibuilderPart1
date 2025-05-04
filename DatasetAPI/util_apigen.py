"""
Utility functions for API generation dataset.
Provides shared functionality across the APIGen pipeline.
"""
import os
import json
import pandas as pd
import time
import logging
import re # Added for regex checks
from typing import List, Dict, Any, Tuple, Union, Optional
import matplotlib.pyplot as plt
import numpy as np

# --- Logging Configuration ---
def setup_logger(log_file: str = None, level=logging.INFO) -> logging.Logger:
    """Setup and return a logger with proper formatting."""
    # Create logger 
    logger = logging.getLogger('apigen')
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# --- Data Loading and Saving ---
def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSON Lines file."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    data.append(json.loads(line))
        return data
    except Exception as e:
        logging.error(f"Error loading JSONL file {file_path}: {e}")
        return []

def save_jsonl(data: List[Dict], file_path: str) -> bool:
    """Save data to a JSON Lines file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        return True
    except Exception as e:
        logging.error(f"Error saving JSONL file {file_path}: {e}")
        return False

def save_csv(data: List[Dict], file_path: str, columns: List[str] = None) -> bool:
    """Save data to a CSV file with optional column selection."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df = pd.DataFrame(data)
        if columns:
            df = df[columns]
        df.to_csv(file_path, index=False, encoding='utf-8')
        return True
    except Exception as e:
        logging.error(f"Error saving CSV file {file_path}: {e}")
        return False

def convert_to_simplified_format(data: List[Dict]) -> List[Dict]:
    """Convert the full execution results format to a simplified format for non-technical users."""
    simplified = []
    for item in data:
        query = item.get("query", "")
        results = item.get("execution_results", [])
        
        # Extract only successful calls and their outputs
        simplified_calls = []
        for result in results:
            if result.get("execution_success", False):
                call = result.get("call", {})
                simplified_calls.append({
                    "api": call.get("name", ""),
                    "parameters": call.get("arguments", {}),
                    "response": result.get("execution_output", {})
                })
        
        simplified.append({
            "query": query,
            "api_calls": simplified_calls
        })
    
    return simplified

# --- API Definition Loading ---
def load_api_definitions(filepath: str) -> Optional[Dict]:
    """Load API definitions from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            definitions = json.load(f)
        return definitions
    except Exception as e:
        logging.error(f"Error loading API definitions from {filepath}: {e}")
        return None

# --- Argument Plausibility Check ---
def check_argument_plausibility(api_name: str, arguments: Dict, api_library: Dict) -> Tuple[bool, List[str]]:
    """
    Performs basic plausibility checks on argument values based on rules in API_LIBRARY.
    Returns (is_plausible, list_of_warnings_or_errors).
    """
    if api_name not in api_library:
        return True, [] # Cannot check if API is unknown

    api_def = api_library[api_name]
    plausibility_rules = api_def.get("plausibility_checks", {})
    issues = []
    is_plausible = True

    for arg_name, value in arguments.items():
        if arg_name in plausibility_rules:
            rules = plausibility_rules[arg_name]

            # Regex check
            if "regex" in rules:
                if not isinstance(value, str) or not re.match(rules["regex"], value):
                    issues.append(f"Argument '{arg_name}' value '{value}' does not match regex '{rules['regex']}'")
                    is_plausible = False # Regex mismatch is usually an error

            # Min value check
            if "min_value" in rules:
                try:
                    if float(value) < rules["min_value"]:
                        issues.append(f"Argument '{arg_name}' value {value} is less than minimum {rules['min_value']}")
                        # Consider this a warning unless it's clearly invalid (like negative amount)
                except (ValueError, TypeError):
                    issues.append(f"Argument '{arg_name}' value '{value}' could not be compared to min_value {rules['min_value']}")

            # Max value check
            if "max_value" in rules:
                 try:
                    if float(value) > rules["max_value"]:
                        issues.append(f"Argument '{arg_name}' value {value} is greater than maximum {rules['max_value']}")
                 except (ValueError, TypeError):
                    issues.append(f"Argument '{arg_name}' value '{value}' could not be compared to max_value {rules['max_value']}")

            # Allowed values check
            if "allowed_values" in rules:
                if value not in rules["allowed_values"]:
                    issues.append(f"Argument '{arg_name}' value '{value}' is not in allowed values: {rules['allowed_values']}")
                    is_plausible = False # Not allowed is an error

    return is_plausible, issues


# --- Semantic Similarity Check (Placeholder) ---
def check_semantic_similarity(query: str, api_description: str, model_name: str) -> float:
    """
    Placeholder function for checking semantic similarity using embeddings.
    Requires sentence-transformers library and a suitable model.
    Returns a similarity score (e.g., 0.0 to 1.0).
    """
    # Implementation requires:
    # 1. pip install sentence-transformers
    # 2. Loading the model: model = SentenceTransformer(model_name)
    # 3. Getting embeddings: embeddings = model.encode([query, api_description])
    # 4. Calculating cosine similarity: score = util.cos_sim(embeddings[0], embeddings[1])
    logging.warning("Semantic similarity check is not fully implemented.")
    # Simulate a basic check based on word overlap for now
    query_words = set(query.lower().split())
    desc_words = set(api_description.lower().split())
    overlap = len(query_words.intersection(desc_words))
    union = len(query_words.union(desc_words))
    return overlap / union if union > 0 else 0.0


# --- Statistics and Analytics ---
def generate_statistics(data: List[Dict]) -> Dict:
    """Generate statistics about the generated API calls dataset."""
    if not data:
        return {"error": "No data to analyze"}
    
    stats = {
        "total_samples": len(data),
        "api_call_counts": {},
        "query_lengths": {
            "min": float('inf'),
            "max": 0,
            "avg": 0,
            "distribution": {}
        },
        "calls_per_query": {
            "min": float('inf'),
            "max": 0,
            "avg": 0,
            "distribution": {}
        },
        "needs_review_count": 0, # Added stat
        "negative_sample_count": 0 # Added stat
    }
    
    # Collect data for stats
    total_query_length = 0
    total_calls = 0
    for item in data:
        # Check for needs_review flag
        if item.get("needs_review", False):
            stats["needs_review_count"] += 1

        # Check if it's a negative sample (assuming a flag or specific structure)
        # Let's assume negative samples have an empty 'execution_results' list
        # AND potentially a flag like 'is_negative_sample': True
        is_negative = item.get("is_negative_sample", False) or \
                      (not item.get("execution_results") and not item.get("answer")) # Heuristic

        if is_negative:
             stats["negative_sample_count"] += 1

        query = item.get("query", "")
        results = item.get("execution_results", [])
        
        # Query length stats
        query_length = len(query)
        total_query_length += query_length
        stats["query_lengths"]["min"] = min(stats["query_lengths"]["min"], query_length)
        stats["query_lengths"]["max"] = max(stats["query_lengths"]["max"], query_length)
        
        # Bucket query lengths by 10s (0-10, 11-20, etc)
        length_bucket = (query_length // 10) * 10
        bucket_key = f"{length_bucket}-{length_bucket+9}"
        stats["query_lengths"]["distribution"][bucket_key] = stats["query_lengths"]["distribution"].get(bucket_key, 0) + 1
        
        # Calls per query stats
        num_calls = len(results)
        total_calls += num_calls
        if num_calls > 0:  # Avoid counting 0 calls for min
            stats["calls_per_query"]["min"] = min(stats["calls_per_query"]["min"], num_calls)
        stats["calls_per_query"]["max"] = max(stats["calls_per_query"]["max"], num_calls)
        
        # Distribution of number of calls
        stats["calls_per_query"]["distribution"][num_calls] = stats["calls_per_query"]["distribution"].get(num_calls, 0) + 1
        
        # API call types
        for result in results:
            if result.get("execution_success", False):
                call = result.get("call", {})
                api_name = call.get("name", "unknown")
                stats["api_call_counts"][api_name] = stats["api_call_counts"].get(api_name, 0) + 1
    
    # Calculate averages
    if stats["total_samples"] > 0:
        stats["query_lengths"]["avg"] = total_query_length / stats["total_samples"]
        stats["calls_per_query"]["avg"] = total_calls / stats["total_samples"]
    
    # Handle case when no data had min/max values
    if stats["query_lengths"]["min"] == float('inf'):
        stats["query_lengths"]["min"] = 0
    if stats["calls_per_query"]["min"] == float('inf'):
        stats["calls_per_query"]["min"] = 0
    
    return stats

def visualize_statistics(stats: Dict, output_dir: str) -> None:
    """Generate visualization charts for the dataset statistics."""
    if not stats or "error" in stats:
        print("No statistics to visualize")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Chart 1: API Call Distribution (Pie Chart)
    if stats["api_call_counts"]:
        plt.figure(figsize=(10, 6))
        labels = list(stats["api_call_counts"].keys())
        sizes = list(stats["api_call_counts"].values())
        
        # Sort by frequency (optional)
        sorted_data = sorted(zip(labels, sizes), key=lambda x: x[1], reverse=True)
        labels = [x[0] for x in sorted_data]
        sizes = [x[1] for x in sorted_data]
        
        # If there are too many APIs, group the less frequent ones
        if len(labels) > 10:
            top_n = 9  # Show top 9 + "Other"
            other_sum = sum(sizes[top_n:])
            labels = labels[:top_n] + ["Other"]
            sizes = sizes[:top_n] + [other_sum]
        
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Distribution of API Calls')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'api_call_distribution.png'))
        plt.close()
    
    # Chart 2: Query Length Distribution (Histogram)
    if stats["query_lengths"]["distribution"]:
        plt.figure(figsize=(10, 6))
        buckets = list(stats["query_lengths"]["distribution"].keys())
        counts = list(stats["query_lengths"]["distribution"].values())
        
        # Sort buckets by their numeric value
        sorted_data = sorted(zip(buckets, counts), key=lambda x: int(x[0].split('-')[0]))
        buckets = [x[0] for x in sorted_data]
        counts = [x[1] for x in sorted_data]
        
        plt.bar(buckets, counts)
        plt.title('Query Length Distribution')
        plt.xlabel('Character Length Range')
        plt.ylabel('Number of Queries')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'query_length_distribution.png'))
        plt.close()
    
    # Chart 3: Calls Per Query Distribution (Bar Chart)
    if stats["calls_per_query"]["distribution"]:
        plt.figure(figsize=(10, 6))
        num_calls = list(stats["calls_per_query"]["distribution"].keys())
        counts = list(stats["calls_per_query"]["distribution"].values())
        
        # Sort by number of calls
        sorted_data = sorted(zip(num_calls, counts), key=lambda x: x[0])
        num_calls = [str(x[0]) for x in sorted_data]
        counts = [x[1] for x in sorted_data]
        
        plt.bar(num_calls, counts)
        plt.title('API Calls Per Query Distribution')
        plt.xlabel('Number of API Calls')
        plt.ylabel('Number of Queries')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'calls_per_query_distribution.png'))
        plt.close()
        
    print(f"Statistics visualizations saved to {output_dir}")

# --- Cache System ---
class LLMResponseCache:
    """Simple cache for LLM responses to avoid redundant API calls."""
    
    def __init__(self, cache_file: str = None, max_size: int = 1000):
        """Initialize the cache with optional persistence."""
        self.cache = {}
        self.hits = 0
        self.misses = 0
        self.cache_file = cache_file
        self.max_size = max_size
        
        # Load cache from file if provided
        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                print(f"Loaded {len(self.cache)} cached responses from {cache_file}")
            except Exception as e:
                print(f"Error loading cache from {cache_file}: {e}")
                self.cache = {}
    
    def get(self, prompt: str) -> Optional[str]:
        """Get a cached response for a prompt."""
        if prompt in self.cache:
            self.hits += 1
            return self.cache[prompt]
        
        self.misses += 1
        return None
    
    def set(self, prompt: str, response: str) -> None:
        """Store a response in the cache."""
        # Simple LRU: if cache exceeds max size, remove oldest entries
        if len(self.cache) >= self.max_size:
            # Remove 10% of oldest entries
            num_to_remove = max(1, int(self.max_size * 0.1))
            for _ in range(num_to_remove):
                if self.cache:
                    self.cache.pop(next(iter(self.cache)))
        
        self.cache[prompt] = response
        
        # Persist cache if file is specified
        if self.cache_file:
            try:
                os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.cache, f, ensure_ascii=False)
            except Exception as e:
                print(f"Error saving cache to {self.cache_file}: {e}")
    
    def get_stats(self) -> Dict:
        """Get cache performance statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "total_requests": total
        }
