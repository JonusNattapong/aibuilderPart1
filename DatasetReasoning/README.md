# Reasoning Dataset Generator

Interactive tool for generating reasoning datasets using large language models with Streamlit interface.

## Features

- Multiple reasoning approaches supported:
  - Chain of Thought (CoT)
  - ReAct (Reasoning + Acting)
  - Tree of Thought (ToT)
  - Meta Reasoning

- Interactive interface:
  - Direct prompt input or file upload
  - Batch processing
  - Real-time results preview
  - Step-by-step editing
  - Multiple export formats

- Advanced reasoning capabilities:
  - Step-by-step reasoning
  - Multiple reasoning paths
  - Strategy selection
  - Self-monitoring
  - Confidence evaluation

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd DatasetReasoning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download language models:
```bash
python -m spacy download en_core_web_sm
```

## Usage

1. Start the Streamlit interface:
```bash
streamlit run streamlit_app.py
```

2. Using the interface:
   - Select reasoning approach
   - Choose model
   - Configure parameters
   - Input prompts
   - Process reasoning
   - Review and edit steps
   - Export dataset

## Supported Approaches

### Chain of Thought (CoT)
- Models:
  - FLAN-T5
  - GPT-Neo
- Parameters:
  - Maximum tokens
  - Temperature

### ReAct (Reasoning + Acting)
- Models:
  - FLAN-T5
  - GPT-Neo
  - OPT
- Parameters:
  - Maximum tokens
  - Temperature

### Tree of Thought (ToT)
- Models:
  - GPT-Neo
  - OPT
  - FLAN-T5
- Parameters:
  - Maximum tokens
  - Temperature
  - Number of branches

### Meta Reasoning
- Models:
  - GPT-Neo
  - FLAN-T5
  - OPT
- Parameters:
  - Maximum tokens
  - Temperature

## Output Formats

### JSON
```json
{
  "task": "cot",
  "model": "google/flan-t5-large",
  "prompt": "Solve this problem...",
  "reasoning_steps": [
    "First, we need to...",
    "Then, considering...",
    "Finally, we can..."
  ],
  "answer": "The solution is..."
}
```

### CSV
Contains flattened version of JSON output for easy analysis.

### JSONL
Line-by-line JSON format for large datasets.

## Configuration

Edit `config.py` to modify:
- Model settings
- Task parameters
- Template prompts
- Cache settings
- Output paths

## Directory Structure

```
DatasetReasoning/
├── streamlit_app.py     # Streamlit interface
├── reasoning_utils.py   # Reasoning utilities
├── config.py           # Configuration settings
├── requirements.txt    # Dependencies
├── output/            # Generated datasets
├── .model_cache/      # Cached models
└── logs/             # Application logs
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.0+
- CUDA (optional, for GPU support)
- See requirements.txt for complete list

## GPU Support

Enable GPU usage by setting environment variable:
```bash
export USE_GPU=1
```

## Memory Optimization

For large models:
1. Use FP16 precision:
```bash
export USE_FP16=1
```

2. Adjust batch size and number of branches
3. Enable gradient checkpointing for large models

## Contributing

Feel free to submit issues and pull requests for:
- New reasoning approaches
- Additional models
- Template improvements
- Interface enhancements
- Bug fixes

## License

MIT License