# NLP Dataset Generator

Interactive tool for generating natural language processing datasets using pre-trained models with Streamlit interface.

## Features

- Multiple NLP tasks supported:
  - Text Classification
  - Token Classification (NER)
  - Text Generation

- Interactive interface:
  - Direct text input or file upload
  - Batch processing
  - Real-time results preview
  - Result editing and validation
  - Multiple export formats

- Multilingual support:
  - English
  - Thai
  - Japanese
  - Korean
  - Chinese
  - Vietnamese

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd DatasetNLP
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download spaCy models:
```bash
python -m spacy download en_core_web_sm
python -m spacy download th_core_news_lg
```

## Usage

1. Start the Streamlit interface:
```bash
streamlit run streamlit_app.py
```

2. Using the interface:
   - Select NLP task
   - Choose model
   - Configure task parameters
   - Input text or upload file
   - Process text
   - Review and edit results
   - Export dataset

## Supported Tasks

### Text Classification
- Models:
  - BERT Multilingual
  - XLM-RoBERTa
  - Muppet RoBERTa
- Parameters:
  - Top K predictions
  - Batch size

### Token Classification (NER)
- Models:
  - BERT NER
  - XLM-RoBERTa NER
  - DistilBERT NER
- Parameters:
  - Confidence threshold
  - Batch size

### Text Generation
- Models:
  - GPT-2
  - GPT-Neo
  - OPT
- Parameters:
  - Maximum new tokens
  - Temperature
  - Include prompt option

## Output Formats

### JSON
```json
{
  "task": "text_classification",
  "model": "bert-base-multilingual-cased",
  "language": "EN",
  "config": {
    "top_k": 5
  },
  "input": "Example text",
  "result": {
    "labels": [
      {
        "label": "positive",
        "confidence": 0.95
      }
    ]
  }
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
- Language options
- Cache settings
- Output paths
- Visualization options

## Directory Structure

```
DatasetNLP/
├── streamlit_app.py    # Streamlit interface
├── nlp_utils.py       # NLP processing utilities
├── config.py         # Configuration settings
├── requirements.txt  # Dependencies
├── output/          # Generated datasets
├── .model_cache/    # Cached models
└── logs/           # Application logs
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.0+
- spaCy 3.0+
- CUDA (optional, for GPU support)
- See requirements.txt for complete list

## GPU Support

Enable GPU usage by setting environment variable:
```bash
export USE_GPU=1
```

## Memory Optimization

For large models or datasets:
1. Use FP16 precision:
```bash
export USE_FP16=1
```

2. Adjust batch size in task parameters
3. Enable gradient checkpointing for large models

## Contributing

Feel free to submit issues and pull requests for:
- New NLP tasks
- Additional models
- Language support
- Interface improvements
- Bug fixes

## License

MIT License