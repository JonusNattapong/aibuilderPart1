# Dataset Translation Tool

A Streamlit-based tool for translating datasets using the DeepL API with quality validation and interactive review capabilities.

## Features

- Support for CSV, JSON, and JSONL file formats
- Batch translation with progress tracking
- Language detection and validation
- Interactive translation review and editing
- Multiple export formats
- Translation quality checks

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd DatasetTranslation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up DeepL API:
   - Get an API key from [DeepL](https://www.deepl.com/pro-api)
   - Create a `.env` file:
     ```
     DEEPL_API_KEY=your-api-key-here
     ```

## Usage

1. Start the Streamlit interface:
```bash
streamlit run streamlit_app.py
```

2. Using the interface:
   - Upload your dataset file (CSV/JSON/JSONL)
   - Select source and target languages
   - Configure batch size and translation settings
   - Start translation
   - Review and edit translations
   - Export the final dataset

## Translation Quality Checks

The tool performs several validation checks:
- Length comparison between original and translated text
- Language detection verification
- Format and consistency checks
- Manual review flagging for suspicious translations

## Output Formats

- CSV: Tabular format with original and translated text
- JSON: Structured format with all metadata
- JSONL: Line-by-line JSON format for large datasets

## Directory Structure

```
DatasetTranslation/
├── streamlit_app.py      # Main Streamlit application
├── translation_utils.py   # Utility functions
├── requirements.txt      # Dependencies
├── .env                 # API key configuration
└── DataOutput/          # Generated translations
    └── translations/    # Output files
```

## Environment Variables

- `DEEPL_API_KEY`: Your DeepL API key

## Supported Languages

- English (EN)
- German (DE)
- French (FR)
- Spanish (ES)
- Italian (IT)
- Japanese (JA)
- Korean (KO)
- Chinese (ZH)
- Russian (RU)
- Portuguese (PT)
- Dutch (NL)
- Polish (PL)
- Turkish (TR)
- Thai (TH)
- Vietnamese (VI)
- Indonesian (ID)

## Error Handling

The tool includes comprehensive error handling for:
- API failures
- File format issues
- Language detection problems
- Validation failures
- Export errors

## Contributing

Feel free to submit issues and enhancement requests!