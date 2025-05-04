# DatasetAPI Documentation

## Overview
DatasetAPI is a system for generating and validating API call datasets using LLM (Language Learning Model). It simulates API interactions and generates high-quality training data for function calling models.

## Architecture

### Core Components

1. Configuration (`config_apigen_llm.py`)
   - API definitions and parameters
   - LLM generation settings
   - Output and validation configurations
   - Feature flags for different components

2. API Simulation (`api_execution_sim.py`)
   - Simulated API executors for different categories
   - Realistic response generation
   - Error simulation and handling

3. Pipeline (`run_apigen_simulation.py`)
   - LLM data generation
   - Multi-stage validation
   - Output processing and statistics

4. Utilities (`util_apigen.py`)
   - Data loading/saving helpers
   - Format conversion
   - Statistics generation
   - Visualization tools

## Supported APIs

### Finance APIs
- `get_stock_price`: Get current stock price
  - Parameters: ticker_symbol (required)
- `transfer_money`: Transfer funds between accounts
  - Parameters: from_account_id, to_account_id, amount (required), currency (optional)
- `get_account_balance`: Check account balance
  - Parameters: account_id (required), include_pending (optional)
- `get_transaction_history`: Get transaction records
  - Parameters: account_id (required), start_date, end_date, max_transactions (optional)

### Health APIs
- `find_nearby_doctors`: Locate healthcare providers
  - Parameters: location (required), specialty, radius_km (optional)
- `book_medical_appointment`: Schedule medical visits
  - Parameters: doctor_id, appointment_date, appointment_time, patient_name (required), reason (optional)

### Tool APIs
- `set_timer`: Create countdown timer
  - Parameters: duration (required), label (optional)
- `calculate`: Perform calculations
  - Parameters: expression (required)
- `set_reminder`: Schedule reminders
  - Parameters: message, time (required), priority (optional)

### Thai-specific APIs
- `translate_th_en`: Thai to English translation
  - Parameters: text (required), formal (optional)
- `find_thai_food`: Find Thai restaurants/dishes
  - Parameters: location (required), dish_type, spice_level, max_price (optional)

## Validation Pipeline

1. **Format Check**
   - JSON structure validation
   - Type checking and conversion
   - Required parameter validation
   - Plausibility checks on values

2. **Execution Check**
   - API simulation execution
   - Error handling and logging
   - Success/failure tracking

3. **Semantic Check**
   - Keyword matching
   - Query-API semantic similarity
   - Context validation

## Output Formats

1. **JSONL** (Raw Data)
```json
{
  "query": "User query text",
  "execution_results": [
    {
      "call": {
        "name": "api_name",
        "arguments": {}
      },
      "execution_success": true,
      "execution_output": {}
    }
  ]
}
```

2. **CSV** (Simplified View)
- query
- api_name
- arguments
- success
- output

3. **Simplified JSONL**
```json
{
  "query": "User query text",
  "api_calls": [
    {
      "api": "api_name",
      "parameters": {},
      "response": {}
    }
  ]
}
```

## Usage

### Installation
```bash
# Clone repository
git clone <repository-url>
cd aibuilderPart1

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your DEEPSEEK_API_KEY
```

### Configuration
Edit `config_apigen_llm.py` to customize:
- Number of examples to generate
- Generation parameters
- Validation settings
- Output formats
- Feature flags

### Running

There are two ways to run the system:

1. Command Line Interface:
```bash
python DatasetAPI/run_apigen_simulation.py
```

2. Streamlit Web Interface:
```bash
streamlit run DatasetAPI/streamlit_app.py
```

The Streamlit interface provides:
- Interactive configuration of all parameters
- Real-time generation monitoring
- Results viewer with filtering and sorting
- Interactive visualizations and statistics
- Easy API key management

#### Streamlit Interface Features

1. **Configuration Panel (Sidebar)**
   - API key input
   - Number of examples to generate
   - Temperature setting
   - Language options (Thai/English)
   - Validation settings
   - Output format options

2. **Generator Tab**
   - Available APIs overview
   - Generation progress monitoring
   - Status updates
   - Error reporting

3. **Results Tab**
   - Generated data table
   - Filtering and sorting capabilities
   - JSON prettification
   - Success/failure tracking

4. **Visualization Tab**
   - API call distribution charts
   - Query length analysis
   - Success rate metrics
   - Interactive Plotly graphs

### Output Files
```
DataOutput/
└── apigen_llm_diverse/
    ├── verified_api_calls_llm_diverse.jsonl
    ├── verified_api_calls_llm_diverse.csv
    ├── simplified_api_calls_llm_diverse.jsonl
    ├── api_call_statistics.json
    ├── logs/
    │   └── apigen_llm.log
    └── visualizations/
        ├── api_call_distribution.png
        ├── query_length_distribution.png
        └── calls_per_query_distribution.png
```

## Quality Control

1. **Needs Review Flagging**
   - Type conversion cases
   - Low semantic similarity
   - Plausibility check warnings

2. **Statistics**
   - Success/failure rates
   - API usage distribution
   - Query length analysis
   - Error type tracking

3. **Visualizations**
   - API call distribution
   - Query length distribution
   - Success rate trends

## Error Handling

1. **LLM Generation**
   - API timeout/errors
   - Invalid response format
   - Retry mechanism

2. **Validation**
   - Type mismatches
   - Missing parameters
   - Invalid values
   - Semantic inconsistencies

3. **Output**
   - File write errors
   - Format conversion errors
   - Statistics generation errors

## Performance Considerations

1. **LLM Generation**
   - Configurable batch size
   - Response caching
   - Retry delays

2. **Validation**
   - Configurable validation stages
   - Early failure detection
   - Parallel processing support

3. **Output**
   - Streaming file writes
   - Memory-efficient processing
   - Configurable output formats

## Extensibility

1. **Adding New APIs**
   - Define in `config_apigen_llm.py`
   - Implement simulator in `api_execution_sim.py`
   - Add semantic keywords

2. **Custom Validation**
   - Extend format checker
   - Add plausibility rules
   - Enhance semantic checking

3. **New Output Formats**
   - Implement format converter
   - Add validation rules
   - Update statistics tracking