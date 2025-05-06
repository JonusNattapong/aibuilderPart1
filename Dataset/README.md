# Dataset Generator

Interactive tool for generating CSV and Parquet datasets with customizable schemas and realistic data using Streamlit interface.

## Features

- Multiple dataset formats supported:
  - CSV
  - Parquet

- Pre-configured dataset types:
  - CSV:
    - Question-Answer Dataset
    - User Profiles Dataset
    - Product Reviews Dataset
  - Parquet:
    - Sales Transaction Dataset
    - IoT Sensor Dataset
    - Application Logs Dataset

- Interactive interface:
  - Dataset preview
  - Schema customization
  - Data validation
  - Statistical analysis
  - Visualization tools

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Dataset
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit interface:
```bash
streamlit run streamlit_app.py
```

2. Using the interface:
   - Select dataset format (CSV/Parquet)
   - Choose dataset type
   - Configure parameters
   - Generate data
   - Review and validate
   - Export dataset

## Dataset Types

### CSV Datasets

#### Question-Answer Dataset
- Fields:
  - question (string)
  - answer (string)
  - category (categorical)
  - difficulty (numeric)
  - timestamp (datetime)

#### User Profiles Dataset
- Fields:
  - name (string)
  - email (string)
  - age (numeric)
  - country (string)
  - joined_date (datetime)

#### Product Reviews Dataset
- Fields:
  - product_id (string)
  - user_id (string)
  - rating (numeric)
  - review_text (string)
  - review_date (datetime)

### Parquet Datasets

#### Sales Transaction Dataset
- Fields:
  - transaction_id (string)
  - customer_id (string)
  - products (list)
  - total_amount (numeric)
  - payment_method (categorical)
  - transaction_date (datetime)

#### IoT Sensor Dataset
- Fields:
  - device_id (string)
  - temperature (numeric)
  - humidity (numeric)
  - pressure (numeric)
  - timestamp (datetime)

#### Application Logs Dataset
- Fields:
  - event_id (string)
  - service_name (categorical)
  - level (categorical)
  - message (string)
  - timestamp (datetime)

## Configuration

Edit `config.py` to modify:
- Dataset schemas
- Data generators
- Output settings
- Validation rules
- Visualization options

## Directory Structure

```
Dataset/
├── streamlit_app.py    # Streamlit interface
├── dataset_utils.py    # Data generation utilities
├── config.py          # Configuration settings
├── requirements.txt   # Dependencies
├── output/           # Generated datasets
└── logs/            # Application logs
```

## Data Generation

- Uses Faker for realistic data
- Configurable batch sizes
- Progress tracking
- Validation checks

## Export Formats

### CSV
- Configurable delimiter
- Quote handling options
- UTF-8 encoding

### Parquet
- Compression options (snappy, gzip)
- Row group size configuration
- Column encoding options

## Customization

Add new dataset types by:
1. Define schema in config.py
2. Add data generation logic in dataset_utils.py
3. Update UI elements in streamlit_app.py

## Data Validation

Validates:
- Data types
- Value ranges
- Category membership
- Required fields
- Format constraints

## Visualization

Built-in visualizations for:
- Data distributions
- Category frequencies
- Time series trends
- Correlations
- Missing data patterns

## Contributing

Feel free to submit issues and pull requests for:
- New dataset types
- Additional data generators
- Visualization improvements
- Bug fixes

## License

MIT License