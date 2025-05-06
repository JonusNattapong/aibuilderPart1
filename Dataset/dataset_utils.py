"""
Utility functions for dataset generation and handling.
"""
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Callable
from faker import Faker
from datetime import datetime, timedelta
import pyarrow as pa
import pyarrow.parquet as pq
from config import (
    CSV_CONFIG, PARQUET_CONFIG,
    SUPPORTED_FORMATS, SUPPORTED_TASKS
)

class DatasetManager:
    """Manages dataset generation and processing."""
    
    def __init__(self, format_type: str):
        self.format_type = format_type
        self.fake = Faker()
        self.config = CSV_CONFIG if format_type == "CSV" else PARQUET_CONFIG

    def generate_dataset(self, 
                        task: str,
                        num_rows: int,
                        task_config: Dict[str, Any],
                        progress_callback: Callable = None) -> pd.DataFrame:
        """Generate dataset based on format and task."""
        
        if self.format_type == "CSV":
            return self._generate_csv_dataset(task, num_rows, task_config, progress_callback)
        else:
            return self._generate_parquet_dataset(task, num_rows, task_config, progress_callback)

    def _generate_csv_dataset(self, task, num_rows, config, progress_callback):
        """Generate CSV dataset."""
        data = []
        schema = CSV_CONFIG["tasks"][task]["schema"]
        
        for i in range(num_rows):
            row = {}
            for col, col_config in schema.items():
                if col_config["type"] == "string":
                    if col_config.get("fake_type"):
                        row[col] = getattr(self.fake, col_config["fake_type"])()
                    else:
                        row[col] = self.fake.text(max_nb_chars=col_config.get("max_length", 100))
                
                elif col_config["type"] == "number":
                    row[col] = np.random.uniform(
                        col_config.get("min", 0),
                        col_config.get("max", 100)
                    )
                
                elif col_config["type"] == "category":
                    row[col] = np.random.choice(col_config["categories"])
                
                elif col_config["type"] == "datetime":
                    start_date = datetime.strptime(
                        col_config.get("start_date", "2020-01-01"),
                        "%Y-%m-%d"
                    )
                    end_date = datetime.strptime(
                        col_config.get("end_date", "2025-12-31"),
                        "%Y-%m-%d"
                    )
                    delta = end_date - start_date
                    random_days = np.random.randint(0, delta.days)
                    row[col] = start_date + timedelta(days=random_days)
            
            data.append(row)
            
            if progress_callback:
                progress_callback(i / num_rows)

        return pd.DataFrame(data)

    def _generate_parquet_dataset(self, task, num_rows, config, progress_callback):
        """Generate Parquet dataset."""
        data = []
        schema = PARQUET_CONFIG["tasks"][task]["schema"]
        
        for i in range(num_rows):
            row = {}
            for col, col_config in schema.items():
                if col_config["type"] == "string":
                    if col_config.get("fake_type"):
                        row[col] = getattr(self.fake, col_config["fake_type"])()
                    else:
                        row[col] = self.fake.text(max_nb_chars=col_config.get("max_length", 100))
                
                elif col_config["type"] == "number":
                    if col_config.get("is_integer", False):
                        row[col] = np.random.randint(
                            col_config.get("min", 0),
                            col_config.get("max", 100)
                        )
                    else:
                        row[col] = np.random.uniform(
                            col_config.get("min", 0),
                            col_config.get("max", 100)
                        )
                
                elif col_config["type"] == "category":
                    row[col] = np.random.choice(col_config["categories"])
                
                elif col_config["type"] == "datetime":
                    start_date = datetime.strptime(
                        col_config.get("start_date", "2020-01-01"),
                        "%Y-%m-%d"
                    )
                    end_date = datetime.strptime(
                        col_config.get("end_date", "2025-12-31"),
                        "%Y-%m-%d"
                    )
                    delta = end_date - start_date
                    random_days = np.random.randint(0, delta.days)
                    row[col] = start_date + timedelta(days=random_days)
                
                elif col_config["type"] == "list":
                    num_items = np.random.randint(
                        col_config.get("min_items", 1),
                        col_config.get("max_items", 5)
                    )
                    row[col] = [
                        self.fake.word() 
                        for _ in range(num_items)
                    ]
            
            data.append(row)
            
            if progress_callback:
                progress_callback(i / num_rows)

        return pd.DataFrame(data)

def get_supported_formats() -> List[str]:
    """Get list of supported file formats."""
    return SUPPORTED_FORMATS

def get_supported_tasks(format_type: str) -> Dict[str, Dict]:
    """Get dictionary of supported tasks for a format."""
    if format_type == "CSV":
        return CSV_CONFIG["tasks"]
    else:
        return PARQUET_CONFIG["tasks"]

def load_dataset(file_path: str) -> pd.DataFrame:
    """Load dataset from file."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.csv':
        return pd.read_csv(file_path)
    elif ext == '.parquet':
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def save_dataset(df: pd.DataFrame,
                filename: str,
                format_type: str,
                output_dir: str) -> str:
    """Save dataset in specified format."""
    os.makedirs(output_dir, exist_ok=True)
    
    base_filename = os.path.splitext(filename)[0]
    output_path = os.path.join(
        output_dir,
        f"{base_filename}.{format_type.lower()}"
    )
    
    try:
        if format_type.upper() == "CSV":
            df.to_csv(output_path, index=False)
        elif format_type.upper() == "PARQUET":
            table = pa.Table.from_pandas(df)
            pq.write_table(table, output_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        return output_path
    except Exception as e:
        raise RuntimeError(f"Error saving dataset: {str(e)}")

def validate_dataset(df: pd.DataFrame,
                    format_type: str,
                    task_config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate dataset against schema and constraints."""
    issues = []
    schema = (CSV_CONFIG if format_type == "CSV" else PARQUET_CONFIG)["tasks"][task_config["task"]]["schema"]
    
    # Check schema
    for col, config in schema.items():
        if col not in df.columns:
            issues.append(f"Missing column: {col}")
            continue
            
        # Type checking
        if config["type"] == "number":
            if not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"Column {col} should be numeric")
            else:
                if "min" in config and df[col].min() < config["min"]:
                    issues.append(f"Column {col} contains values below minimum {config['min']}")
                if "max" in config and df[col].max() > config["max"]:
                    issues.append(f"Column {col} contains values above maximum {config['max']}")
        
        elif config["type"] == "category":
            invalid_cats = set(df[col].unique()) - set(config["categories"])
            if invalid_cats:
                issues.append(f"Column {col} contains invalid categories: {invalid_cats}")
        
        elif config["type"] == "datetime":
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                issues.append(f"Column {col} should be datetime")
    
    return {
        "is_valid": len(issues) == 0,
        "issues": issues
    }