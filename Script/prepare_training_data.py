import pandas as pd
import os
from sklearn.model_selection import train_test_split
import json

def prepare_data_splits(input_file, output_dir, test_size=0.1, val_size=0.1, random_state=42):
    """
    Split data into train/validation/test sets and prepare for fine-tuning
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} samples")
    
    # First split out test set
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['label']
    )
    
    # Then split training into train/val
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size/(1-test_size),  # Adjust val_size ratio
        random_state=random_state,
        stratify=train_val_df['label']
    )
    
    print("\nData split sizes:")
    print(f"Train: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    print("\nClass distribution:")
    print("\nTrain set:")
    print(train_df['label'].value_counts())
    print("\nValidation set:")
    print(val_df['label'].value_counts())
    print("\nTest set:")
    print(test_df['label'].value_counts())
    
    # Save splits
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    # Create label mapping
    labels = sorted(df['label'].unique())
    label_map = {label: idx for idx, label in enumerate(labels)}
    
    # Save label mapping
    with open(os.path.join(output_dir, 'label_map.json'), 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    
    print(f"\nFiles saved to: {output_dir}")
    print(f"Label mapping: {label_map}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Prepare training data splits')
    parser.add_argument('--input_file', type=str, required=True,
                      help='Path to input CSV file')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory for split datasets')
    parser.add_argument('--test_size', type=float, default=0.1,
                      help='Proportion of data to use for test set')
    parser.add_argument('--val_size', type=float, default=0.1,
                      help='Proportion of data to use for validation set')
    
    args = parser.parse_args()
    prepare_data_splits(
        args.input_file,
        args.output_dir,
        args.test_size,
        args.val_size
    )

if __name__ == '__main__':
    main()