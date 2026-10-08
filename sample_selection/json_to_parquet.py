#!/usr/bin/env python3
"""Convert filtered JSON to parquet format.

Extracts db_id, question, query fields from filtered JSON and saves as parquet.

Usage:
    python json_to_parquet.py --input outputs/14b_sampling_results_filtered.json --output outputs/14b_train_variance_not_zero.parquet
"""

import json
import argparse
from pathlib import Path
import pandas as pd


def convert_json_to_parquet(input_file: str, output_file: str):
    """Convert filtered JSON to parquet format."""
    
    print(f"Loading filtered data from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    # Extract fields
    parquet_data = []
    for sample in data:
        parquet_data.append({
            'db_id': sample['db_id'],
            'question': sample['question'],
            'query': sample['query']
        })
    
    # Create DataFrame
    df = pd.DataFrame(parquet_data)
    
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst 3 samples:")
    for i, row in df.head(3).iterrows():
        print(f"\n  Sample {i+1}:")
        print(f"    db_id: {row['db_id']}")
        print(f"    question: {row['question']}")
        print(f"    query: {row['query']}")
    
    # Save to parquet
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to {output_file}...")
    df.to_parquet(output_file, index=False)
    
    print(f"✅ Done! Saved {len(df)} samples to {output_file}")
    
    # Verify
    print(f"\nVerifying parquet file...")
    df_verify = pd.read_parquet(output_file)
    print(f"  Shape: {df_verify.shape}")
    print(f"  Columns: {df_verify.columns.tolist()}")
    print(f"  ✅ Verification passed!")


def main():
    parser = argparse.ArgumentParser(description="Convert filtered JSON to parquet")
    parser.add_argument(
        "--input",
        default="outputs/14b_sampling_results_filtered.json",
        help="Input JSON file path"
    )
    parser.add_argument(
        "--output",
        default="outputs/14b_train_variance_not_zero.parquet",
        help="Output parquet file path"
    )
    
    args = parser.parse_args()
    
    convert_json_to_parquet(args.input, args.output)


if __name__ == "__main__":
    main()
