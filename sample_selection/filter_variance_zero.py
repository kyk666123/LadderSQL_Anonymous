#!/usr/bin/env python3
"""Filter samples with variance = 0 (all 1.0 or all 0.0).

Filters out samples where all 16 rollouts have the same reward,
keeping only samples with mixed outcomes (variance > 0).

Usage:
    python filter_variance_zero.py --input outputs/14b_sampling_results.json --output outputs/14b_sampling_results_filtered.json
"""

import json
import argparse
from pathlib import Path


def filter_variance_zero(input_file: str, output_file: str):
    """Filter samples with variance = 0."""
    
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    # Filter samples
    filtered_data = []
    all_ones = 0
    all_zeros = 0
    mixed = 0
    timed_out = 0
    
    for sample in data:
        reward_list = sample.get('reward_list', [])
        
        # Drop timed-out samples (硬超时轨迹, 奖励不可信)
        if sample.get('timed_out') or not reward_list or any(r is None for r in reward_list):
            timed_out += 1
            continue
        
        # Check if all rewards are the same
        if all(r == 1.0 for r in reward_list):
            all_ones += 1
            continue  # Skip this sample
        elif all(r == 0.0 for r in reward_list):
            all_zeros += 1
            continue  # Skip this sample
        else:
            mixed += 1
            filtered_data.append(sample)
    
    # Print statistics
    print(f"\n{'='*80}")
    print(f"Filtering Results:")
    print(f"{'='*80}")
    print(f"Total samples:          {len(data):>6}")
    print(f"Timed-out/None (removed):{timed_out:>6} ({timed_out/len(data)*100:.1f}%)")
    print(f"All 1.0 (removed):      {all_ones:>6} ({all_ones/len(data)*100:.1f}%)")
    print(f"All 0.0 (removed):      {all_zeros:>6} ({all_zeros/len(data)*100:.1f}%)")
    print(f"Mixed (kept):           {mixed:>6} ({mixed/len(data)*100:.1f}%)")
    print(f"{'='*80}")
    
    # Save filtered data
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving filtered data to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Done! Saved {len(filtered_data)} samples")
    
    # Show some examples of kept samples
    print(f"\n{'='*80}")
    print(f"Example of kept samples (mixed outcomes):")
    print(f"{'='*80}")
    for i, sample in enumerate(filtered_data[:3]):
        print(f"\nSample {i+1}:")
        print(f"  DB: {sample['db_id']}")
        print(f"  Question: {sample['question']}")
        print(f"  Rewards: {sample['reward_list']}")
        avg_reward = sum(sample['reward_list']) / len(sample['reward_list'])
        print(f"  Avg Reward: {avg_reward:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Filter samples with variance = 0")
    parser.add_argument(
        "--input",
        default="outputs/14b_sampling_results.json",
        help="Input JSON file path"
    )
    parser.add_argument(
        "--output",
        default="outputs/14b_sampling_results_filtered.json",
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    filter_variance_zero(args.input, args.output)


if __name__ == "__main__":
    main()
