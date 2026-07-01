#!/usr/bin/env python3
"""
Script to download and test loading a Hugging Face dataset.
Usage: python download_hf_dataset.py <dataset_name>
"""

import argparse
import sys
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Download and test loading a Hugging Face dataset"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the Hugging Face dataset (e.g., 'squad', 'imdb')"
    )

    args = parser.parse_args()

    print(f"Downloading dataset: {args.dataset_name}")
    print("-" * 50)

    try:
        # Load the dataset with default settings to ./data folder
        dataset = load_dataset(args.dataset_name, cache_dir="./data")

        print("\n✓ Dataset loaded successfully!")
        print(f"\nDataset info:")
        print(f"Type: {type(dataset)}")

        # Display dataset structure (all splits)
        print(f"\nAvailable splits: {list(dataset.keys())}")
        for split_name, split_data in dataset.items():
            print(f"\n{split_name}:")
            print(f"  Features: {split_data.features}")
            print(f"  Number of examples: {len(split_data)}")

        print("\n✓ Test completed successfully!")

    except Exception as e:
        print(f"\n✗ Error loading dataset: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
