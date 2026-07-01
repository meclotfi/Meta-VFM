"""Utility script to download Stanford Cars via torchvision and build a DataLoader."""
from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets


def build_loader(
    root: Path,
    batch_size: int,
    num_workers: int,
    split: str,
) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.StanfordCars(
        root=str(root),
        split=split,
        transform=transform,
        download=True,
    )
    shuffle = split == "train"
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Stanford Cars and build a DataLoader for a chosen split.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("./data"),
        help="Directory where the dataset should be stored.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for the loader.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker processes.")
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
        help="Dataset split to download and load.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loader = build_loader(
        args.root.expanduser(),
        args.batch_size,
        args.num_workers,
        args.split,
    )
    dataset = loader.dataset
    sample_batch, _ = next(iter(loader))
    print(f"Stanford Cars split '{args.split}' downloaded to: {dataset.root}")
    print(f"Samples: {len(dataset)}, classes: {len(dataset.classes)}")
    print(f"Example batch shape: {tuple(sample_batch.shape)}")


if __name__ == "__main__":
    main()
