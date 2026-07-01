"""Utility script to download Caltech-256 via torchvision and build a DataLoader."""
from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets


def build_loader(root: Path, batch_size: int, num_workers: int) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.Caltech256(
        root=str(root),
        transform=transform,
        download=True,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Caltech-256 and build a loader.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("./data"),
        help="Directory where the dataset should be stored.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for the loader.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker processes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loader = build_loader(args.root.expanduser(), args.batch_size, args.num_workers)
    dataset = loader.dataset
    sample_batch, _ = next(iter(loader))
    print(f"Caltech-256 downloaded to: {dataset.root}")
    print(f"Samples: {len(dataset)}, classes: {len(dataset.categories)}")
    print(f"Example batch shape: {tuple(sample_batch.shape)}")


if __name__ == "__main__":
    main()
