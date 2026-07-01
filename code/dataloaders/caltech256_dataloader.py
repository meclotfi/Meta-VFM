"""Caltech-256 dataloader that downloads via torchvision and returns loaders."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision import datasets


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


def _split_indices(
    num_samples: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    if not 0 <= val_ratio < 1:
        raise ValueError(f"val_ratio must be within [0, 1); received {val_ratio}")
    if not 0 <= test_ratio < 1:
        raise ValueError(f"test_ratio must be within [0, 1); received {test_ratio}")
    if val_ratio + test_ratio >= 1:
        raise ValueError("Sum of val_ratio and test_ratio must be less than 1.")

    indices = list(range(num_samples))
    rng = random.Random(seed)
    rng.shuffle(indices)

    test_count = int(num_samples * test_ratio)
    val_count = int(num_samples * val_ratio)

    if test_ratio > 0 and test_count == 0:
        test_count = 1
    if val_ratio > 0 and val_count == 0 and num_samples - test_count > 1:
        val_count = 1

    if val_count + test_count >= num_samples:
        raise RuntimeError("Requested validation/test split leaves no training data.")

    test_indices = indices[:test_count]
    val_indices = indices[test_count:test_count + val_count]
    train_indices = indices[test_count + val_count:]

    return train_indices, val_indices, test_indices


def get_data_loaders_caltech256(config):
    """Download Caltech-256 (if needed) and produce train/val/test loaders."""
    data_root = Path(config.data_root).expanduser()
    train_transform, eval_transform = get_transforms()

    base_dataset = datasets.Caltech256(
        root=str(data_root),
        transform=None,
        download=True,
    )
    num_samples = len(base_dataset)
    class_names = base_dataset.categories

    val_ratio = getattr(config, "val_split", 0.1)
    test_ratio = getattr(config, "test_split", 0.1)
    seed = getattr(config, "seed", 42)

    train_indices, val_indices, test_indices = _split_indices(
        num_samples,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_dataset = datasets.Caltech256(
        root=str(data_root),
        transform=train_transform,
        download=False,
    )
    val_dataset_full = datasets.Caltech256(
        root=str(data_root),
        transform=eval_transform,
        download=False,
    )
    test_dataset_full = datasets.Caltech256(
        root=str(data_root),
        transform=eval_transform,
        download=False,
    )

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset_full, val_indices) if val_indices else None
    test_subset = Subset(test_dataset_full, test_indices) if test_indices else None

    batch_size = config.batch_size
    test_batch_size = getattr(config, "test_batch_size", batch_size)
    num_workers = getattr(config, "num_workers", 4)
    pin_memory = getattr(config, "pin_memory", True)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = (
        DataLoader(
            val_subset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        if val_subset
        else None
    )
    test_loader = (
        DataLoader(
            test_subset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        if test_subset
        else None
    )

    print(f"Training samples: {len(train_subset)}")
    if val_subset:
        print(f"Validation samples: {len(val_subset)}")
    else:
        print("Validation samples: 0 (validation loader disabled)")
    if test_subset:
        print(f"Test samples: {len(test_subset)}")
    else:
        print("Test samples: 0 (test loader disabled)")
    print(f"Number of classes: {len(class_names)}")

    data_info = {
        "name": "Caltech-256",
        "num_classes": len(class_names),
        "class_names": class_names,
    }

    return train_loader, val_loader, test_loader, data_info


if __name__ == "__main__":
    from types import SimpleNamespace

    config = SimpleNamespace(
        data_root="./data",
        batch_size=32,
        test_batch_size=64,
        num_workers=4,
        pin_memory=True,
        val_split=0.1,
        test_split=0.1,
        seed=42,
    )
    get_data_loaders_caltech256(config)
