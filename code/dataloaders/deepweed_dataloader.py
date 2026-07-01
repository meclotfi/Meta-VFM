"""DeepWeed dataloader using provided subset CSV annotations."""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.RandomResizedCrop(299, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


def _load_label_mapping(labels_csv: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    with labels_csv.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                label_idx = int(row["Label"])
            except (ValueError, KeyError) as exc:
                raise ValueError(f"Invalid label row in {labels_csv}: {row}") from exc
            name = row.get("Species") or row.get("Class", "")
            mapping.setdefault(label_idx, name)
    if not mapping:
        raise RuntimeError(f"No labels found in {labels_csv}")
    return mapping


def _load_subset(csv_path: Path) -> List[Tuple[str, int]]:
    samples: List[Tuple[str, int]] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            filename = row.get("Filename")
            label_value = row.get("Label")
            if not filename or label_value is None:
                continue
            try:
                label = int(label_value)
            except ValueError as exc:
                raise ValueError(f"Invalid label in {csv_path}: {row}") from exc
            samples.append((filename, label))
    if not samples:
        raise RuntimeError(f"No samples found in {csv_path}")
    return samples


def _create_validation_split(
    samples: List[Tuple[str, int]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    if val_ratio <= 0:
        return samples, []
    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be between 0 and 1; received {val_ratio}")
    indices = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = int(len(samples) * val_ratio)
    if val_size == 0:
        val_size = 1
    val_indices = set(indices[:val_size])
    train_split = [sample for idx, sample in enumerate(samples) if idx not in val_indices]
    val_split = [sample for idx, sample in enumerate(samples) if idx in val_indices]
    if not train_split:
        raise ValueError("Validation ratio too large; no training samples remain.")
    return train_split, val_split


class DeepWeedDataset(Dataset):
    def __init__(
        self,
        image_root: Path,
        samples: Sequence[Tuple[str, int]],
        transform: Optional[Callable] = None,
    ) -> None:
        self.image_root = image_root
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        rel_path, label = self.samples[index]
        image_path = self.image_root / rel_path
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_data_loaders_deepweed(config):
    """Return train/val/test dataloaders for DeepWeed."""
    root = Path(config.data_root).expanduser() / "DeepWeed"
    image_root = root / "images"
    labels_root = root / "labels"
    if not image_root.exists() or not labels_root.exists():
        raise FileNotFoundError(f"DeepWeed folders not found under {root}")

    subset_index = getattr(config, "subset_index", 0)
    train_csv = labels_root / f"train_subset{subset_index}.csv"
    test_csv = labels_root / f"test_subset{subset_index}.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(
            f"Expected subset files {train_csv.name} and {test_csv.name} in {labels_root}"
        )

    label_mapping = _load_label_mapping(labels_root / "labels.csv")
    train_entries = _load_subset(train_csv)
    test_entries = _load_subset(test_csv)

    val_ratio = getattr(config, "val_split", 0.1)
    seed = getattr(config, "seed", 42)
    train_entries, val_entries = _create_validation_split(train_entries, val_ratio, seed)

    train_transform, eval_transform = get_transforms()

    train_dataset = DeepWeedDataset(image_root, train_entries, transform=train_transform)
    val_dataset = DeepWeedDataset(image_root, val_entries, transform=eval_transform) if val_entries else None
    test_dataset = DeepWeedDataset(image_root, test_entries, transform=eval_transform)

    batch_size = config.batch_size
    test_batch_size = getattr(config, "test_batch_size", batch_size)
    num_workers = getattr(config, "num_workers", 4)
    pin_memory = getattr(config, "pin_memory", True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        if val_dataset
        else None
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    class_count = len(label_mapping)
    class_names = [label_mapping[idx] for idx in sorted(label_mapping.keys())]

    print(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")
    else:
        print("Validation samples: 0 (validation loader disabled)")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {class_count}")

    data_info = {
        "name": "DeepWeed",
        "num_classes": class_count,
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
        seed=42,
        subset_index=0,
    )
    get_data_loaders_deepweed(config)
