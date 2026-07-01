"""Stanford Online Products dataloader using official train/test metadata."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


Sample = Tuple[Path, int]


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
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


def _read_split_file(file_path: Path) -> List[str]:
    if not file_path.exists():
        raise FileNotFoundError(f"Split file not found: {file_path}")

    samples: List[str] = []
    with file_path.open("r") as handle:
        header_seen = False
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if not header_seen:
                header_seen = True
                if line.lower().startswith("image_id"):
                    continue
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Unexpected row in {file_path}: {line}")
            rel_path = parts[3]
            samples.append(rel_path)

    if not samples:
        raise RuntimeError(f"No samples parsed from {file_path}")
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


class StanfordOnlineProductsDataset(Dataset):
    """Dataset wrapper that loads RGB images for Stanford Online Products."""

    def __init__(
        self,
        root: Path,
        samples: Sequence[Tuple[str, int]],
        class_names: Sequence[str],
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = root
        self.samples = list(samples)
        self.class_names = list(class_names)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        rel_path, label = self.samples[index]
        image_path = self.root / rel_path
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_data_loaders_stanford_online_products(config):
    """Return train/val/test DataLoaders for the Stanford Online Products dataset."""
    dataset_root = Path(config.data_root).expanduser() / "Stanford_Online_Products"
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    train_file = dataset_root / "Ebay_train.txt"
    test_file = dataset_root / "Ebay_test.txt"

    train_paths = _read_split_file(train_file)
    test_paths = _read_split_file(test_file) if test_file.exists() else []

    def class_name_from_path(rel_path: str) -> str:
        parts = Path(rel_path).parts
        if not parts:
            raise ValueError(f"Invalid relative path in split file: {rel_path}")
        return parts[0]

    class_names = sorted({class_name_from_path(p) for p in train_paths + test_paths})
    class_to_idx: Dict[str, int] = {name: idx for idx, name in enumerate(class_names)}

    def assign_labels(paths: List[str]) -> List[Tuple[str, int]]:
        samples: List[Tuple[str, int]] = []
        for rel_path in paths:
            name = class_name_from_path(rel_path)
            samples.append((rel_path, class_to_idx[name]))
        return samples

    train_entries = assign_labels(train_paths)
    test_entries = assign_labels(test_paths)

    val_ratio = getattr(config, "val_split", 0.1)
    seed = getattr(config, "seed", 42)
    train_entries, val_entries = _create_validation_split(train_entries, val_ratio, seed)

    class_count = len(class_names)

    train_transform, eval_transform = get_transforms()

    train_dataset = StanfordOnlineProductsDataset(dataset_root, train_entries, class_names, transform=train_transform)
    val_dataset = StanfordOnlineProductsDataset(dataset_root, val_entries, class_names, transform=eval_transform) if val_entries else None
    test_dataset = StanfordOnlineProductsDataset(dataset_root, test_entries, class_names, transform=eval_transform) if test_entries else None

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
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        if test_dataset
        else None
    )

    print(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")
    else:
        print("Validation samples: 0 (validation loader disabled)")
    if test_dataset:
        print(f"Test samples: {len(test_dataset)}")
    else:
        print("Test samples: 0 (test loader disabled)")
    print(f"Number of classes: {class_count}")

    data_info = {
        "name": "Stanford Online Products",
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
    )
    get_data_loaders_stanford_online_products(config)
