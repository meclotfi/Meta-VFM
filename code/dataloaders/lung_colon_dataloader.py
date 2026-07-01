"""Lung/Colon histopathology dataloader matching the Food-101/Food-11 helpers."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


ALLOWED_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _iter_class_folders(root: Path) -> Iterable[Path]:
    for path in sorted(root.iterdir()):
        if path.is_dir():
            yield path


def _gather_samples(root: Path) -> Tuple[List[Tuple[Path, int]], List[str]]:
    """Return list of (path, class_idx) pairs and ordered class names."""
    class_names = [folder.name for folder in _iter_class_folders(root)]
    if not class_names:
        raise RuntimeError(f"No class folders found in {root}")
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    samples: List[Tuple[Path, int]] = []
    for class_name in class_names:
        class_dir = root / class_name
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() in ALLOWED_EXTENSIONS and image_path.is_file():
                samples.append((image_path, class_to_idx[class_name]))

    if not samples:
        raise RuntimeError(f"No image files with supported extensions found under {root}")

    return samples, class_names


@dataclass
class LungColonDataset(Dataset):
    """Dataset for the histopathology images that loads RGB tensors."""

    samples: List[Tuple[Path, int]]
    class_names: Sequence[str]
    transform: Optional[Callable] = None

    def __post_init__(self) -> None:
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        if not self.samples:
            raise RuntimeError("Attempted to create a dataset with zero samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
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


def _split_samples(
    samples: List[Tuple[Path, int]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    if not 0 <= val_ratio < 1:
        raise ValueError(f"val_ratio must be in [0, 1); received {val_ratio}")

    if not val_ratio:
        return samples, []

    indices = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_size = int(len(samples) * val_ratio)
    if val_size == 0:
        val_size = 1  # ensure at least one validation sample if ratio > 0
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    if not train_indices:
        raise ValueError("Validation split too large; no training samples remain.")

    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]
    return train_samples, val_samples


def get_data_loaders_lung_colon(config):
    """Build DataLoaders for the Lung/Colon histopathology dataset."""
    root = Path(config.data_root).expanduser() / "lung_colon_image_set"
    train_val_dir = root / "Train and Validation Set"
    test_dir = root / "Test Set"

    train_transform, eval_transform = get_transforms()

    train_samples, class_names = _gather_samples(train_val_dir)
    test_samples, _ = _gather_samples(test_dir)

    val_ratio = getattr(config, "val_split", 0.2)
    seed = getattr(config, "seed", 42)
    train_samples, val_samples = _split_samples(train_samples, val_ratio, seed)

    train_dataset = LungColonDataset(train_samples, class_names, transform=train_transform)
    val_dataset = (
        LungColonDataset(val_samples, class_names, transform=eval_transform)
        if val_samples
        else None
    )
    test_dataset = LungColonDataset(test_samples, class_names, transform=eval_transform)

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

    num_classes = len(class_names)

    print(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")
    else:
        print("Validation samples: 0 (validation loader disabled)")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {num_classes}")

    data_info = {
        "name": "Lung-Colon Histopathology",
        "num_classes": num_classes,
        "class_names": list(class_names),
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
        val_split=0.2,
        seed=42,
    )
    get_data_loaders_lung_colon(config)
