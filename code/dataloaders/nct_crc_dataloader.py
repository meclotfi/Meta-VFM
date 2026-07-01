"""NCT CRC histology dataloader with train/val/test splits."""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, DefaultDict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


ALLOWED_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _iter_class_folders(root: Path) -> Iterable[Path]:
    for path in sorted(root.iterdir()):
        if path.is_dir():
            yield path


def _gather_samples(root: Path) -> Tuple[List[Tuple[Path, int]], List[str]]:
    class_names = [folder.name for folder in _iter_class_folders(root)]
    if not class_names:
        raise RuntimeError(f"No class folders found under {root}")

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    samples: List[Tuple[Path, int]] = []
    for class_name in class_names:
        class_dir = root / class_name
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in ALLOWED_EXTENSIONS:
                samples.append((image_path, class_to_idx[class_name]))

    if not samples:
        raise RuntimeError(f"No image files with supported extensions found under {root}")

    return samples, class_names


def _stratified_split(
    samples: List[Tuple[Path, int]],
    class_count: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    if not 0 <= val_ratio < 1:
        raise ValueError(f"val_ratio must be within [0, 1); received {val_ratio}")
    if not 0 <= test_ratio < 1:
        raise ValueError(f"test_ratio must be within [0, 1); received {test_ratio}")
    if val_ratio + test_ratio >= 1:
        raise ValueError("Sum of val_ratio and test_ratio must be less than 1.")

    by_class: DefaultDict[int, List[Tuple[Path, int]]] = defaultdict(list)
    for path, label in samples:
        by_class[label].append((path, label))

    rng = random.Random(seed)
    train_samples: List[Tuple[Path, int]] = []
    val_samples: List[Tuple[Path, int]] = []
    test_samples: List[Tuple[Path, int]] = []

    for label in range(class_count):
        class_samples = by_class[label]
        rng.shuffle(class_samples)

        total = len(class_samples)
        test_size = int(total * test_ratio)
        val_size = int(total * val_ratio)

        if test_ratio > 0 and test_size == 0:
            test_size = 1
        if val_ratio > 0 and val_size == 0 and total - test_size > 1:
            val_size = 1

        if val_size + test_size >= total:
            raise ValueError(
                f"Not enough samples in class index {label} to satisfy the requested splits."
            )

        test_samples.extend(class_samples[:test_size])
        val_samples.extend(class_samples[test_size:test_size + val_size])
        train_samples.extend(class_samples[test_size + val_size:])

    if not train_samples:
        raise RuntimeError("Training split resulted in zero samples.")
    if test_ratio > 0 and not test_samples:
        raise RuntimeError("Test split requested but resulted in zero samples.")
    if val_ratio > 0 and not val_samples:
        raise RuntimeError("Validation split requested but resulted in zero samples.")

    return train_samples, val_samples, test_samples


@dataclass
class NCTCRCDataset(Dataset):
    samples: List[Tuple[Path, int]]
    class_names: Sequence[str]
    transform: Optional[Callable] = None

    def __post_init__(self) -> None:
        if not self.samples:
            raise RuntimeError("Attempted to create dataset with zero samples.")

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


def get_data_loaders_nct_crc(config):
    """Return train/val/test dataloaders for the NCT CRC dataset."""
    root = Path(config.data_root).expanduser() / "NCT-CRC-HE-100K"
    samples, class_names = _gather_samples(root)

    val_ratio = getattr(config, "val_split", 0.1)
    test_ratio = getattr(config, "test_split", 0.1)
    seed = getattr(config, "seed", 42)

    train_samples, val_samples, test_samples = _stratified_split(
        samples,
        len(class_names),
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_transform, eval_transform = get_transforms()

    train_dataset = NCTCRCDataset(train_samples, class_names, transform=train_transform)
    val_dataset = (
        NCTCRCDataset(val_samples, class_names, transform=eval_transform)
        if val_samples
        else None
    )
    test_dataset = (
        NCTCRCDataset(test_samples, class_names, transform=eval_transform)
        if test_samples
        else None
    )

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
    print(f"Number of classes: {len(class_names)}")

    data_info = {
        "name": "NCT-CRC-HE-100K",
        "num_classes": len(class_names),
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
        val_split=0.1,
        test_split=0.1,
        seed=42,
    )
    get_data_loaders_nct_crc(config)
