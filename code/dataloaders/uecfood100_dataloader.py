"""UEC-Food100 dataloader using folder hierarchy with stratified splits."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple
from collections import defaultdict

from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


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


def _collect_samples(root: Path) -> Tuple[List[Tuple[Path, str]], List[str]]:
    samples: List[Tuple[Path, str]] = []
    class_names: List[str] = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        class_name = class_dir.name
        class_names.append(class_name)
        for image_path in sorted(class_dir.glob("*")):
            if image_path.is_file() and image_path.suffix.lower() in ALLOWED_EXTENSIONS:
                samples.append((image_path, class_name))
    if not samples:
        raise RuntimeError(f"No image files found under {root}")
    return samples, class_names


def _index_samples(
    samples: Iterable[Tuple[Path, str]],
    class_to_idx: Dict[str, int],
) -> List[Tuple[Path, int]]:
    indexed: List[Tuple[Path, int]] = []
    for path, class_name in samples:
        if class_name not in class_to_idx:
            raise KeyError(f"Unknown class name: {class_name}")
        indexed.append((path, class_to_idx[class_name]))
    return indexed


def _stratified_split(
    samples: List[Tuple[Path, int]],
    num_classes: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    if not 0 <= val_ratio < 1:
        raise ValueError(f"val_ratio must be in [0, 1); received {val_ratio}")
    if not 0 <= test_ratio < 1:
        raise ValueError(f"test_ratio must be in [0, 1); received {test_ratio}")
    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be less than 1.")

    by_class: DefaultDict[int, List[Tuple[Path, int]]] = defaultdict(list)
    for sample in samples:
        by_class[sample[1]].append(sample)

    rng = random.Random(seed)
    train_split: List[Tuple[Path, int]] = []
    val_split: List[Tuple[Path, int]] = []
    test_split: List[Tuple[Path, int]] = []

    for class_idx in range(num_classes):
        class_samples = by_class[class_idx]
        rng.shuffle(class_samples)

        total = len(class_samples)
        test_size = int(total * test_ratio)
        val_size = int(total * val_ratio)

        if test_ratio > 0 and test_size == 0:
            test_size = 1
        if val_ratio > 0 and val_size == 0 and total - test_size > 1:
            val_size = 1

        if val_size + test_size >= total:
            raise ValueError(f"Not enough samples in class index {class_idx} for requested splits.")

        test_split.extend(class_samples[:test_size])
        val_split.extend(class_samples[test_size:test_size + val_size])
        train_split.extend(class_samples[test_size + val_size:])

    if not train_split:
        raise RuntimeError("Training split contains zero samples.")
    if test_ratio > 0 and not test_split:
        raise RuntimeError("Test split requested but contains zero samples.")
    if val_ratio > 0 and not val_split:
        raise RuntimeError("Validation split requested but contains zero samples.")

    return train_split, val_split, test_split


class UECFood100Dataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Tuple[Path, int]],
        transform: Optional[Callable] = None,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_data_loaders_uecfood100(config):
    """Return train/val/test loaders for the UEC-Food100 dataset."""
    root = Path(config.data_root).expanduser() / "UECFOOD100"
    if not root.exists():
        raise FileNotFoundError(f"UECFOOD100 root not found: {root}")

    raw_samples, class_names = _collect_samples(root)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    indexed_samples = _index_samples(raw_samples, class_to_idx)

    val_ratio = getattr(config, "val_split", 0.1)
    test_ratio = getattr(config, "test_split", 0.1)
    seed = getattr(config, "seed", 42)

    train_samples, val_samples, test_samples = _stratified_split(
        indexed_samples,
        num_classes=len(class_names),
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_transform, eval_transform = get_transforms()

    train_dataset = UECFood100Dataset(train_samples, transform=train_transform)
    val_dataset = UECFood100Dataset(val_samples, transform=eval_transform) if val_samples else None
    test_dataset = UECFood100Dataset(test_samples, transform=eval_transform) if test_samples else None

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
        "name": "UEC-Food100",
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
    get_data_loaders_uecfood100(config)
