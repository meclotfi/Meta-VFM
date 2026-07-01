"""APTOS 2019 dataloader that consumes the Kaggle folder under ./data."""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


@dataclass(frozen=True)
class APTOSSample:
    image_path: Path
    label: Optional[int]
    id_code: str


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomResizedCrop(448, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


def _load_samples(csv_path: Path, images_dir: Path, require_labels: bool) -> List[APTOSSample]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    samples: List[APTOSSample] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if "id_code" not in reader.fieldnames:
            raise ValueError(f"'id_code' column missing from {csv_path}")
        for row in reader:
            image_id = row["id_code"].strip()
            if not image_id:
                continue
            image_path = images_dir / f"{image_id}.png"
            if not image_path.exists():
                raise FileNotFoundError(f"Image referenced in CSV is missing: {image_path}")
            label: Optional[int] = None
            if require_labels:
                if "diagnosis" not in row or row["diagnosis"] == "":
                    raise ValueError(f"Diagnosis label missing for sample {image_id} in {csv_path}")
                label = int(row["diagnosis"])
            samples.append(APTOSSample(image_path=image_path, label=label, id_code=image_id))
    if not samples:
        raise RuntimeError(f"No samples parsed from {csv_path}")
    return samples


def _train_val_split(
    samples: Sequence[APTOSSample],
    val_ratio: float,
    seed: int,
) -> Tuple[List[APTOSSample], List[APTOSSample]]:
    if not 0 <= val_ratio < 1:
        raise ValueError(f"val_ratio must be within [0, 1); received {val_ratio}")
    if val_ratio == 0:
        return list(samples), []
    indices = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = max(1, int(len(samples) * val_ratio))
    val_indices = set(indices[:val_count])
    train_split = [sample for idx, sample in enumerate(samples) if idx not in val_indices]
    val_split = [sample for idx, sample in enumerate(samples) if idx in val_indices]
    if not train_split:
        raise ValueError("Validation split leaves no training samples. Reduce val_ratio.")
    return train_split, val_split


class APTOSDataset(Dataset):
    """Dataset wrapper returning (tensor, label) tuples for APTOS samples."""

    def __init__(
        self,
        samples: Sequence[APTOSSample],
        transform: Optional[Callable] = None,
        unlabeled_value: int = -1,
        return_id: bool = False,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform
        self.unlabeled_value = unlabeled_value
        self.return_id = return_id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.image_path) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = sample.label if sample.label is not None else self.unlabeled_value
        if self.return_id:
            return image, label, sample.id_code
        return image, label


def get_data_loaders_aptos(config):
    """Return train/val/test dataloaders for the local APTOS 2019 dataset."""
    root = Path(config.data_root).expanduser() / "aptos2019-blindness-detection"
    train_csv = root / "train.csv"
    train_images = root / "train_images"
    train_samples = _load_samples(train_csv, train_images, require_labels=True)

    val_ratio = getattr(config, "val_split", 0.1)
    seed = getattr(config, "seed", 42)
    train_split, val_split = _train_val_split(train_samples, val_ratio=val_ratio, seed=seed)

    train_transform, eval_transform = get_transforms()

    train_dataset = APTOSDataset(train_split, transform=train_transform)
    val_dataset = APTOSDataset(val_split, transform=eval_transform) if val_split else None

    test_loader = None
    test_csv = root / "test.csv"
    test_images = root / "test_images"
    if test_csv.exists() and test_images.exists():
        test_samples = _load_samples(test_csv, test_images, require_labels=False)
        test_dataset = APTOSDataset(test_samples, transform=eval_transform, return_id=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=getattr(config, "test_batch_size", getattr(config, "batch_size")),
            shuffle=False,
            num_workers=getattr(config, "num_workers", 4),
            pin_memory=getattr(config, "pin_memory", True),
        )

    batch_size = config.batch_size
    num_workers = getattr(config, "num_workers", 4)
    pin_memory = getattr(config, "pin_memory", True)
    test_batch_size = getattr(config, "test_batch_size", batch_size)

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

    num_classes = len({sample.label for sample in train_samples if sample.label is not None})
    class_names = [f"Grade {idx}" for idx in range(num_classes)]

    data_info = {
        "name": "APTOS 2019 Blindness Detection",
        "num_classes": num_classes,
        "class_names": class_names,
    }

    return train_loader, val_loader, test_loader, data_info
