"""Human Action Recognition dataloader with train/val splits from CSV annotations."""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


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


def _read_training_csv(csv_path: Path) -> List[Tuple[str, str]]:
    samples: List[Tuple[str, str]] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if "filename" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"Expected columns 'filename' and 'label' in {csv_path}")
        for row in reader:
            filename = row.get("filename", "").strip()
            label = row.get("label", "").strip()
            if not filename or not label:
                continue
            samples.append((filename, label))
    if not samples:
        raise RuntimeError(f"No samples parsed from {csv_path}")
    return samples


def _read_test_csv(csv_path: Path) -> List[str]:
    filenames: List[str] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if "filename" not in reader.fieldnames:
            raise ValueError(f"Expected column 'filename' in {csv_path}")
        for row in reader:
            filename = row.get("filename", "").strip()
            if filename:
                filenames.append(filename)
    return filenames


def _create_validation_split(
    samples: List[Tuple[str, str]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
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
        raise ValueError("Validation split too large; no training samples remain.")

    return train_split, val_split


class HumanActionRecognitionDataset(Dataset):
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
        filename, label = self.samples[index]
        image_path = self.image_root / filename
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class HumanActionRecognitionTestDataset(Dataset):
    """Unlabeled dataset for the official test split."""

    def __init__(
        self,
        image_root: Path,
        filenames: Sequence[str],
        transform: Optional[Callable] = None,
    ) -> None:
        self.image_root = image_root
        self.filenames = list(filenames)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int):
        filename = self.filenames[index]
        image_path = self.image_root / filename
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, filename  # Return filename for downstream tracking


def get_data_loaders_human_action_recognition(config):
    """Return data loaders for the Human Action Recognition dataset."""
    dataset_root = Path(config.data_root).expanduser() / "Human-Action-Recognition"
    train_dir = dataset_root / "train"
    test_dir = dataset_root / "test"
    train_csv = dataset_root / "Training_set.csv"
    test_csv = dataset_root / "Testing_set.csv"

    if not train_dir.exists() or not train_csv.exists():
        raise FileNotFoundError(
            f"Expected directories and CSV annotations under {dataset_root}"
        )

    train_records = _read_training_csv(train_csv)
    test_filenames = _read_test_csv(test_csv) if test_csv.exists() else []

    class_names = sorted({label for _, label in train_records})
    label_to_index: Dict[str, int] = {label: idx for idx, label in enumerate(class_names)}
    indexed_records = [(filename, label_to_index[label]) for filename, label in train_records]

    val_ratio = getattr(config, "val_split", 0.1)
    seed = getattr(config, "seed", 42)
    train_samples, val_samples = _create_validation_split(indexed_records, val_ratio, seed)

    train_transform, eval_transform = get_transforms()

    train_dataset = HumanActionRecognitionDataset(train_dir, train_samples, transform=train_transform)
    val_dataset = HumanActionRecognitionDataset(train_dir, val_samples, transform=eval_transform) if val_samples else None

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

    print("Human Action Recognition")
    print(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")
    else:
        print("Validation samples: 0 (validation loader disabled)")

    data_info = {
        "name": "Human Action Recognition",
        "num_classes": len(class_names),
        "class_names": class_names,
    }

    return train_loader, val_loader, val_loader, data_info
