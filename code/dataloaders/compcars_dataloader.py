"""CompCars dataloader using official classification train/test splits."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


SplitSample = Tuple[str, int]


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


def _resolve_compcars_root(base_root: Path) -> Path:
    """Return the folder that contains `image/` and `train_test_split/`."""
    candidates = [
        base_root,
        base_root / "CompCars",
        base_root / "data",
        base_root / "CompCars" / "data",
        base_root.parent / "compCars",
        base_root.parent / "compcars",
    ]
    for candidate in candidates:
        if (candidate / "image").exists():
            return candidate
    # Fallback: search one level down for case-insensitive matches.
    for child in base_root.parent.glob("*compcar*"):
        if child.is_dir() and (child / "image").exists():
            return child
    raise FileNotFoundError(
        f"Could not locate CompCars `image/` folder under {base_root}. "
        "Make sure the archive is fully extracted (see README instructions)."
    )


def _parse_split_file(file_path: Path) -> List[SplitSample]:
    if not file_path.exists():
        raise FileNotFoundError(f"Split file not found: {file_path}")

    samples: List[SplitSample] = []
    with file_path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            rel_path = parts[0]
            label: Optional[int] = None
            if len(parts) >= 2:
                try:
                    label = int(parts[1])
                except ValueError:
                    label = None
            if label is None:
                path_parts = Path(rel_path).parts
                if not path_parts:
                    raise ValueError(f"Unable to parse split line: {line}")
                # Infer label from the first directory (dataset uses folder IDs for classes).
                try:
                    label = int(path_parts[0])
                except ValueError:
                    raise ValueError(
                        f"Unable to infer label from split line (expected numeric folder): {line}"
                    ) from None
            if label is None:
                raise ValueError(f"Missing label information in split file line: {line}")
            samples.append((rel_path, label))
    if not samples:
        raise RuntimeError(f"No samples parsed from {file_path}")
    return samples


def _normalise_labels(
    *splits: Iterable[SplitSample],
) -> Tuple[List[SplitSample], ...]:
    labels = sorted({label for split in splits for _, label in split})
    label_to_index: Dict[int, int] = {label: idx for idx, label in enumerate(labels)}

    def convert(split: Iterable[SplitSample]) -> List[SplitSample]:
        return [(path, label_to_index[label]) for path, label in split]

    return tuple(convert(split) for split in splits)


def _create_validation_split(
    samples: List[SplitSample],
    val_ratio: float,
    seed: int,
) -> Tuple[List[SplitSample], List[SplitSample]]:
    if val_ratio <= 0:
        return samples, []

    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be between 0 and 1 (exclusive); received {val_ratio}")

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


class CompCarsDataset(Dataset):
    """Dataset wrapper returning (image tensor, label) pairs from CompCars."""

    def __init__(
        self,
        root: Path,
        samples: Sequence[SplitSample],
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = root
        self.samples = list(samples)
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


def get_data_loaders_compcars(config):
    """Return train/val/test dataloaders for the CompCars classification task."""
    base_root = Path(config.data_root).expanduser() / "CompCars"
    dataset_root = _resolve_compcars_root(base_root)

    split_root = dataset_root / "train_test_split"
    classification_dir = split_root / "classification"
    if classification_dir.exists():
        train_file = classification_dir / "train.txt"
        test_file = classification_dir / "test.txt"
    else:
        train_file = split_root / "classification_train.txt"
        test_file = split_root / "classification_test.txt"

    train_samples = _parse_split_file(train_file)
    test_samples = _parse_split_file(test_file) if test_file.exists() else []

    train_samples, test_samples = _normalise_labels(train_samples, test_samples)

    val_ratio = getattr(config, "val_split", 0.1)
    seed = getattr(config, "seed", 42)
    train_samples, val_samples = _create_validation_split(train_samples, val_ratio, seed)

    train_transform, eval_transform = get_transforms()

    image_root = dataset_root / "image"
    if not image_root.exists():
        raise FileNotFoundError(
            f"Expected `image/` directory under {dataset_root}, but it was not found."
        )

    train_dataset = CompCarsDataset(image_root, train_samples, transform=train_transform)
    val_dataset = CompCarsDataset(image_root, val_samples, transform=eval_transform) if val_samples else None
    test_dataset = CompCarsDataset(image_root, test_samples, transform=eval_transform) if test_samples else None

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

    num_classes = 0
    if train_samples:
        num_classes = max(label for _, label in train_samples) + 1

    print(f"Training samples: {len(train_dataset)}")
    if val_loader:
        print(f"Validation samples: {len(val_dataset)}")
    else:
        print("Validation samples: 0 (validation loader disabled)")
    if test_loader:
        print(f"Test samples: {len(test_dataset)}")
    else:
        print("Test samples: 0 (test loader disabled)")
    print(f"Number of classes: {num_classes}")

    data_info = {
        "name": "CompCars Classification",
        "num_classes": num_classes,
        "class_names": [str(idx) for idx in range(num_classes)],
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
    get_data_loaders_compcars(config)
