"""Chest X-ray (pneumonia) dataloader with optional validation split."""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


ALLOWED_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")


def _resolve_chest_xray_root(data_root: Path) -> Path:
    candidates = [
        data_root / "chest_xray",
        data_root / "ChestXRay",
        data_root / "chest-xray",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = list(data_root.glob("*chest*ray*"))
    if not matches:
        raise FileNotFoundError(
            f"Could not locate a chest X-ray dataset folder under {data_root}. "
            "Ensure the dataset is extracted (e.g., chest_xray/train, chest_xray/test)."
        )
    if len(matches) > 1:
        match_paths = ", ".join(str(path) for path in matches)
        raise RuntimeError(
            "Multiple possible chest X-ray dataset folders found: "
            f"{match_paths}. Please keep only the intended folder."
        )
    return matches[0]


def _available_split_dirs(root: Path) -> Dict[str, Path]:
    splits: Dict[str, Path] = {}
    for split_name in ("train", "val", "test"):
        split_path = root / split_name
        if split_path.exists():
            splits[split_name] = split_path
    if "train" not in splits:
        raise FileNotFoundError(
            f"Train split not found under {root}. Expected a 'train' directory."
        )
    return splits


def _discover_class_names(split_dirs: Iterable[Path]) -> List[str]:
    names = set()
    for split_root in split_dirs:
        for entry in split_root.iterdir():
            if entry.is_dir():
                names.add(entry.name)
    if not names:
        raise RuntimeError("No class folders detected across the available splits.")
    return sorted(names)


def _gather_split_samples(split_root: Path, class_to_idx: Dict[str, int]) -> List[Tuple[Path, int]]:
    samples: List[Tuple[Path, int]] = []
    for class_dir in sorted(split_root.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        if class_name not in class_to_idx:
            raise KeyError(f"Unknown class folder '{class_name}' encountered under {split_root}")
        label = class_to_idx[class_name]
        for file_path in class_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ALLOWED_EXTENSIONS:
                samples.append((file_path, label))
    return samples


def _train_val_split(
    train_samples: List[Tuple[Path, int]],
    class_count: int,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    if not 0 <= val_ratio < 1:
        raise ValueError(f"val_ratio must lie within [0, 1); received {val_ratio}")
    if val_ratio == 0:
        return train_samples, []

    by_class: DefaultDict[int, List[Tuple[Path, int]]] = defaultdict(list)
    for path, label in train_samples:
        by_class[label].append((path, label))

    rng = random.Random(seed)
    new_train: List[Tuple[Path, int]] = []
    val_samples: List[Tuple[Path, int]] = []

    for label in range(class_count):
        class_samples = by_class[label]
        if not class_samples:
            raise RuntimeError(f"Class index {label} does not contain any samples.")
        rng.shuffle(class_samples)
        total = len(class_samples)
        val_size = int(total * val_ratio)
        if val_size == 0 and total > 1:
            val_size = 1
        if val_size >= total:
            raise ValueError(
                f"val_ratio={val_ratio} left no samples for training in class index {label}."
            )
        val_samples.extend(class_samples[:val_size])
        new_train.extend(class_samples[val_size:])

    if not new_train:
        raise RuntimeError("Validation split consumed all training samples.")
    if not val_samples:
        raise RuntimeError("Validation ratio requested but produced zero validation samples.")

    return new_train, val_samples


@dataclass
class ChestXRayDataset(Dataset):
    samples: Sequence[Tuple[Path, int]]
    transform: Optional[Callable] = None

    def __post_init__(self) -> None:
        if not self.samples:
            raise RuntimeError("Attempted to initialize dataset with zero samples.")

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
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
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


def get_data_loaders_chest_xray(config):
    """Return train/val/test loaders for the chest X-ray pneumonia dataset."""
    data_root = Path(config.data_root).expanduser()
    dataset_root = _resolve_chest_xray_root(data_root)
    split_dirs = _available_split_dirs(dataset_root)

    class_names = _discover_class_names(split_dirs.values())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    train_samples = _gather_split_samples(split_dirs["train"], class_to_idx)
    val_samples: List[Tuple[Path, int]] = []
    test_samples: List[Tuple[Path, int]] = []

    seed = getattr(config, "seed", 42)
    val_ratio = getattr(config, "val_split", 0.1)

    if "val" in split_dirs:
        val_samples = _gather_split_samples(split_dirs["val"], class_to_idx)
    else:
        train_samples, val_samples = _train_val_split(
            train_samples,
            len(class_names),
            val_ratio=val_ratio,
            seed=seed,
        )

    if "test" in split_dirs:
        test_samples = _gather_split_samples(split_dirs["test"], class_to_idx)

    train_transform, eval_transform = get_transforms()

    train_dataset = ChestXRayDataset(train_samples, transform=train_transform)
    val_dataset = ChestXRayDataset(val_samples, transform=eval_transform) if val_samples else None
    test_dataset = ChestXRayDataset(test_samples, transform=eval_transform) if test_samples else None

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

    print(f"Dataset root: {dataset_root}")
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
        "name": "ChestXRay-Pneumonia",
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
        seed=42,
    )
    get_data_loaders_chest_xray(config)
