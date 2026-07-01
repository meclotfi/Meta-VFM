"""Fruits360 dataloader using the original-size train/val/test splits."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
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


def _gather_image_paths(split_dir: Path) -> List[Tuple[Path, str]]:
    samples: List[Tuple[Path, str]] = []
    if not split_dir.exists():
        return samples
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        class_name = class_dir.name
        for image_path in sorted(class_dir.glob("*")):
            if image_path.suffix.lower() in ALLOWED_EXTENSIONS and image_path.is_file():
                samples.append((image_path, class_name))
    return samples


class Fruits360Dataset(Dataset):
    """Dataset wrapper that infers labels from directory names."""

    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        class_names: List[str],
        transform: Optional[Callable] = None,
    ) -> None:
        if not samples:
            raise RuntimeError("Attempted to build Fruits360Dataset with zero samples.")
        self.samples = samples
        self.class_names = class_names
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def _index_samples(
    raw_samples: Iterable[Tuple[Path, str]],
    class_to_idx: Dict[str, int],
) -> List[Tuple[Path, int]]:
    indexed: List[Tuple[Path, int]] = []
    for path, name in raw_samples:
        if name not in class_to_idx:
            raise ValueError(f"Class '{name}' not found in reference class list.")
        indexed.append((path, class_to_idx[name]))
    return indexed


def get_data_loaders_fruits360(config):
    """Return train/val/test loaders for the Fruits360 dataset."""
    base_root = Path(config.data_root).expanduser() / "Fruits360" / "fruits-360-original-size"
    train_dir = base_root / "Training"
    val_dir = base_root / "Validation"
    test_dir = base_root / "Test"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Expected training directory at {train_dir}. "
            "Make sure Fruits360 is extracted under 'Fruits360/fruits-360-original-size'."
        )

    train_transform, eval_transform = get_transforms()

    train_raw = _gather_image_paths(train_dir)
    if not train_raw:
        raise RuntimeError(f"No training images found under {train_dir}")

    class_names = sorted({name for _, name in train_raw})
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    train_samples = _index_samples(train_raw, class_to_idx)
    val_samples = _index_samples(_gather_image_paths(val_dir), class_to_idx) if val_dir.exists() else []
    test_samples = _index_samples(_gather_image_paths(test_dir), class_to_idx) if test_dir.exists() else []

    train_dataset = Fruits360Dataset(train_samples, class_names, transform=train_transform)
    val_dataset = Fruits360Dataset(val_samples, class_names, transform=eval_transform) if val_samples else None
    test_dataset = Fruits360Dataset(test_samples, class_names, transform=eval_transform) if test_samples else None

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
        "name": "Fruits360",
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
    )
    get_data_loaders_fruits360(config)
