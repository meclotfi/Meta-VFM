"""Intel Image Scene Classification dataloader with train/val/test loaders."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


ALLOWED_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")


def _iter_class_folders(root: Path) -> Iterable[Path]:
    for path in sorted(root.iterdir()):
        if path.is_dir():
            yield path


def _collect_split(root: Path) -> Tuple[List[Tuple[Path, int]], List[str]]:
    class_names = [folder.name for folder in _iter_class_folders(root)]
    if not class_names:
        raise RuntimeError(f"No class folders found in {root}")

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


@dataclass
class IntelImageDataset(Dataset):
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
        transforms.RandomRotation(10),
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


def get_data_loaders_intel_image(config):
    """Return dataloaders for the Intel Image Scene Classification dataset."""
    root = Path(config.data_root).expanduser() / "Intel-image"
    train_dir = root / "seg_train"
    val_dir = root / "seg_test"  # provided validation split
    test_dir = root / "seg_pred"  # unlabeled predictions folder

    train_transform, eval_transform = get_transforms()

    train_samples, class_names = _collect_split(train_dir)
    val_samples, _ = _collect_split(val_dir)

    train_dataset = IntelImageDataset(train_samples, class_names, transform=train_transform)
    val_dataset = IntelImageDataset(val_samples, class_names, transform=eval_transform)

    test_dataset = None
    if test_dir.exists():
        test_images = sorted(
            path for path in test_dir.iterdir()
            if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
        )
        if test_images:
            # Use dummy label -1 because ground truth is not provided
            test_dataset = IntelImageDataset(
                [(path, -1) for path in test_images],
                class_names,
                transform=eval_transform,
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
    val_loader = DataLoader(
        val_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
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
    print(f"Validation samples: {len(val_dataset)}")
    if test_dataset:
        print(f"Unlabeled prediction samples: {len(test_dataset)}")
    else:
        print("Unlabeled prediction samples: 0 (folder empty or missing)")
    print(f"Number of classes: {len(class_names)}")

    data_info = {
        "name": "Intel Image Scene Classification",
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
    )
    get_data_loaders_intel_image(config)
