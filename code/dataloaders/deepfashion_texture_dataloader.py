"""DeepFashion texture dataloader with garment-specific splits."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


ANN_FILENAMES: Dict[str, str] = {
    "upper": "upper_fused.txt",
    "lower": "lower_fused.txt",
    "outer": "outer_fused.txt",
}


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.RandomResizedCrop(299, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


def _load_split(
    dataset_root: Path,
    garment: str,
    split: str,
) -> List[Tuple[Path, int]]:
    ann_dir = dataset_root / "texture_ann" / split
    ann_filename = ANN_FILENAMES[garment]
    ann_path = ann_dir / ann_filename
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    if split in {"train", "val"}:
        image_root = dataset_root / "train_images"
    else:
        image_root = dataset_root / "test_images"

    samples: List[Tuple[Path, int]] = []
    with ann_path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid annotation line in {ann_path}: {line}")
            rel_path, label_str = parts
            try:
                label = int(label_str)
            except ValueError as exc:
                raise ValueError(f"Invalid label value in {ann_path}: {label_str}") from exc
            image_path = image_root / rel_path
            if not image_path.exists():
                raise FileNotFoundError(f"Image listed in {ann_path} does not exist: {image_path}")
            samples.append((image_path, label))

    if not samples:
        raise RuntimeError(f"No samples parsed from {ann_path}")
    return samples


def _collect_label_set(dataset_root: Path, garment: str) -> List[int]:
    labels = set()
    for split in ["train", "val", "test"]:
        ann_path = dataset_root / "texture_ann" / split / ANN_FILENAMES[garment]
        if not ann_path.exists():
            continue
        with ann_path.open("r") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 2:
                    try:
                        labels.add(int(parts[1]))
                    except ValueError:
                        continue
    if not labels:
        raise RuntimeError(f"No labels discovered for garment '{garment}'.")
    return sorted(labels)


class DeepFashionTextureDataset(Dataset):
    """Dataset that loads (image, label) pairs for a specific garment type."""

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
        image_path, label = self.samples[index]
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_data_loaders_deepfashion_texture(config):
    """Return train/val/test loaders for the DeepFashion texture task."""
    garment = getattr(config, "garment_type", "upper").lower()
    if garment not in ANN_FILENAMES:
        raise ValueError(f"garment_type must be one of {list(ANN_FILENAMES.keys())}, got '{garment}'")

    dataset_root = Path(config.data_root).expanduser() / "deepfashion"
    if not dataset_root.exists():
        raise FileNotFoundError(f"DeepFashion root not found: {dataset_root}")

    train_samples = _load_split(dataset_root, garment, "train")
    val_samples = _load_split(dataset_root, garment, "val")
    test_samples = _load_split(dataset_root, garment, "test")

    train_transform, eval_transform = get_transforms()

    train_dataset = DeepFashionTextureDataset(train_samples, transform=train_transform)
    val_dataset = DeepFashionTextureDataset(val_samples, transform=eval_transform)
    test_dataset = DeepFashionTextureDataset(test_samples, transform=eval_transform)

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
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    unique_labels = _collect_label_set(dataset_root, garment)
    class_names = [f"{garment}_texture_{label}" for label in unique_labels]

    print(f"Garment type: {garment}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {len(class_names)}")

    data_info = {
        "name": f"DeepFashion-{garment}-texture",
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
        garment_type="upper",
    )
    get_data_loaders_deepfashion_texture(config)
