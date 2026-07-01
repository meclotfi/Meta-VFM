"""Food-11 dataloader with the same interface as the Food-101 helper."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


def _collect_images(root: Path, extensions: Sequence[str]) -> Iterable[Path]:
    """Return sorted image paths under `root` with the given extensions."""
    return sorted(
        path
        for path in root.iterdir()
        if path.suffix.lower() in extensions and path.is_file()
    )


class Food11Dataset(Dataset):
    """Dataset that infers labels from the filename prefix (e.g. '3_42.jpg')."""

    _ALLOWED_EXT = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(self, root: Path | str, transform: Optional[Callable] = None) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Folder not found: {self.root}")

        self.transform = transform
        self.samples = list(_collect_images(self.root, self._ALLOWED_EXT))
        if not self.samples:
            raise RuntimeError(f"No images found in {self.root}")

        self.labels = [self._extract_label(path.stem) for path in self.samples]
        self.class_ids = sorted(set(self.labels))

    @staticmethod
    def _extract_label(stem: str) -> int:
        try:
            return int(stem.split("_", 1)[0])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Unable to parse label from filename '{stem}'") from exc

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple:
        path = self.samples[index]
        label = self.labels[index]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


def get_data_loaders_food11(config):
    """Return train/val/test loaders and metadata for Food-11."""
    root = Path(config.data_root).expanduser() / "food-11"
    train_transform, eval_transform = get_transforms()

    train_dataset = Food11Dataset(root / "training", transform=train_transform)
    val_dataset = Food11Dataset(root / "validation", transform=eval_transform)
    test_dataset = Food11Dataset(root / "evaluation", transform=eval_transform)

    num_workers = getattr(config, "num_workers", 4)
    batch_size = config.batch_size
    test_batch_size = getattr(config, "test_batch_size", batch_size)
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

    num_classes = max(train_dataset.class_ids) + 1
    class_names = [str(idx) for idx in range(num_classes)]

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Evaluation samples: {len(test_dataset)}")
    print(f"Number of classes: {num_classes}")

    data_info = {
        "name": "Food-11",
        "num_classes": num_classes,
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
    get_data_loaders_food11(config)
