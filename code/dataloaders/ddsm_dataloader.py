"""DDSM mammography dataloader pulling training TFRecords into PyTorch loaders."""
from __future__ import annotations

import math
import numpy as np
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


def get_transforms():
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((320, 320)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((320, 320)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


def _available_folds(root: Path) -> List[int]:
    folds: List[int] = []
    for path in root.glob("training10_*"):
        if path.is_dir():
            try:
                fold_idx = int(path.name.split("_")[-1])
                folds.append(fold_idx)
            except ValueError:
                continue
    return sorted(folds)


def _decode_filename(raw: bytes) -> str:
    if not raw:
        return ""
    for encoding in ("utf-16", "utf-16le", "utf-8"):
        try:
            decoded = raw.decode(encoding).strip("\x00")
            if decoded:
                return decoded
        except UnicodeDecodeError:
            continue
    return raw.decode("latin1", errors="ignore").strip("\x00")


def _read_single_tfrecord(
    tfrecord_path: Path,
    label_key: str,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "TensorFlow is required to read the DDSM TFRecord files. "
            "Install it with `pip install tensorflow`."
        ) from exc

    if not tfrecord_path.exists():
        raise FileNotFoundError(f"TFRecord file not found: {tfrecord_path}")

    dataset = tf.data.TFRecordDataset(str(tfrecord_path))
    images: List[np.ndarray] = []
    labels: List[int] = []

    for raw_record in dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        features = example.features.feature

        if label_key not in features:
            raise KeyError(
                f"Label key '{label_key}' missing in TFRecord {tfrecord_path}"
            )
        label = int(features[label_key].int64_list.value[0])
        image_bytes = features["image"].bytes_list.value[0]

        flat = np.frombuffer(image_bytes, dtype=np.uint8)
        side = int(math.isqrt(flat.size))
        if side * side != flat.size:
            raise ValueError(
                f"Unexpected image length ({flat.size}) in {tfrecord_path}; "
                "cannot reshape into square."
            )
        image = flat.reshape(side, side)
        images.append(image)
        labels.append(label)

    if not images:
        raise RuntimeError(f"No samples read from TFRecord {tfrecord_path}")

    return np.stack(images), np.asarray(labels, dtype=np.int64)


def _load_training_folds(
    root: Path,
    folds: Sequence[int],
    label_key: str,
    use_cache: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    processed_dir = root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    fold_tag = "all" if not folds else "_".join(map(str, sorted(folds)))
    cache_name = f"train_{label_key}_{fold_tag}.npz"
    cache_path = processed_dir / cache_name

    if use_cache and cache_path.exists():
        cached = np.load(cache_path)
        return cached["images"], cached["labels"]

    images_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []

    for fold in folds:
        tfrecord_path = root / f"training10_{fold}" / f"training10_{fold}.tfrecords"
        fold_images, fold_labels = _read_single_tfrecord(tfrecord_path, label_key)
        images_list.append(fold_images)
        labels_list.append(fold_labels)

    images = np.concatenate(images_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    if use_cache:
        np.savez_compressed(cache_path, images=images, labels=labels)

    return images, labels


def _load_numpy_images(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Numpy array not found: {path}")
    data = np.load(path, allow_pickle=True)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    return data.astype(np.uint8)


class DDSMArrayDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: Optional[Callable] = None,
    ) -> None:
        if images.shape[0] != labels.shape[0]:
            raise ValueError("Images and labels must have the same length.")
        self.images = images
        self.labels = labels.astype(np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int):
        image = self.images[index]
        if self.transform:
            image = self.transform(image)
        else:
            tensor_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ])
            image = tensor_transform(image)
        label = int(self.labels[index])
        return image, label


def get_data_loaders_ddsm(config):
    """Build PyTorch DataLoaders for the DDSM dataset stored as TFRecords."""
    root = Path(config.data_root).expanduser() / "DDSM"
    if not root.exists():
        raise FileNotFoundError(f"DDSM root folder not found at {root}")

    available = _available_folds(root)
    if not available:
        raise RuntimeError(f"No training folds found under {root}")

    requested_folds = getattr(config, "folds", "all")
    if requested_folds == "all" or requested_folds is None:
        folds = available
    elif isinstance(requested_folds, int):
        folds = [requested_folds]
    else:
        folds = list(requested_folds)

    missing = sorted(set(folds) - set(available))
    if missing:
        raise ValueError(f"Requested folds {missing} are not available (have {available})")

    label_key = "label_normal" if getattr(config, "use_label_normal", False) else "label"
    use_cache = getattr(config, "use_cache", True)

    train_images, train_labels = _load_training_folds(root, folds, label_key, use_cache)

    val_strategy = getattr(config, "val_strategy", "cv")
    if val_strategy == "cv":
        val_images = _load_numpy_images(root / "cv10_data" / "cv10_data.npy")
        val_labels = np.load(root / "cv10_labels.npy")
    elif val_strategy == "split":
        val_ratio = getattr(config, "val_split", 0.1)
        seed = getattr(config, "seed", 42)
        indices = np.arange(train_images.shape[0])
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
        val_size = max(1, int(len(indices) * val_ratio))
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]
        val_images = train_images[val_idx]
        val_labels = train_labels[val_idx]
        train_images = train_images[train_idx]
        train_labels = train_labels[train_idx]
    else:
        raise ValueError(f"Unknown val_strategy '{val_strategy}' (use 'cv' or 'split').")

    test_images = _load_numpy_images(root / "test10_data" / "test10_data.npy")
    test_labels = np.load(root / "test10_labels.npy")

    train_transform, eval_transform = get_transforms()

    train_dataset = DDSMArrayDataset(train_images, train_labels, transform=train_transform)
    val_dataset = DDSMArrayDataset(val_images, val_labels, transform=eval_transform)
    test_dataset = DDSMArrayDataset(test_images, test_labels, transform=eval_transform)

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

    class_ids = sorted({int(x) for x in np.unique(np.concatenate([
        train_labels,
        val_labels,
        test_labels,
    ]))})
    class_names = [str(cls) for cls in class_ids]

    print("DDSM dataset")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {len(class_names)}")

    data_info = {
        "name": "DDSM",
        "num_classes": len(class_names),
        "class_names": class_names,
    }

    return train_loader, val_loader, test_loader, data_info


if __name__ == "__main__":
    from types import SimpleNamespace

    config = SimpleNamespace(
        data_root="./data",
        batch_size=16,
        test_batch_size=32,
        num_workers=4,
        pin_memory=True,
        folds="all",
        use_label_normal=False,
        use_cache=True,
        val_strategy="cv",
        val_split=0.1,
        seed=42,
    )
    get_data_loaders_ddsm(config)
