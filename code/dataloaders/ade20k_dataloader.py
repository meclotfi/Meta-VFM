"""ADE20K (DatasetNinja export) polygon-segmentation loader.

Dataset layout under ``root`` (e.g. ``~/Downloads/ade20k-DatasetNinja``):
    - meta.json: class list with ids/titles/colors
    - training/
         - img/*.jpg (RGB images)
         - ann/*.jpg.json (per-image polygon annotations)
    - validation/ (same structure as training)

Each annotation JSON contains:
    - size: {"height": H, "width": W}
    - objects: list with {"classId", "classTitle", "geometryType": "polygon",
      "points": {"exterior": [[x, y], ...], "interior": [[[x, y], ...], ...]}}

This loader returns the class titles as text (no numeric encoding) plus one
binary mask per annotated object.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader, Dataset

Split = str
Polygon = Sequence[Sequence[float]]


@dataclass(frozen=True)
class ADE20KSample:
    image_path: Path
    ann_path: Path
    size: Optional[Tuple[int, int]]  # (height, width)


def _load_class_lookup(meta_path: Path) -> Dict[int, str]:
    with meta_path.open("r") as handle:
        meta = json.load(handle)
    classes = meta.get("classes", [])
    return {int(entry["id"]): entry["title"] for entry in classes if "id" in entry and "title" in entry}


def _polygon_to_mask(
    polygon: Polygon,
    holes: Iterable[Polygon],
    size: Tuple[int, int],
) -> torch.Tensor:
    """Rasterize polygon with optional holes into a boolean mask."""
    height, width = size
    mask_img = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    draw.polygon([tuple(pt) for pt in polygon], outline=1, fill=1)
    for hole in holes:
        draw.polygon([tuple(pt) for pt in hole], outline=0, fill=0)
    mask = np.array(mask_img, dtype=np.uint8)
    return torch.from_numpy(mask.astype(bool))


class ADE20KDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: Split = "training",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root).expanduser()
        if split not in {"training", "validation"}:
            raise ValueError(f"split must be 'training' or 'validation', got {split}")
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        self.images_dir = self.root / split / "img"
        self.anns_dir = self.root / split / "ann"
        if not self.images_dir.exists() or not self.anns_dir.exists():
            raise FileNotFoundError(f"Could not find expected split folders at {self.images_dir} and {self.anns_dir}")

        self.class_id_to_title = _load_class_lookup(self.root / "meta.json")
        self.samples: List[ADE20KSample] = []
        for ann_path in sorted(self.anns_dir.glob("*.json")):
            stem = ann_path.stem  # e.g. ADE_frame_00000001.jpg
            image_path = self.images_dir / stem
            if not image_path.exists():
                # Skip orphan annotations but keep going.
                continue
            self.samples.append(ADE20KSample(image_path=image_path, ann_path=ann_path, size=None))
        if not self.samples:
            raise RuntimeError(f"No paired images/annotations found under {self.root}/{split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        with sample.ann_path.open("r") as handle:
            ann = json.load(handle)
        image = Image.open(sample.image_path).convert("RGB")

        size = ann.get("size", {}) if isinstance(ann, dict) else {}
        height, width = size.get("height"), size.get("width")
        if height is None or width is None:
            width, height = image.size
        resolved_size = (height, width)

        class_titles: List[str] = []
        masks: List[torch.Tensor] = []
        polygons: List[Polygon] = []
        interiors: List[List[Polygon]] = []

        for obj in ann.get("objects", []):
            points = obj.get("points", {})
            polygon: Polygon = points.get("exterior", [])
            holes: List[Polygon] = points.get("interior", []) or []
            if not polygon:
                continue
            polygons.append(polygon)
            interiors.append(holes)
            class_title = obj.get("classTitle")
            if not class_title and "classId" in obj:
                class_title = self.class_id_to_title.get(int(obj["classId"]), "unknown")
            if not class_title:
                class_title = "unknown"
            class_titles.append(class_title)
            masks.append(_polygon_to_mask(polygon, holes, resolved_size))

        target = {
            "class_titles": class_titles,
            "masks": masks,  # List[torch.bool] masks aligned with class_titles
            "polygons": polygons,
            "size": resolved_size,
            "image_path": sample.image_path,
            "ann_path": sample.ann_path,
        }

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target


def ade20k_collate(batch):
    """Keeps variable-length masks/labels intact."""
    images, targets = zip(*batch)
    return list(images), list(targets)


def make_dataloader(
    root: str | Path,
    split: Split = "training",
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 0,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
) -> DataLoader:
    dataset = ADE20KDataset(
        root=root,
        split=split,
        transform=transform,
        target_transform=target_transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=ade20k_collate,
    )


if __name__ == "__main__":
    # Minimal sanity check that masks/classes load.
    root_dir = Path("~/Downloads/ade20k-DatasetNinja").expanduser()
    ds = ADE20KDataset(root=root_dir, split="training")
    img, target = ds[0]
    print(f"Loaded image {target['image_path'].name} with {len(target['class_titles'])} objects")
    print("First classes:", target["class_titles"][:5])
