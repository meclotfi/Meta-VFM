import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
from meta_space_tools import arch_to_vector

import torch
from torch.utils.data import Dataset

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy should normally be present
    np = None  # type: ignore


def _normalize_name(name: str) -> str:
    return name.lower()


def _unique_preserve(seq: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered

def custom_collate(batch):
    # Get all possible keys from the first item
    keys = batch[0].keys()
    
    # Initialize the output dictionary
    output = {}
    
    for key in keys:
        # Skip non-tensor items or metadata
        if key in ['dataset_file_index', 'adapter_path', 'dataset_encoding_metadata', 'confusion_matrix', 'adapter_header']:
            continue
            
        # Handle tensor data
        if isinstance(batch[0][key], torch.Tensor):
            output[key] = torch.stack([item[key] for item in batch])
        elif key=="dataset_encoding":
            output[key] = torch.stack([torch.tensor(item[key]["encodings"]) for item in batch])
            output["null_arch_encoding"] = torch.stack([torch.tensor(item[key]["null_arch_encoding"]) for item in batch])
        # Handle embeddings (convert to tensor if needed)
        elif key == "embeddings":
            # trasform tensor dim from 1536 to 768,2 and then mean over the last 2 dim 

            output[key] = torch.stack([torch.tensor(item[key])-torch.tensor(item["dataset_encoding"]["null_arch_encoding"]) if not isinstance(item[key], torch.Tensor) else item[key] for item in batch])
        elif key == "architecture":
            output[key] = torch.stack([torch.tensor(arch_to_vector(item[key])[0],dtype=torch.float32) for item in batch])
        # Handle metrics (extract specific values)
        elif key == 'dataset' or key == 'model' or key == 'architecture_index':
            output[key] = [item[key] for item in batch]
        elif key == 'metrics':
            output[key] = {
                'accuracy': torch.tensor([item[key].get('accuracy', 0.0) for item in batch]),
                'f1_score': torch.tensor([item[key].get('f1_score', 0.0) for item in batch])
            }
            output['performance'] = torch.tensor([item[key].get('accuracy', 0.0) for item in batch])
            
    return output

def _extract_model_from_adapter(filename: str, dataset_name: str) -> Optional[str]:
    stem = filename
    if stem.startswith("AdaptersEncodings_"):
        stem = stem[len("AdaptersEncodings_") :]

    candidates = [
        f"_{dataset_name}_",
        f"_{dataset_name}.",
        f"_{dataset_name.upper()}_",
        f"_{dataset_name.lower()}_",
    ]

    for marker in candidates:
        if marker in stem:
            return stem.split(marker, 1)[0]
    return None


def _extract_model_from_supernet(filename: str, dataset_name: str) -> Optional[str]:
    prefix = f"Supernet_{dataset_name}_"
    if filename.startswith(prefix):
        remainder = filename[len(prefix) :]
        remainder = remainder.rsplit(".pth", 1)[0]
        if "_[" in remainder:
            remainder = remainder.split("_[", 1)[0]
        return remainder
    return None


def _to_python(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    if isinstance(value, dict):
        return {key: _to_python(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_python(item) for item in value]
    return value


def _load_torch_file(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def _load_fid_report(path: Path) -> Dict[str, Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}

    results: Dict[str, Dict[str, Any]] = {}
    for comparison in payload.get("comparisons", []):
        dataset_b = comparison.get("dataset_b")
        if not dataset_b:
            continue
        results[dataset_b] = {
            "fid": comparison.get("fid"),
            "runtime_sec": comparison.get("runtime_sec"),
            "stats_b_path": comparison.get("stats_b_path"),
        }
    return results


def _index_dataset_directory(dataset_dir: Path, dataset_name: str) -> Dict[str, Any]:
    index: Dict[str, Any] = {
        "adapters": {},
        "supernets": {},
        "stats": [],
        "encodings": [],
        "metadata": [],
        "other": [],
        "dataset_encoding": None,
        "dataset_encoding_metadata": None,
        "fid_reports": [],
    }

    for path in sorted(dataset_dir.iterdir()):
        if path.is_dir() or path.suffix == ".log":
            continue

        name = path.name

        if name.startswith("AdaptersEncodings_") and path.suffix == ".pth":
            model = _extract_model_from_adapter(name, dataset_name)
            if model:
                index["adapters"][model] = path
            else:  # fallback when parsing fails
                print("Could not extract model name from adapter:", name)
                index["other"].append(path)
        elif name.startswith("Supernet_") and path.suffix == ".pth":
            model = _extract_model_from_supernet(name, dataset_name)
            if model:
                index["supernets"][model] = path
            else:
                index["other"].append(path)

        elif name.startswith("Stats_") and path.suffix == ".npz":
            index["stats"].append(path)

        elif ("dinov3_encodings" in name.lower() or "dinov2_encodings" in name.lower()) and path.suffix == ".pth":
            index["encodings"].append(path)

        elif name.startswith("fid_batch_") and path.suffix == ".json":
            index["fid_reports"].append(path)
        elif ("dinov3_encodings" in name.lower() or "dinov2_encodings" in name.lower()) and path.suffix == ".json":
            index["metadata"].append(path)
        else:
            index["other"].append(path)

    for path in index["encodings"]:
        if "dinov3_encodings" in path.name or "dinov2_encodings" in path.name:
            index["dataset_encoding"] = path
            break

    for path in index["metadata"]:
        if "dinov3_encodings" in path.name or "dinov2_encodings" in path.name:
            index["dataset_encoding_metadata"] = path
            break

    return index


class AdapterArchitectureDataset(Dataset):
    """
    PyTorch Dataset that exposes adapter architectures and metrics across multiple datasets/models.

    Each item corresponds to a single architecture evaluation drawn from one of the
    ``AdaptersEncodings`` checkpoints.
    """

    def __init__(
        self,
        dataset_names: Optional[Sequence[str]] = None,
        model_names: Optional[Sequence[str]] = None,
        work_dir: Union[Path, str] = "work_dir",
        include_null_architecture: bool = False,
        include_predictions: bool = False,
    ) -> None:
        self.work_dir = Path(work_dir)
        if not self.work_dir.exists():
            raise FileNotFoundError(f"work_dir not found: {self.work_dir}")

        if dataset_names is None:
            dataset_names = [
                path.name for path in sorted(self.work_dir.iterdir()) if path.is_dir()
            ]
        self.dataset_names = _unique_preserve(str(name) for name in dataset_names)

        if model_names:
            self.model_filter_lookup = {
                _normalize_name(name): str(name) for name in model_names
            }
            self.model_filter = set(self.model_filter_lookup.keys())
        else:
            self.model_filter_lookup = {}
            self.model_filter = None
        self.include_null_architecture = include_null_architecture
        self.include_predictions = include_predictions

        self.dataset_file_index: Dict[str, Dict[str, Any]] = {}
        self.dataset_encodings: Dict[str, Any] = {}
        self.dataset_encoding_metadata: Dict[str, Any] = {}
        self.dataset_fids: Dict[str, Dict[str, Any]] = {}
        self.entries: List[Dict[str, Any]] = []
        self.missing_models: Dict[str, List[str]] = {}
        self.model_metrics={}

        self._build_index()

    def _build_index(self) -> None:
        for dataset in self.dataset_names:
            print(f"Indexing dataset: {dataset}")
            dataset_dir = self.work_dir / dataset
            if not dataset_dir.exists():
                raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

            index = _index_dataset_directory(dataset_dir, dataset)

            adapters = {
                model: path
                for model, path in index["adapters"].items()
                if self.model_filter is None
                or _normalize_name(model) in self.model_filter
            }

            if self.model_filter:
                available_normalized = {
                    _normalize_name(model) for model in adapters
                }
                missing = sorted(
                    self.model_filter_lookup[name]
                    for name in self.model_filter
                    if name not in available_normalized
                )
                if missing:
                    self.missing_models[dataset] = missing

            supernets = {
                model: path
                for model, path in index["supernets"].items()
                if self.model_filter is None
                or _normalize_name(model) in self.model_filter
            }

            encoding_path = index.get("dataset_encoding")
            encoding_metadata_path = index.get("dataset_encoding_metadata")

            encoding_payload = None
            if isinstance(encoding_path, Path):
                encoding_payload = _to_python(_load_torch_file(encoding_path))

            metadata_payload = None
            if isinstance(encoding_metadata_path, Path):
                metadata_payload = json.loads(encoding_metadata_path.read_text())

            self.dataset_encodings[dataset] = encoding_payload
            if isinstance(encoding_payload, dict) and "encodings" in encoding_payload:
                enc_tensor = torch.tensor(
                    encoding_payload["encodings"], dtype=torch.float32
                )
                self.dataset_encodings[dataset]["encodings"] = enc_tensor.mean(dim=0)
            self.dataset_encoding_metadata[dataset] = metadata_payload

            fid_scores: Dict[str, Dict[str, Any]] = {}
            for fid_path in index["fid_reports"]:
                fid_scores.update(_load_fid_report(fid_path))
            self.dataset_fids[dataset] = fid_scores

            dataset_paths = {
                "adapters": adapters,
                "supernets": supernets,
                "stats": index["stats"],
                "encodings": index["encodings"],
                "metadata": index["metadata"],
                "other": index["other"],
                "dataset_encoding": encoding_payload,
                "dataset_encoding_metadata": metadata_payload,
                "dataset_encoding_path": encoding_path,
                "dataset_encoding_metadata_path": encoding_metadata_path,
                "fid_reports": index["fid_reports"],
                "fid_scores": fid_scores,
            }
            self.dataset_file_index[dataset] = dataset_paths

            

            for model_key, adapter_path in adapters.items():
                adapter_payload = _load_torch_file(adapter_path)
                #print(model_key,adapter_path)
                adapter_header = {
                    "model_name": adapter_payload.get("model_name", model_key),
                    "dataset_name": adapter_payload.get("dataset_name", dataset),
                    "num_classes": adapter_payload.get("num_classes"),
                    "ranks": adapter_payload.get("ranks"),
                    "target_modules": adapter_payload.get("target_modules"),
                    "num_architectures": adapter_payload.get("num_architectures"),
                }
                

                evaluations = adapter_payload.get("evaluation_results", [])
                accuracy=[]
                for index, evaluation in enumerate(evaluations):
                    #print(evaluation.keys())
                    arch_name = evaluation.get("arch_name")
                    if (arch_name == "NULL_ARCHITECTURE"):
                        self.dataset_encodings[dataset]["null_arch_encoding"]=evaluation.get("embeddings")
                        continue
                    

                    metrics = _to_python(evaluation.get("metrics", {}))
                    architecture = _to_python(evaluation.get("architecture", {}))
                    confusion = evaluation.get("confusion_matrix")
                    embeddings = evaluation.get("embeddings")
                    self.model_metrics[(index,dataset)]=metrics
                    accuracy.append(metrics.get("accuracy",0.0))

                    if embeddings is None:
                        print(
                            f"Warning: Missing embeddings for {dataset} / {model_key} / {arch_name} / {index}"
                        )
                    item: Dict[str, Any] = {
                        "dataset": dataset,
                        "model": adapter_header["model_name"],
                        "adapter_model_key": model_key,
                        "arch_name": arch_name,
                        "architecture_index": index,
                        "architecture": architecture,
                        "metrics": metrics,
                        "num_trainable": metrics.get("num_trainable"),
                        "num_total": metrics.get("num_total"),
                        "embeddings": _to_python(embeddings)
                        if embeddings is not None
                        else None,
                        "adapter_path": adapter_path,
                        "dataset_file_index": dataset_paths,
                        "dataset_encoding": self.dataset_encodings.get(dataset),
                        "dataset_encoding_metadata": self.dataset_encoding_metadata.get(
                            dataset
                        ),
                        "fid_scores": self.dataset_fids.get(dataset, {}),
                        "confusion_matrix": _to_python(confusion)
                        if confusion is not None
                        else None,
                        "adapter_header": adapter_header,
                    }

                    if self.include_predictions and "predictions" in evaluation:
                        item["predictions"] = _to_python(evaluation["predictions"])

                    self.entries.append(item)
                # append min max and mean acc to fid scores of each dataset
                if len(accuracy)>0:
                    self.dataset_fids[dataset]["range"]={
                        "min_accuracy": min(accuracy),
                        "max_accuracy": max(accuracy),
                        "mean_accuracy": sum(accuracy)/len(accuracy)
                    }

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, Any]:  # type: ignore[override]
        return self.entries[index]

    def available_models(self, dataset: Optional[str] = None) -> Dict[str, List[str]]:
        if dataset:
            paths = self.dataset_file_index.get(dataset, {})
            adapters = paths.get("adapters", {})
            return {dataset: sorted(adapters.keys())}
        return {
            dataset_name: sorted(paths["adapters"].keys())
            for dataset_name, paths in self.dataset_file_index.items()
        }
    
    def get_model_metrics(self, dataset: str, model: int) -> List[Dict[str, Any]]:
        return self.model_metrics.get((model,dataset),{})
    def compute_acc_matrix(self,batch_items,scaler=0.8):
        n = len(batch_items["dataset"])
        acc_matrix = torch.zeros((n, n))
        for i in range(n):
            for j in range(n):
                acc_value=self.get_model_metrics(batch_items["dataset"][j], batch_items["architecture_index"][i])["accuracy"]/100.0
                min_acc= self.dataset_fids.get(batch_items["dataset"][j], {}).get("range",{}).get("min_accuracy",0.0)/100.0
                max_acc= self.dataset_fids.get(batch_items["dataset"][j], {}).get("range",{}).get("max_accuracy",1.0)/100.0
                #
                acc_value_norm = (acc_value - min_acc) / (max_acc - min_acc + 1e-8) if (max_acc - min_acc) > 0 else acc_value
                range_min=self.dataset_fids.get(batch_items["dataset"][j], {}).get("range",{}).get("min_accuracy",0.0)/100.0
                if batch_items["dataset"][j] == batch_items["dataset"][i]:
                    acc_matrix[i, j] = acc_value
                else:
                    #acc_matrix[i, j]=0
                    fid_alpha= (1.0 - min(self.dataset_fids.get(batch_items["dataset"][j], {}).get(batch_items["dataset"][i], {}).get("fid", 100.0), 100.0)/100.0)
                    acc_matrix[i, j] =  scaler * fid_alpha * acc_value_norm
                    
                    #fid_power= int(min(self.dataset_fids.get(batch_items["dataset"][j], {}).get(batch_items["dataset"][i], {}).get("fid", 100.0), 100.0)/2)
                    #acc_matrix[i, j] = fid_alpha * max((acc_value**fid_power), 1e-3)
        return acc_matrix
    
    def get_fid_scores(self, batch_items) -> Dict[str, Dict[str, Any]]:
        # 2d matrix of fid scores with ignoring duplicates
        # Build matrix only for unique datasets (preserve order)
        unique_datasets = []
        seen = set()
        index_unique_datasets=[]
        for i,d in enumerate(batch_items["dataset"]):
            if d not in seen:
                seen.add(d)
                unique_datasets.append(d)
                index_unique_datasets.append(i)

        n = len(unique_datasets)
        fid_matrix = torch.zeros((n, n))
        for i, dataset_a in enumerate(unique_datasets):
            for j, dataset_b in enumerate(unique_datasets):
                if dataset_a == dataset_b:
                    fid_matrix[i, j] = 0.0
                else:
                    fid_matrix[i, j] = self.dataset_fids.get(dataset_a, {}).get(dataset_b, {}).get("fid", -1.0)

        return fid_matrix,unique_datasets


__all__ = ["AdapterArchitectureDataset"]
