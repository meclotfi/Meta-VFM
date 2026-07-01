# Checkpoint Structure

## Meta-Space Trials (`checkpoints/meta_space_trials/`)

Seven training trials of the meta-space model, each containing:

| File | Description |
|------|-------------|
| `best_0.pth` | Best meta-space model weights |
| `best_config_0.json` | Config for the best checkpoint |
| `meta-space_0_0.pth` | Epoch checkpoint |
| `meta-space_0_config_0.json` | Epoch config |
| `dataset_embeddings_embeddings.pt` | Pre-computed dataset embeddings (torch tensor) |
| `dataset_embeddings_fid_matrix.npy` | Pairwise FID matrix across datasets |
| `dataset_embeddings_metadata.json` | Dataset metadata registry |
| `dataset_embeddings_sklearn_models.pkl` | PCA / sklearn models used for embedding |
| `tsne_epoch_0.png` | T-SNE visualization of meta-space at that epoch |

## Shared Embeddings (`checkpoints/embeddings/`)

These are the root-level pre-computed embeddings used across all trials:

| File | Description |
|------|-------------|
| `dataset_embeddings_embeddings.pt` | Mean-pooled frozen DINOv3 features (PCA-compressed) |
| `dataset_embeddings_fid_matrix.npy` | Pairwise FID distances between all datasets |
| `dataset_embeddings_metadata.json` | JSON registry: dataset names, sizes, splits |
| `dataset_embeddings_sklearn_models.pkl` | PCA transform and sklearn preprocessing models |

## Adapter Zoo Checkpoints (not included — to be released)

Per the paper, upon acceptance the following will be released for each backbone:

- **Super-Adapter Checkpoints**: Fully trained weight-sharing supernetworks
- **Distilled Checkpoints**: Top-k (k ∈ {1, 5, 10}) standard LoRA checkpoints per dataset and parameter budget
- **Metadata**: JSON registries with Pareto-optimal ranks/placements, training budgets, and validation metrics

## Model Results (`../model_results/` in parent folder)

Evaluation results per dataset (top-1 accuracy logs):
- AID, NABirds, APTOS-2019, Caltech-256, CIFAR-100, CUB-200, DTD, FGVC-Aircraft, MIT-Indoor, Stanford Dogs
