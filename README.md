# MetaVFM — Meta-Learned Efficient Adaptation of Vision Foundation Models

**Paper:** MetaVFM: Meta-Learned Efficient Adaptation of Vision Foundation Models for Medical Imaging
**Venue:** CVPR 2026 (Submission #18890 — Confidential. Do not distribute.)

---

## Overview

MetaVFM is a meta-learning framework that predicts adapter placement, rank, and initialization for unseen datasets with no per-task hyperparameter search. It learns a **meta-space** from a shared-weight **Super-Adapter** trained on 90+ datasets, enabling fast retrieval of an adapter configuration and initialization that speeds fine-tuning convergence by **2–3×** compared to standard PEFT baselines.

**Key contributions:**
- **Super-Adapter**: A shared-weight supernet unifying multi-rank LoRA across all insertion sites of a ViT (72 sites for ViT-B, ranks r ∈ {0, 2, 4, 8, 16, 32}).
- **Adapter Zoo**: Large-scale zoo of Super-Adapters trained on 91 diverse datasets across 5 ViT backbones (DINOv2-S/B, DINOv3-S/S+/B).
- **Meta-Adapter Space**: Joint dataset–adapter embedding space aligned by performance (MSE + ranking + FID losses), enabling retrieval-based initialization.

---

## Repository Structure

```
MetaVFM-release/
├── README.md                    ← this file
├── .gitignore                   ← excludes checkpoints, weights, configs, large data files
│
├── code/
│   ├── dataloaders/             ← Dataset-specific PyTorch dataloaders (19 datasets)
│   ├── meta_space/              ← Meta-space training and inference
│   │   ├── meta_space_tools.py  ← Core meta-space utilities
│   │   ├── train_meta-space.py  ← Meta-space training script
│   │   └── inference_metaspace.ipynb  ← Inference / retrieval demo
│   ├── training/
│   │   └── multi_dataset_adapter_dataset.py  ← Multi-dataset adapter dataset loader
│   └── utils/
│       ├── download_*.py        ← Dataset download scripts
│       └── move_datasets*.sh    ← Dataset organization scripts
│
├── assets/
│   ├── meta_space.png           ← Meta-space visualization
│   ├── meta_space_mainplot.png  ← Main meta-space plot (color-coded)
│   └── tsne_final.png           ← Final T-SNE of the learned meta-space (trial 7, epoch 9)
│
├── datasets/
│   └── DATASET_CATALOG.md       ← Full 91-dataset catalog with sizes, classes, and domains
│
└── docs/
    ├── COMPUTE_INFO.md          ← Hardware specs, GPU-hours, wall-clock times
    └── CHECKPOINTS.md           ← Checkpoint file descriptions and release plan

> Checkpoints, model weights, configs, and embeddings are excluded via .gitignore
> and will be released separately upon paper acceptance.
```

---

## Method Summary

### 1. Super-Adapter (Shared-Weight Supernet)

Multi-rank LoRA inserted into 6 matrices per ViT block (Wq, Wk, Wv, Wo, Wffn1, Wffn2):

```
y = (W + α · ΔW^(k)) x,   ΔW^(k) := B1 ··· Bk · Ak ··· A1
```

Selecting prefix k gives a LoRA of rank r_k. After training, factors are collapsed into a standard (A^(k), B^(k)) pair — zero inference overhead vs. a plain LoRA.

### 2. Five-Phase Training Curriculum

1. **Individual Rank (Stabilization)** — single global rank, no dropout
2. **Regularized Rank** — dropout + stochastic depth enabled
3. **Mix Ranks** — heterogeneous ranks across sites
4. **Mix Targets** — vary active target matrices per block
5. **Sandwich Rule** — min + max + random architecture per batch

### 3. Meta-Space Training

Composite loss: `L = L_perf + L_rank + L_FID`

- **L_perf**: MSE regression of validation accuracy from (dataset_emb, adapter_emb)
- **L_rank**: Pairwise logistic loss preserving performance ordering
- **L_FID**: FID-weighted cross-dataset regularizer for embedding smoothness

### 4. Inference

Given new dataset D_new:
1. Compute dataset embedding u_new (mean-pooled DINOv3 features + PCA + learned encoder)
2. Retrieve top-k adapters by cosine similarity within parameter budget B
3. Initialize fine-tuning from the highest-scoring adapter (structure + weights)
4. Fine-tune for fixed T steps

---

## Results (100-epoch, ViT-B/DINOv2)

| Method | Fine-grained | General | Scene/Sat | Medical | Total Avg |
|--------|-------------|---------|-----------|---------|-----------|
| LoRA | 88.5 | 90.4 | 86.8 | 93.8 | 90.1 |
| APLA | 89.4 | 92.1 | 88.2 | 95.1 | 91.6 |
| **MetaVFM-T1** | 89.5 | 91.7 | 89.3 | 94.8 | **91.6** |
| **MetaVFM-T5** | **89.9** | **92.0** | **89.3** | **95.2** | **91.8** |

Evaluated on 17 unseen datasets (none overlap with meta-space training).

---

## Compute Requirements

See `docs/COMPUTE_INFO.md` for full details.

- **Supernet training**: 8× AMD Instinct MI210 (64 GB HBM2e each) → 910 GPU-hours total, ~4.7 days wall-clock
- **Meta-space construction**: MacBook Pro (Apple M3), ~1 hour

---

## Dataloaders Available

| File | Dataset |
|------|---------|
| `ade20k_dataloader.py` | ADE20K (segmentation) |
| `aptos_dataloader.py` | APTOS 2019 (diabetic retinopathy) |
| `caltech256_dataloader.py` | Caltech-256 |
| `chest_xray_dataloader.py` | Chest X-Ray |
| `compcars_dataloader.py` | CompCars |
| `ddsm_dataloader.py` | DDSM (mammography) |
| `deepfashion_texture_dataloader.py` | DeepFashion (texture attributes) |
| `deepweed_dataloader.py` | DeepWeeds |
| `food11_dataloader.py` | Food-11 |
| `fruits360_dataloader.py` | Fruits-360 |
| `human_action_recognition_dataloader.py` | Human Action Recognition |
| `hyperkvasir_dataloader.py` | HyperKvasir (endoscopy) |
| `intel_image_dataloader.py` | Intel Image Classification |
| `ketahk_texture_dataloader.py` | KETAHK Texture |
| `lung_colon_dataloader.py` | LC25000 (Lung/Colon histology) |
| `nct_crc_dataloader.py` | NCT-CRC-HE-100K (colorectal histology) |
| `sipakmed_dataloader.py` | SIPaKMeD (cervical cytology) |
| `stanford_online_products_dataloader.py` | Stanford Online Products |
| `uecfood100_dataloader.py` | UECFood-100 |

---

## Citation

```
@inproceedings{metavfm2026,
  title     = {MetaVFM: Meta-Learned Efficient Adaptation of Vision Foundation Models for Medical Imaging},
  author    = {Anonymized Authors},
  booktitle = {CVPR},
  year      = {2026},
  note      = {Submission \#18890 — Confidential}
}
```
