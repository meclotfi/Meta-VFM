# Compute & Hardware Details

Source: Supplementary Material, Section 1.5 (CVPR 2026 Submission #18890).

## Supernet Training (Adapter Zoo Construction)

| Item | Details |
|------|---------|
| Node | Single node |
| GPUs | 8× AMD Instinct MI210 (CDNA2 architecture) |
| GPU memory | 64 GB HBM2e per GPU → **512 GB total** |
| Avg. compute per backbone | 2 GPU-hours |
| Total configurations | 5 backbone variants × 91 datasets |
| Total GPU-hours | 2 × 5 × 91 = **910 GPU-hours** (~37.9 GPU-days) |
| Wall-clock (8 GPUs parallel) | ~114 hours (**~4.7 days**) |

### Backbones trained
- DINOv3-S
- DINOv3-S+
- DINOv3-B
- DINOv2-S
- DINOv2-B

## Meta-Space Construction

| Item | Details |
|------|---------|
| Hardware | MacBook Pro (Apple M3 chip) |
| Total compute time | ~1 hour |

## Super-Adapter Parameter Count (ViT-B reference)

| Configuration | Params |
|--------------|--------|
| Fixed r=32 LoRA per site | 49,152 |
| Super-Adapter (ranks {2,4,8,16,32}) | 50,512 |
| Overhead vs fixed LoRA | ~2.8% |

- 6 target matrices per ViT block: Wq, Wk, Wv, Wo, Wffn1, Wffn2
- 12 blocks in ViT-B → **72 adapter insertion sites**
- Rank search space: r ∈ {0, 2, 4, 8, 16, 32}
- Theoretical search space: 6^72 ≈ 10^56; restricted to ~10^30 during sampling

## Training Optimizer

- Adam, lr = 1e-4, cosine annealing
- Monitored metrics: MSE, MAE, Spearman ρ, Kendall-τ

## Pre-computed Backbone Checkpoint

- `dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth` (~2.1 GB, in parent folder)
