


"""
Meta-space Encoders and Performance Predictor
Implements: Model Encoder, Dataset Encoder, and Performance Predictor
Based on MedNNS architecture for model-dataset matching in meta-space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, List
from enum import Enum

def arch_to_vector(arch_rep):
    """
    Transform architecture representation to vector representation.
    
    For each layer, creates a vector where each element represents the rank
    assigned to a specific target module. If a module is in target_modules,
    it gets the layer's rank; otherwise, it gets 0.
    
    Args:
        arch_rep (dict): Dictionary mapping layer indices to configurations
                        with 'rank' and 'target_modules' keys
    
    Returns:
        dict: Dictionary mapping layer indices to vectors of ranks for all possible modules
    """
    # Define all possible target modules
    all_possible_modules = ['query', 'key', 'value', 'dense', 'fc1', 'fc2']
    
    vector_rep = {}
    
    for layer_idx, config in arch_rep.items():
        rank = config['rank']
        target_modules = config['target_modules']
        
        # Create vector where each module gets rank if it's in target_modules, else 0
        layer_vector = [
            rank if module in target_modules else 0
            for module in all_possible_modules
        ]
        
        vector_rep[layer_idx] = layer_vector
    
    vector_representation=[]
    for i in sorted(vector_rep.keys()):
        vector_representation.extend(vector_rep[i])
    vector_rep_list=vector_representation
    vector_rep_dict=vector_rep
    return vector_rep_list, vector_rep_dict



def print_vector_representation(vector_rep, module_names=None):
    """
    Pretty print the vector representation with module names as headers.
    
    Args:
        vector_rep (dict): Output from arch_to_vector function
        module_names (list): Names of modules for header (optional)
    """
    if module_names is None:
        module_names = ['query', 'key', 'value', 'dense', 'fc1', 'fc2']
    
    # Print header
    print("Layer | " + " | ".join(f"{name:^6}" for name in module_names))
    print("-" * (7 + len(module_names) * 9))
    
    # Print each layer's vector
    for layer_idx in sorted(vector_rep.keys()):
        vector = vector_rep[layer_idx]
        print(f"{layer_idx:^5} | " + " | ".join(f"{val:^6}" for val in vector))


class LossType(Enum):
    """Enumeration of available loss combinations."""
    MSE_ONLY = "mse_only"
    MSE_RANK_ADJACENT = "mse_rank_adjacent"
    MSE_RANK_CONTRASTIVE = "mse_rank_contrastive"
    MSE_ALIGNMENT = "mse_alignment"
    MSE_ALIGNMENT_DIVERSITY = "mse_alignment_diversity"
    MSE_RANK_WEIGHTED = "mse_rank_weighted"
    TRIPLET = "triplet"
    COMBINED_ALL = "combined_all"


class ModelEncoder(nn.Module):
    """
    Embed a candidate model (architecture + functional features) into meta-space.
    
    Inputs:
        - Architecture features a ∈ R^d_a (depth, width, FLOPs, params, etc.)
        - Functional features f ∈ R^d_f (penultimate activations from probe)
    
    Output:
        - Model embedding e_m ∈ R^D (L2-normalized)
    """
    
    def __init__(self, d_a: int, d_f: int, D: int = 256, hidden_dim: int = 512):
        """
        Args:
            d_a: Dimension of architecture features
            d_f: Dimension of functional features
            D: Output embedding dimension (default 256)
            hidden_dim: Hidden layer dimension (default 1024)
        """
        super().__init__()
        
        self.d_a = d_a
        self.d_f = d_f
        self.D = D
        
        #linear projection before concatenation
        self.arch_proj = nn.Linear(d_a, 512)
        self.func_proj = nn.Linear(d_f, d_f//2)
        self.func_up_proj = nn.Linear(d_f//2, d_f//2)
        self.null_proj = nn.Linear(d_f, d_f//4)
        self.shared_func_proj = nn.Linear(d_f//2, 512)



        # Concatenated input dimension
        input_dim = 1024  # d_a + d_f after projections
        
        # MLP: [d_a+d_f → 1024 → 512 → D]
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, D)
        )
    
    def forward(self, arch_features: torch.Tensor, func_features: torch.Tensor, null_arch_features : torch.Tensor, use_null=False) -> torch.Tensor:
        """
        Forward pass to encode a model into meta-space.
        
        Args:
            arch_features: Architecture features [batch_size, d_a]
            func_features: Functional features [batch_size, d_f]
        
        Returns:
            e_m: Model embeddings [batch_size, D] (L2-normalized)
        """
        # normaize arch and func features
        #arch_features = F.normalize(arch_features, p=2, dim=1)
        #func_features = F.normalize(func_features, p=2, dim=1)
        # linear projections
        arch_features = self.arch_proj(arch_features)
        func_features = self.func_proj(func_features)
        if use_null:
            null_arch_features = self.null_proj(null_arch_features)
            concat_func_features = torch.cat([func_features, null_arch_features], dim=1)
        else:
            concat_func_features = self.func_up_proj(F.relu(func_features))

        func_features = self.shared_func_proj(concat_func_features)
        # Concatenate features
        x = torch.cat([arch_features, func_features], dim=1)  # [batch_size, d_a+d_f]
        
        # Pass through MLP
        e_m = self.mlp(x)  # [batch_size, D]
        
        # L2 normalize
        e_m = F.normalize(e_m, p=2, dim=1)
        
        return e_m


class DatasetEncoder(nn.Module):
    """
    Embed a dataset into meta-space using a frozen backbone and MLP head.
    
    Inputs:
        - Batch of K images with per-image features from backbone
    
    Output:
        - Dataset embedding e_d ∈ R^D (L2-normalized)
    """
    
    def __init__(self, B: int, D: int = 256, hidden_dim: int = 512):
        """
        Args:
            B: Backbone feature dimension (e.g., 2048 for ResNet-50, 768 for ViT-B)
            D: Output embedding dimension (default 256)
            hidden_dim: Hidden layer dimension (default 1024)
        """
        super().__init__()
        
        self.B = B
        self.D = D
        
        # MLP head: [B → 1024 → 512 → D]
        self.mlp_head = nn.Sequential(
            nn.Linear(B, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(512, D)
        )
    
    def forward(self, dataset_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to encode a dataset into meta-space.
        
        Args:
            per_image_features: Per-image backbone features [batch_size (K), B]
        
        Returns:
            e_d: Dataset embedding [1, D] (L2-normalized)
        """
        # Average features across the batch (K images)
        # h_bar = (1/K) * sum(h_i)
        # h_bar = torch.mean(per_image_features, dim=0, keepdim=True)  # [1, B]
        #normalize dataset features
        # print stat about dataset features

        #dataset_features = F.normalize(dataset_features, p=2, dim=1)

        # Pass through MLP head
        e_d = self.mlp_head(dataset_features)  # [1, D]
        
        # L2 normalize
        e_d = F.normalize(e_d, p=2, dim=1)
        
        return e_d


class PerformancePredictor(nn.Module):
    def __init__(self, D: int = 256, constrain_output: bool = False):
        super().__init__()
        self.D = D
        self.constrain_output = constrain_output
        
        # Use full feature set as intended
        input_dim = 4 * D  # Changed back to 4D
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 2 * D),
            nn.LayerNorm(2 * D),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(2 * D, D),
            nn.LayerNorm(D),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(D, 1)
        )
        
        # Remove sigmoid if performances are already scaled
        self.output_activation = nn.Identity()
    
    def forward(self, e_m: torch.Tensor, e_d: torch.Tensor) -> torch.Tensor:
        # Use richer pairwise features
        elementwise_product = e_m * e_d
        absolute_difference = torch.abs(e_m - e_d)
        
        u = torch.cat([e_m, e_d, elementwise_product, absolute_difference], dim=1)
        
        y_hat = self.mlp(u)
        y_hat = self.output_activation(y_hat)
        
        return y_hat




class MetaSpaceSystem(nn.Module):
    """
    Complete meta-space system with corrected loss handling for both:
    1. Pairwise (model, dataset) samples for MSE loss
    2. Matrix form for ranking losses
    """
    
    def __init__(
        self,
        d_a: int,
        d_f: int,
        B: int,
        D: int = 256,
        constrain_perf_output: bool = True
    ):
        super().__init__()
        
        self.model_encoder = ModelEncoder(d_a, d_f, D)
        self.dataset_encoder = DatasetEncoder(B, D)
        self.perf_predictor = PerformancePredictor(D, constrain_perf_output)
        self.D = D
    
    def forward(
        self,
        arch_features: torch.Tensor,
        func_features: torch.Tensor,
        null_arch_features: torch.Tensor,
        dataset_encoding: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the entire system.
        
        Args:
            arch_features: Architecture features [batch_size, d_a]
            func_features: Functional features [batch_size, d_f]
            null_arch_features: Null architecture features [batch_size, d_f]
            dataset_encoding: Dataset encoding [batch_size, B]
        
        Returns:
            Dictionary with embeddings and predictions
        """
        # Encode model
        e_m = self.model_encoder(arch_features, func_features, null_arch_features)

        # Encode dataset
        e_d = self.dataset_encoder(dataset_encoding)

        # Predict performance
        y_hat = self.perf_predictor(e_m, e_d)
        
        return {
            'model_embedding': e_m,
            'dataset_embedding': e_d,
            'performance_prediction': y_hat
        }
    
    def construct_performance_matrix(
        self,
        batch_items: Dict,
        unique_models: Optional[List] = None,
        unique_datasets: Optional[List] = None
    ) -> Tuple[torch.Tensor, List, List]:
        """
        Construct performance matrix from batch items.
        
        This follows your compute_acc_matrix logic but generalized.
        
        Args:
            batch_items: Dictionary containing:
                - 'model_ids': List or tensor of model identifiers
                - 'dataset_ids': List or tensor of dataset identifiers  
                - 'performances': Tensor of performance values [batch_size]
            unique_models: Optional list of unique model IDs (will be computed if not provided)
            unique_datasets: Optional list of unique dataset IDs (will be computed if not provided)
        
        Returns:
            - performance_matrix: [num_unique_models, num_unique_datasets]
            - unique_models: List of unique model IDs
            - unique_datasets: List of unique dataset IDs
        """
        model_ids = batch_items['model_ids']
        dataset_ids = batch_items['dataset_ids']
        performances = batch_items['performances']
        
        # Get unique models and datasets if not provided
        if unique_models is None:
            unique_models = list(set(model_ids.tolist() if torch.is_tensor(model_ids) else model_ids))
        if unique_datasets is None:
            unique_datasets = list(set(dataset_ids.tolist() if torch.is_tensor(dataset_ids) else dataset_ids))
        
        num_models = len(unique_models)
        num_datasets = len(unique_datasets)
        
        # Create mapping from ID to index
        model_to_idx = {m: i for i, m in enumerate(unique_models)}
        dataset_to_idx = {d: i for i, d in enumerate(unique_datasets)}
        
        # Initialize performance matrix (using -1 for missing values, you can use 0 or NaN)
        perf_matrix = torch.full((num_models, num_datasets), -1.0, dtype=performances.dtype)
        
        # Fill in the matrix
        for i in range(len(model_ids)):
            model_idx = model_to_idx[model_ids[i].item() if torch.is_tensor(model_ids) else model_ids[i]]
            dataset_idx = dataset_to_idx[dataset_ids[i].item() if torch.is_tensor(dataset_ids) else dataset_ids[i]]
            perf_matrix[model_idx, dataset_idx] = performances[i]
        
        return perf_matrix, unique_models, unique_datasets
    
    def get_unique_embeddings(
        self,
        embeddings: torch.Tensor,
        ids: List,
        unique_ids: List
    ) -> torch.Tensor:
        """
        Extract unique embeddings based on IDs.
        
        Args:
            embeddings: All embeddings [batch_size, D]
            ids: List of IDs for each embedding
            unique_ids: List of unique IDs to extract
        
        Returns:
            Unique embeddings [num_unique, D]
        """
        id_to_idx = {}
        for i, id_val in enumerate(ids):
            id_val = id_val.item() if torch.is_tensor(id_val) else id_val
            if id_val not in id_to_idx:
                id_to_idx[id_val] = i
        
        unique_embeddings = []
        for uid in unique_ids:
            unique_embeddings.append(embeddings[id_to_idx[uid]])
        
        return torch.stack(unique_embeddings)
    
    # ============================================================================
    # LOSS FUNCTIONS
    # ============================================================================
    
    def compute_mse_loss(
        self,
        y_hat: torch.Tensor,
        y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        MSE loss for pairwise predictions.
        
        Args:
            y_hat: Predicted performance [batch_size, 1]
            y_true: Ground truth performance [batch_size, 1] or [batch_size]
        """
        return F.mse_loss(y_hat.squeeze(), y_true.squeeze())
    
    def compute_rank_loss_adjacent(
        self,
        e_m: torch.Tensor,
        e_d: torch.Tensor,
        perf_matrix: torch.Tensor,
        margin: float = 0.1,
    ) -> torch.Tensor:
        """
        Rank loss comparing adjacent pairs in sorted order.
        
        Args:
            e_m: Model embeddings [num_unique_models, D]
            e_d: Dataset embeddings [num_unique_datasets, D]
            perf_matrix: Performance matrix [num_unique_models, num_unique_datasets]
            margin: Margin for ranking violations
        """
        # Compute similarity matrix
        sims = torch.matmul(e_m, e_d.t())
        
        total = torch.tensor(0.0, dtype=e_m.dtype, device=e_m.device)
        comparisons = 0
        
        # For each dataset
        for dataset_col in range(perf_matrix.size(1)):
            # Get performances for this dataset
            perfs = perf_matrix[:, dataset_col]
            
            # Filter out missing values (-1 or whatever marker you use)
            valid_mask = perfs >= 0
            if valid_mask.sum() <= 1:
                continue
            
            valid_perfs = perfs[valid_mask]
            valid_sims = sims[valid_mask, dataset_col]
            
            # Sort by performance
            order = torch.argsort(valid_perfs, descending=True)
            ranked_sims = valid_sims[order]
            
            # Adjacent pairs comparison
            diffs = ranked_sims[:-1] - ranked_sims[1:]
            total = total + torch.relu(margin - diffs).sum()
            comparisons += diffs.numel()
        
        if comparisons == 0:
            return total
        return total / comparisons
    
    def compute_rank_loss_contrastive(
        self,
        e_m: torch.Tensor,
        e_d: torch.Tensor,
        perf_matrix: torch.Tensor,
        temperature: float = 0.1
    ) -> torch.Tensor:
        """
        Contrastive ranking loss considering all pairs.
        
        Args:
            e_m: Model embeddings [num_unique_models, D]
            e_d: Dataset embeddings [num_unique_datasets, D]
            perf_matrix: Performance matrix [num_unique_models, num_unique_datasets]
            temperature: Temperature scaling
        """
        sims = torch.matmul(e_m, e_d.t()) / temperature
        
        total_loss = torch.tensor(0.0, dtype=e_m.dtype, device=e_m.device)
        num_comparisons = 0
        
        for dataset_col in range(perf_matrix.size(1)):
            perfs = perf_matrix[:, dataset_col]
            
            # Filter valid performances
            valid_mask = perfs >= 0
            if valid_mask.sum() <= 1:
                continue
                
            valid_perfs = perfs[valid_mask]
            valid_sims = sims[valid_mask, dataset_col]
            
            # Compare all pairs
            for i in range(len(valid_perfs)):
                for j in range(i+1, len(valid_perfs)):
                    if torch.abs(valid_perfs[i] - valid_perfs[j]) < 1e-6:
                        continue
                    
                    if valid_perfs[i] > valid_perfs[j]:
                        # Want sim[i] > sim[j]
                        loss = torch.log(1 + torch.exp(valid_sims[j] - valid_sims[i]))
                    else:
                        loss = torch.log(1 + torch.exp(valid_sims[i] - valid_sims[j]))
                    
                    total_loss += loss
                    num_comparisons += 1
        
        if num_comparisons == 0:
            return total_loss
        return total_loss / num_comparisons
    
    def compute_alignment_loss(
        self,
        e_m: torch.Tensor,
        e_d: torch.Tensor,
        perf_matrix: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Direct alignment loss - similarity should match performance.
        
        Args:
            e_m: Model embeddings [num_unique_models, D]
            e_d: Dataset embeddings [num_unique_datasets, D]
            perf_matrix: Performance matrix [num_unique_models, num_unique_datasets]
        """
        # Compute similarities
        sims = torch.matmul(e_m, e_d.t()) / temperature
        
        # Apply sigmoid if performances are in [0,1]
        if self.perf_predictor.constrain_output:
            sims = torch.sigmoid(sims)
        
        # Create mask for valid entries
        valid_mask = perf_matrix >= 0
        
        # Compute loss only on valid entries
        if valid_mask.sum() > 0:
            return F.mse_loss(sims[valid_mask], perf_matrix[valid_mask])
        else:
            return torch.tensor(0.0, device=e_m.device)
    
    def compute_diversity_loss(
        self,
        embeddings: torch.Tensor,
        target_diversity: float = 0.1
    ) -> torch.Tensor:
        """Encourage diverse embeddings to prevent collapse."""
        gram = torch.matmul(embeddings, embeddings.t())
        batch_size = gram.size(0)
        target = torch.eye(batch_size, device=gram.device) * (1 - target_diversity) + target_diversity
        return F.mse_loss(gram, target)
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch_items: Dict,
        loss_type: LossType = LossType.MSE_RANK_CONTRASTIVE,
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss with proper handling of pairwise and matrix forms.
        
        Args:
            outputs: Dictionary from forward() containing embeddings and predictions
            batch_items: Dictionary containing:
                - For MSE: 'performances' tensor [batch_size]
                - For ranking: 'model_ids', 'dataset_ids', 'performances' to construct matrix
            loss_type: Which loss combination to use
            weights: Optional custom weights for loss components
        
        Returns:
            Tuple of (total_loss, dict of individual loss values)
        """
        # Default weights
        default_weights = {
            'mse': 1.0,
            'rank': 0.5,
            'alignment': 0.5,
            'diversity': 0.1,
        }
        
        if weights is not None:
            default_weights.update(weights)
        
        # Extract from outputs
        e_m = outputs['model_embedding']
        e_d = outputs['dataset_embedding']
        y_hat = outputs['performance_prediction']
        
        # Initialize losses
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=e_m.device,requires_grad=True)
        
        # Always compute MSE loss (pairwise)
        if 'performances' in batch_items:
            y_true_pairwise = batch_items['performances']
            if y_true_pairwise.dim() == 1:
                y_true_pairwise = y_true_pairwise.unsqueeze(1)
            
            mse_loss = self.compute_mse_loss(y_hat, y_true_pairwise)
            loss_dict['mse'] = mse_loss.item()
        else:
            mse_loss = torch.tensor(0.0, device=e_m.device)
            loss_dict['mse'] = 0.0
        
        # Compute ranking losses if needed (require matrix form)
        needs_matrix = loss_type in [
            LossType.MSE_RANK_ADJACENT,
            LossType.MSE_RANK_CONTRASTIVE,
            LossType.MSE_ALIGNMENT,
            LossType.MSE_ALIGNMENT_DIVERSITY,
            LossType.MSE_RANK_WEIGHTED,
            LossType.COMBINED_ALL
        ]
        
        if needs_matrix and 'model_ids' in batch_items and 'dataset_ids' in batch_items:
            # Construct performance matrix
            perf_matrix, unique_models, unique_datasets = self.construct_performance_matrix(batch_items)
            
            # Get unique embeddings
            e_m_unique = self.get_unique_embeddings(e_m, batch_items['model_ids'], unique_models)
            e_d_unique = self.get_unique_embeddings(e_d, batch_items['dataset_ids'], unique_datasets)
            
            # Now compute ranking losses with unique embeddings and matrix
            if loss_type == LossType.MSE_RANK_ADJACENT:
                rank_loss = self.compute_rank_loss_adjacent(e_m_unique, e_d_unique, perf_matrix)
                total_loss = default_weights['mse'] * mse_loss + default_weights['rank'] * rank_loss
                loss_dict['rank_adjacent'] = rank_loss.item()
                
            elif loss_type == LossType.MSE_RANK_CONTRASTIVE:
                rank_loss = self.compute_rank_loss_contrastive(e_m_unique, e_d_unique, perf_matrix)
                total_loss = default_weights['mse'] * mse_loss + default_weights['rank'] * rank_loss
                loss_dict['rank_contrastive'] = rank_loss.item()
                
            elif loss_type == LossType.MSE_ALIGNMENT:
                align_loss = self.compute_alignment_loss(e_m_unique, e_d_unique, perf_matrix)
                total_loss = default_weights['mse'] * mse_loss + default_weights['alignment'] * align_loss
                loss_dict['alignment'] = align_loss.item()
                
            elif loss_type == LossType.MSE_ALIGNMENT_DIVERSITY:
                align_loss = self.compute_alignment_loss(e_m_unique, e_d_unique, perf_matrix)
                div_loss = self.compute_diversity_loss(e_m_unique) + self.compute_diversity_loss(e_d_unique)
                total_loss = (default_weights['mse'] * mse_loss + 
                            default_weights['alignment'] * align_loss +
                            default_weights['diversity'] * div_loss)
                loss_dict['alignment'] = align_loss.item()
                loss_dict['diversity'] = div_loss.item() / 2
                
            elif loss_type == LossType.COMBINED_ALL:
                rank_loss = self.compute_rank_loss_contrastive(e_m_unique, e_d_unique, perf_matrix)
                align_loss = self.compute_alignment_loss(e_m_unique, e_d_unique, perf_matrix)
                div_loss = self.compute_diversity_loss(e_m_unique) + self.compute_diversity_loss(e_d_unique)
                
                total_loss = (default_weights['mse'] * mse_loss + 
                            default_weights['rank'] * rank_loss +
                            default_weights['alignment'] * align_loss +
                            default_weights['diversity'] * div_loss)
                
                loss_dict['rank_contrastive'] = rank_loss.item()
                loss_dict['alignment'] = align_loss.item()
                loss_dict['diversity'] = div_loss.item() / 2
        
        elif loss_type == LossType.MSE_ONLY:
            total_loss = mse_loss
        else:
            # If we can't compute ranking losses, fall back to MSE only
            print(f"Warning: Cannot compute {loss_type.value}, falling back to MSE only")
            total_loss = mse_loss
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict





