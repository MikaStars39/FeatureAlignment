import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F

@torch.no_grad()
def sae_match(
    W_enc_i: torch.Tensor,
    W_enc_j: torch.Tensor,
    W_dec_i: torch.Tensor,
    W_dec_j: torch.Tensor,
    b_enc_i: torch.Tensor,
    b_enc_j: torch.Tensor,
    theta_i: torch.Tensor,
    theta_j: torch.Tensor
) -> torch.Tensor:
    """
    SAE Match function to align features across layers i and j using GPU acceleration.
    Uses Frobenius inner product maximization to find the optimal permutation matrix.

    Args:
        W_enc_i (torch.Tensor): Encoder weights of layer i, shape (hidden_dim, feature_dim).
        W_enc_j (torch.Tensor): Encoder weights of layer j, shape (hidden_dim, feature_dim).
        W_dec_i (torch.Tensor): Decoder weights of layer i, shape (feature_dim, hidden_dim).
        W_dec_j (torch.Tensor): Decoder weights of layer j, shape (feature_dim, hidden_dim).
        b_enc_i (torch.Tensor): Encoder bias of layer i, shape (feature_dim,).
        b_enc_j (torch.Tensor): Encoder bias of layer j, shape (feature_dim,).
        theta_i (torch.Tensor): Activation thresholds of layer i, shape (feature_dim,).
        theta_j (torch.Tensor): Activation thresholds of layer j, shape (feature_dim,).

    Returns:
        permutation_matrix (torch.Tensor): Permutation matrix that aligns features from layer i to layer j.
    """
    device = W_enc_i.device

    # Step 1: Parameter Folding (keeping tensors on GPU)
    W_enc_i_folded = W_enc_i * (1 / theta_i).unsqueeze(0)
    b_enc_i_folded = b_enc_i * (1 / theta_i)
    W_dec_i_folded = W_dec_i * theta_i.unsqueeze(1)

    W_enc_j_folded = W_enc_j * (1 / theta_j).unsqueeze(0)
    b_enc_j_folded = b_enc_j * (1 / theta_j)
    W_dec_j_folded = W_dec_j * theta_j.unsqueeze(1)

    # Step 2: Compute Frobenius inner product between decoder weights
    # <W_dec_i, W_dec_j> = Tr(W_dec_i^T @ W_dec_j)
    cost_matrix = torch.mm(W_dec_i_folded, W_dec_j_folded.t())

    # Step 3: Solve the Linear Assignment Problem (LAP)
    # Note: We negate the cost matrix since linear_sum_assignment minimizes
    # while we want to maximize the Frobenius inner product
    cost_matrix_np = -cost_matrix.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix_np)

    # Step 4: Create the permutation matrix on GPU
    permutation_matrix = torch.zeros_like(cost_matrix, device=device)
    permutation_matrix[row_ind, col_ind] = 1

    return permutation_matrix

# Example usage:

# permutation_matrix = sae_match(hidden_i, hidden_j, W_enc_i, W_enc_j, W_dec_i, W_dec_j, b_enc_i, b_enc_j, theta_i, theta_j)
# print(permutation_matrix)