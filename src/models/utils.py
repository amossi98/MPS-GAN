import torch
import numpy as np

def compute_MPS_MPS_product(MPS0, MPS1):
    """
    Computes the inner product <MPS0 | MPS1> using a density matrix approach.

    Args:
        MPS0 (torch.Tensor): Shape (n, D, D, d), representing the first MPS.
        MPS1 (torch.Tensor): Shape (n, D, D, d), representing the second MPS.

    Returns:
        torch.Tensor: A scalar tensor representing the overlap between the two MPS.
    """
    n = MPS0.shape[0]
    contr = torch.einsum('...rd, ...Rd->...rR', MPS0[0, 0], MPS1[0, 0])

    for i in range(1, n-1):
        contr = torch.einsum('...rR, ...rld->...lRd', contr, MPS0[i])
        contr = torch.einsum('...rRd, ...RLd->...rL', contr, MPS1[i])

    contr = torch.einsum('...rR, ...rd->...Rd', contr, MPS0[n-1, :, 0])
    contr = torch.einsum('...Rd, ...Rd->...', contr, MPS1[n-1, :, 0])

    return contr


def sample_from_v(v, percentile, emb, nbins=1000):
    """
    Samples a value based on the conditional probability distribution p(x_i | x_{i+1}, ..., x_{n-1}).

    Args:
        v (torch.Tensor): Matrix defining the quadratic form used in sampling.
        percentile (float): Target percentile for sampling.
        emb: Embedding function instance.
        nbins (int, optional): Number of bins for discretization. Defaults to 1000.

    Returns:
        torch.Tensor: Sampled value.
    """
    if percentile < 1e-3:
        percentile = 1e-3  # Avoid percentile too close to zero

    lin = np.linspace(0, 1, nbins)
    x = torch.tensor(lin, dtype=torch.float32).squeeze(0).unsqueeze(-1)
    y = emb(x).squeeze(1)

    y = torch.einsum("bi,ij,bj->b", y, v, y)
    cpd = y.cumsum(0)
    cpd = cpd / cpd[-1]  # Normalize to [0,1]

    vec = cpd < percentile

    if not vec.any():  # If no values meet the condition, return the smallest x
        return x[0]

    idx = vec.nonzero().max()

    # Linear interpolation
    idx_float = idx.float()
    lower_val = cpd[idx]
    upper_val = cpd[idx + 1]
    interpolated_idx = idx_float + (percentile - lower_val) / (upper_val - lower_val)

    # Interpolate x values
    return torch.lerp(x[idx], x[idx + 1], interpolated_idx - idx_float)


def weight_reg(mps, l):
    """
    Computes a weight regularization term for an MPS.

    Args:
        mps: The MPS instance containing `tensor`.
        l (float): Regularization coefficient.

    Returns:
        torch.Tensor: The computed regularization term.
    """
    n, D, d = mps.tensor.shape[0], mps.tensor.shape[1], mps.tensor.shape[-1]
    return l * (mps.tensor - initialize_MPS_values(n, D, d)).pow(2).sum()


def initialize_MPS_values(n, D, d):
    """
    Initializes an MPS to the identity tensor with noise.

    Args:
        n (int): Number of sites.
        D (int): Bond dimension.
        d (int): Embedding dimension.

    Returns:
        torch.Tensor: Initialized MPS tensor of shape (n, D, D, d).
    """
    id_matrix = torch.eye(D).unsqueeze(2).repeat(1, 1, d) / np.sqrt(d)
    return id_matrix.expand(n, D, D, d)



class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False