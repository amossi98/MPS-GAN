import torch
import torch.nn as nn
from src.models.utils import initialize_MPS_values, sample_from_v
from src.models.emb import Emb  # Ensure correct import

class MPS(nn.Module):
    """
    Matrix Product State (MPS) module.
    Implements core operations including contraction, normalization, and sampling.
    """
    def __init__(self, n, D, d, stddev=0.5, family='fourier'):
        """
        Args:
            n (int): Input size (number of sites).
            D (int): Bond dimension.
            d (int): Physical (embedding) dimension.
            stddev (float): Standard deviation for initializing tensor weights.
            family (str): Embedding family ('fourier', 'legendre', etc.).
        """
        super().__init__()
        self.n = n
        self.D = D
        self.d = d
        self.stddev = stddev
        self.embedding = Emb(d, family=family)

        # Initialize the MPS tensor (identity + noise)
        self.tensor_initialized = initialize_MPS_values(n, D, d)
        self.tensor = nn.Parameter(torch.randn(n, D, D, d) * stddev)

    def forward(self, input):
        """
        Forward pass: Contract the input with the MPS to compute the log-overlap.
        
        Args:
            input (Tensor): Shape (batch_size, n, d)
        
        Returns:
            Tensor: Output of shape (batch_size,)
        """
        b, n, d = input.shape
        assert d == self.d, f"Expected feature size d={self.d}, got {d}."
        assert n == self.n, f"Expected input size n={self.n}, got {n}."

        tensor = self.tensor  # Optionally add self.tensor_initialized if needed
        mps = torch.einsum('bne,nlre->bnlr', input, tensor)

        # Initialize contractions
        left_contr = mps[:, 0, 0]     # Shape: (batch, D)
        right_contr = mps[:, -1, :, 0]  # Shape: (batch, D)

        # Contract sequentially
        for i in range(1, n // 2):
            left_contr = torch.einsum('br,brl->bl', left_contr, mps[:, i])
            right_contr = torch.einsum('bl,brl->br', right_contr, mps[:, n - i - 1])

        if n % 2 == 1:  # If odd number of sites, contract the middle site
            left_contr = torch.einsum('br,brl->bl', left_contr, mps[:, n // 2])

        return torch.einsum('bl,bl->b', left_contr, right_contr)

    def compute_left_sampling_contractions(self):
        """
        Compute left contractions used in the sampling step.
        
        Returns:
            Tensor: Contractions of shape (n-1, D, D)
        """
        sampling_contractions = torch.empty(self.n - 1, self.D, self.D, device=self.tensor.device)
        mps = self.tensor  # Optionally add self.tensor_initialized if needed

        left_contr = mps[0, 0]
        sampling_contractions[0] = torch.einsum("re,de->rd", left_contr, left_contr)

        for i in range(1, self.n - 1):
            left_contr = torch.einsum("le,lre->re", left_contr, mps[i])
            sampling_contractions[i] = torch.einsum("re,de->rd", left_contr, left_contr)

        return sampling_contractions


    def norm(self):
        """
        Compute the norm^2 of the MPS.
        
        Returns:
            Tensor: Scalar norm.
        """
        n = self.n
        mps = self.tensor + self.tensor_initialised
        left_contr = torch.einsum('rd,Rd->rR', mps[0, 0], mps[0, 0])
        right_contr = torch.einsum('ld,Ld->lL', mps[-1, :, 0], mps[-1, :, 0])
        
        for i in range(1, n // 2):
            left_contr = torch.einsum('lL,lrd->rLd', left_contr, mps[i])
            left_contr = torch.einsum('rLd,LRd->rR', left_contr, mps[i])
            right_contr = torch.einsum('rR,lrd->lRd', right_contr, mps[-i])
            right_contr = torch.einsum('lRd,LRd->lL', right_contr, mps[-i])
        
        contr = torch.einsum('td,td->', left_contr, right_contr)
        return torch.sqrt(contr)

    def normalize(self):
        """
        Normalize the MPS tensor.
        """
        norm_val = self.norm()
        normalized_tensor = self.tensor / torch.pow(norm_val, 1 / self.n)
        self.tensor = nn.Parameter(normalized_tensor)

    def sample(self, noise_vector):
        """
        Generate a sample from the MPS given a noise vector.
        
        Args:
            noise_vector (Tensor): Noise vector of shape (n,)
        Returns:
            Tensor: Sampled data of shape (n,)
        """
        n = self.n
        samples = torch.zeros(n, device=self.tensor.device)
        emb = self.embedding
        mps = self.tensor  # Optionally add self.tensor_initialised

        # Compute left contractions used for sampling
        left_sampling_contractions = self.compute_left_sampling_contractions()

        # Sample the last site (starting from the right)
        left_contr = left_sampling_contractions[-1]
        current_tensor = mps[-1, :, 0]
        v = torch.einsum("ud,ui,dj->ij", left_contr, current_tensor, current_tensor)
        sample = sample_from_v(v, noise_vector[-1], self.embedding)
        samples[-1] = sample
        y = emb(sample).squeeze()
        right_contr = torch.einsum("le,e->l", current_tensor, y)

        # Backward iteration for sampling intermediate sites
        for i in range(n - 2, 0, -1):
            left_contr = left_sampling_contractions[i]
            current_tensor = mps[i]
            current_tensor = torch.einsum("lri,r->li", current_tensor, right_contr)
            v = torch.einsum("ud,ui,dj->ij", left_contr, current_tensor, current_tensor)
            sample = sample_from_v(v, noise_vector[i], self.embedding)
            samples[i] = sample
            right_contr = torch.einsum("le,e->l", current_tensor, emb(sample).squeeze())
        
        # Sample the first site
        current_tensor = mps[0, 0]
        current_tensor = torch.einsum("ri,r->i", current_tensor, right_contr)
        v = torch.einsum("i,j->ij", current_tensor, current_tensor)
        samples[0] = sample_from_v(v, noise_vector[0], self.embedding)
        return samples
