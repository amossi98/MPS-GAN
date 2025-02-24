import torch
import torch.nn as nn
import numpy as np
from scipy.special import jn, chebyc
from scipy.linalg import hadamard
import math
import functools

class Emb(nn.Module):
    def __init__(self, d, family, sigma=0):
        """
        Initializes the embedding module with different basis function families.

        Args:
            d (int): Dimension of the embedding.
            family (str): Type of basis function ('fourier', 'legendre', 'bessel', 'haar', 'cheby', 'walsh').
            sigma (float, optional): Noise level for adding perturbations. Default is 0.
        """
        super(Emb, self).__init__()
        self.d = d
        self.sigma = sigma
        self.family = family

        # Precompute basis functions
        if self.family == 'fourier':
            self.basis_functions = self.generate_fourier_basis(d)
        elif self.family == 'legendre':
            self.basis_functions = self.generate_legendre_basis(d)
        elif self.family == 'bessel':
            self.basis_functions = self.generate_bessel_basis(d)
        elif self.family == 'haar':
            self.basis_functions = self.generate_haar_basis(d)
        elif self.family == 'cheby':
            self.basis_functions = self.generate_cheby_basis(d)
        elif self.family == 'walsh':
            assert (d != 0) and (d & (d - 1) == 0), 'd should be a power of 2'
            self.basis_functions = self.generate_walsh_basis(d)

    def generate_fourier_basis(self, d):
        """Precompute Fourier basis functions."""
        return [functools.partial(self.fourier_function, i=i) for i in range(d)]
    
    def fourier_function(self, x, i):
        return torch.cos(torch.pi * i * x)

    def generate_legendre_basis(self, d):
        """Precompute Legendre polynomials."""
        return [functools.partial(self.legendre, n=i) for i in range(d)]
    
    def legendre(self, x, n):
        """Compute the Legendre polynomial of degree n at x using recurrence relation."""
        if n == 0:
            return torch.ones_like(x)
        elif n == 1:
            return x
        else:
            return ((2 * n - 1) * x * self.legendre(x, n - 1) - (n - 1) * self.legendre(x, n - 2)) / n

    def generate_bessel_basis(self, d):
        """Precompute Bessel functions."""
        return [functools.partial(self.bessel_function, i=i) for i in range(d)]
    
    def bessel_function(self, x, i):
        return torch.tensor(jn(i, x), dtype=torch.float32)

    def generate_cheby_basis(self, d):
        """Precompute Chebyshev polynomials."""
        return [functools.partial(self.cheby_function, i=i) for i in range(d)]
    
    def cheby_function(self, x, i):
        return torch.tensor(chebyc(i, 2*x-1), dtype=torch.float32)

    def generate_walsh_basis(self, d):
        """Precompute Walsh basis functions using Hadamard matrices."""
        H = hadamard(d)
        return [functools.partial(self.walsh_function, vec=H[i]) for i in range(d)]
    
    def walsh_function(self, x, vec):
        """Compute Walsh function at x using Hadamard matrix row."""
        indices = (x * (len(vec) - 1)).long()
        return torch.tensor([vec[i] for i in indices], dtype=torch.float32)

    def generate_haar_basis(self, d):
        """Precompute Haar wavelets."""
        return [functools.partial(self.haar_function, d=d, k=i) for i in range(d)]
    
    def haar_function(self, x, d, k):
        return self.psi_n_k(d, k, x)
    
    def psi_0_0(self, t):
        """Base Haar wavelet function."""
        return torch.where(t < 0.5, torch.tensor(1.0), torch.tensor(-1.0))

    def psi_n_k(self, n, k, t):
        """Haar wavelet function for (n, k)."""
        return self.psi_0_0(2 ** n * t - k)

    def forward(self, x):
        """
        Compute embeddings for input x using precomputed basis functions.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n).

        Returns:
            torch.Tensor: Embedded tensor of shape (batch_size, n, d).
        """
        if x.dim() == 0:
            x = x.unsqueeze(0)

        b, n = x.shape
        emb = torch.zeros((b, n, self.d), dtype=torch.float32)

        for i in range(self.d):
            emb[:, :, i] = self.basis_functions[i](x)

        if self.sigma > 0:
            emb += torch.randn_like(emb) * self.sigma

        return emb
