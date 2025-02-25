"""
test_sampling_bins.py

This script analyzes how the number of bins used in the sampling routine affects both the 
sampling error (squared error relative to a target of 0.5) and the runtime. 
It uses an embedding (here, Fourier) to compute a conditional probability distribution and 
samples a value based on a given percentile.
"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.models.emb import Emb  # Assumes emb.py provides an Emb class

def sample_from_v(v: torch.Tensor, percentile: float, emb, nbins: int = 1000) -> torch.Tensor:
    """
    Samples a value from a conditional probability distribution defined by matrix v and embedding emb.
    
    Args:
        v (torch.Tensor): A square matrix (e.g., identity of size d x d).
        percentile (float): The target percentile (between 0 and 1) at which to sample.
        emb: An embedding object (callable) that maps inputs.
        nbins (int): Number of bins to discretize the [0, 1] interval.
    
    Returns:
        torch.Tensor: A scalar tensor representing the sampled value.
    """
    if percentile < 1e-3:
        percentile = 1e-3

    # Create a linearly spaced tensor in [0,1]
    lin = np.linspace(0, 1, nbins)
    x = torch.tensor(lin, dtype=torch.float32).unsqueeze(-1)  # shape (nbins, 1)

    # Obtain the embedding and compute a quadratic form: y^T v y
    y = emb(x).squeeze(1)
    y = torch.einsum("bi,ij,bj->b", y, v, y)

    # Compute the cumulative probability distribution (CPD)
    cpd = y.cumsum(0)
    cpd = cpd / cpd[-1]

    # Find the largest index where cpd is below the target percentile
    vec = cpd < percentile
    if (~vec).all():
        return x[0]
    idx = (vec).nonzero().max()

    # Perform linear interpolation between x[idx] and x[idx+1]
    idx_float = idx.float()
    lower_val = cpd[idx]
    upper_val = cpd[idx + 1]
    interpolated_idx = idx_float + (percentile - lower_val) / (upper_val - lower_val)
    sample = torch.lerp(x[idx], x[idx + 1], interpolated_idx - idx_float)
    return sample

def main():
    emb_obj = Emb(10, 'fourier')
    nbins_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
    samples = []
    times = []

    for nbins in nbins_list:
        start = time.time()
        sample_val = sample_from_v(torch.eye(10), 0.5, emb_obj, nbins)
        end = time.time()
        times.append(end - start)
        samples.append(sample_val.item())
        print(f"Time for nbins={nbins}: {end - start:.6f} seconds")

    errors = [(s - 0.5) ** 2 for s in samples]
    print("Times:", times)
    print("Errors:", errors)

    # Plot error vs. number of bins (log-log scale)
    plt.figure()
    plt.plot(nbins_list, errors, marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of bins', fontsize=14)
    plt.ylabel('Error squared', fontsize=14)
    plt.title("Sampling Error vs. Number of Bins")
    plt.grid(True)
    plt.show()

    # Plot runtime vs. number of bins (log-log scale)
    plt.figure()
    plt.plot(nbins_list, times, marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of bins', fontsize=14)
    plt.ylabel('Time to sample (s)', fontsize=14)
    plt.title("Sampling Time vs. Number of Bins")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
