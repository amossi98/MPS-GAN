"""
This script loads generated samples saved as .npy files from various experiments,
computes an FID-like score comparing the generated samples to their corresponding 
real datasets (iris, 2moon, and spiral), and visualizes scatter plots and PCA projections.
If the generated sample array has more columns than the real data features, the remaining columns are assumed to be
the class labels and are used for coloring the plots.

Expected files (embedding can be legendre or fourier):
  - tgan_iris_legendre_pretrain.npy
  - tgan_iris_legendre_postgan.npy
  - tgan_iris_fourier_pretrain.npy
  - tgan_iris_fourier_postgan.npy
  - tgan_2moon_legendre_pretrain.npy
  - tgan_2moon_legendre_postgan.npy
  - tgan_2moon_fourier_pretrain.npy
  - tgan_2moon_fourier_postgan.npy
  - tgan_spiral_legendre_pretrain.npy
  - tgan_spiral_legendre_postgan.npy
  - tgan_spiral_fourier_pretrain.npy
  - tgan_spiral_fourier_postgan.npy

The FID-like score is computed as:
    FID = ||mu_real - mu_gen||^2 + Tr(sigma_real + sigma_gen - 2*(sigma_real*sigma_gen)^(1/2))
where mu and sigma are the mean and covariance of the real and generated samples.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris, make_moons
from src.datasets.synthetic import generate_spiral_dataset

def calculate_fid(real_samples: np.ndarray, gen_samples: np.ndarray) -> float:
    """
    Compute an FID-like score between real and generated samples.
    
    Args:
        real_samples (np.ndarray): Real data samples, shape (N, d).
        gen_samples (np.ndarray): Generated samples, shape (M, d).
    
    Returns:
        float: The FID-like score.
    """
    mu_real = np.mean(real_samples, axis=0)
    sigma_real = np.cov(real_samples, rowvar=False)
    mu_gen = np.mean(gen_samples, axis=0)
    sigma_gen = np.cov(gen_samples, rowvar=False)
    
    diff = mu_real - mu_gen
    diff_sq = np.dot(diff, diff)
    
    covmean = sqrtm(sigma_real.dot(sigma_gen))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff_sq + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid

def load_real_data(dataset_type: str) -> np.ndarray:
    """
    Load the real dataset corresponding to the provided dataset type.
    
    Args:
        dataset_type (str): One of 'iris', '2moon', or 'spiral'.
    
    Returns:
        np.ndarray: Normalized real data samples.
    """
    if dataset_type.lower() == 'iris':
        iris = load_iris()
        data = iris.data  # Use all four features
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        data_norm = (data - data_min) / (data_max - data_min)
        return data_norm
    elif dataset_type.lower() == '2moon':
        X, _ = make_moons(n_samples=10000, noise=0.1, random_state=42)
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        return (X - X_min) / (X_max - X_min)
    elif dataset_type.lower() == 'spiral':
        X_train, _, X_test, _ = generate_spiral_dataset(N=8000)
        return np.vstack((X_train, X_test))
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

def plot_scatter(gen_samples: np.ndarray, title: str):
    """
    Plot a scatter plot of generated samples.
    If gen_samples has more columns than features, the last column is assumed to be class labels.
    
    Args:
        gen_samples (np.ndarray): Generated samples.
        title (str): Plot title.
    """
    plt.figure(figsize=(5, 5))
    if gen_samples.shape[1] > 2:  # Assuming features + label
        labels = gen_samples[:, -1]
        sc = plt.scatter(gen_samples[:, 0], gen_samples[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(sc, label='Class Label')
    else:
        plt.scatter(gen_samples[:, 0], gen_samples[:, 1], c='blue', alpha=0.5, label='Generated Samples')
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_pca(real_samples: np.ndarray, gen_samples: np.ndarray, title: str, dataset_type: str):
    """
    Project real and generated samples to 2D using PCA and plot.
    
    Args:
        real_samples (np.ndarray): Real samples, shape (N, d).
        gen_samples (np.ndarray): Generated samples, shape (M, d).
        title (str): Plot title.
        dataset_type (str): Dataset type for formatting.
    """
    pca = PCA(n_components=2)
    pca.fit(real_samples)
    real_2d = pca.transform(real_samples)
    gen_2d = pca.transform(gen_samples)
    
    plt.figure(figsize=(5, 5))
    plt.scatter(real_2d[:, 0], real_2d[:, 1], c='gray', alpha=0.5, label='Real Data')
    if gen_samples.shape[1] < real_samples.shape[1] + 1:  # Check if labels are present
        plt.scatter(gen_2d[:, 0], gen_2d[:, 1], c='blue', alpha=0.7, label='Generated Data')
    else:
        labels = gen_samples[:, -1]
        sc = plt.scatter(gen_2d[:, 0], gen_2d[:, 1], c=labels, cmap='viridis', alpha=0.7, label='Generated Data')
        plt.colorbar(sc, label='Class Label')
    plt.title(title)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend()
    plt.grid(True)
    
    # Apply Iris-specific formatting
    if dataset_type.lower() == 'iris':
        plt.xticks([-0.5, 0, 0.5])
        plt.yticks([-0.5, 0, 0.5])
        plt.xlim(-0.8, 0.8)
        plt.ylim(-0.6, 0.6)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.tight_layout()
    plt.show()

def evaluate_file(file_path: str, dataset_type: str) -> float:
    """
    Load generated samples from a file, compute the FID-like score against real data,
    and display scatter and PCA plots.
    
    Args:
        file_path (str): Path to the .npy file.
        dataset_type (str): 'iris', '2moon', or 'spiral'.
    
    Returns:
        float: The computed FID-like score.
    """
    gen_samples = np.load(file_path)
    real_samples = load_real_data(dataset_type)
    # Extract features matching real data dimensions
    real_features = real_samples.shape[1]
    gen_samples_eval = gen_samples[:, :real_features]
    
    fid = calculate_fid(real_samples, gen_samples_eval)
    base_name = os.path.basename(file_path)
    
    # Plot scatter (2D datasets) or PCA (iris)
    if dataset_type != 'iris':
        plot_scatter(gen_samples, f"{base_name}\nFID: {fid:.4f}")
    else:
        plot_pca(real_samples, gen_samples_eval, f"PCA Projection: {base_name}", dataset_type)
    
    return fid

def main():
    # List of generated sample files
    files = [
        # Iris dataset
        # "tgan_iris_legendre_pretrain.npy",
        # "tgan_iris_legendre_postgan.npy",
        "tgan_iris_fourier_pretrain.npy",
        "tgan_iris_fourier_postgan.npy",
        # 2 Moon dataset
        # "tgan_2moon_legendre_pretrain.npy",
        # "tgan_2moon_legendre_postgan.npy",
        "tgan_2moon_fourier_pretrain.npy",
        "tgan_2moon_fourier_postgan.npy",
        # Spiral dataset
        # "tgan_spiral_legendre_pretrain.npy",
        # "tgan_spiral_legendre_postgan.npy",
        "tgan_spiral_fourier_pretrain.npy",
        "tgan_spiral_fourier_postgan.npy",
    ]
    
    data_folder = "./data"
    
    results = {}
    for file_name in files:
        file_path = os.path.join(data_folder, file_name)
        if "iris" in file_name.lower():
            dataset_type = "iris"
        elif "2moon" in file_name.lower():
            dataset_type = "2moon"
        elif "spiral" in file_name.lower():
            dataset_type = "spiral"
        else:
            print(f"Skipping {file_name} (unknown dataset type)")
            continue
        
        print(f"Evaluating {file_name} on dataset '{dataset_type}'...")
        fid = evaluate_file(file_path, dataset_type)
        results[file_name] = fid
    
    print("\nFID Scores:")
    for fname, fid in results.items():
        print(f"{fname}: {fid:.4f}")

if __name__ == '__main__':
    main()