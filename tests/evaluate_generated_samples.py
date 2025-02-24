"""
evaluate_generated_samples.py

This script loads generated samples saved as .npy files for various experiments,
computes an FID-like score comparing the generated samples to their corresponding 
real datasets, and visualizes scatter plots of the generated samples.

Expected files (if not otherwise specified, assume Fourier embedding):
  - mps_2moon_legendre.npy
  - mps_2moon.npy
  - mps_iris_legendre.npy
  - mps_iris.npy
  - mps_spiral_fourier_new.npy
  - mps_spiral_legendre.npy
  - tgan_2moon_legendre.npy
  - tgan_2moon.npy
  - tgan_iris_legendre.npy
  - tgan_iris.npy
  - tgan_spiral_fourier_new.npy
  - tgan_spiral_legendre.npy

The FID-like score is computed as:
    FID = ||mu_real - mu_gen||^2 + Tr(sigma_real + sigma_gen - 2*(sigma_real*sigma_gen)^(1/2))
where mu and sigma are the mean and covariance of the real and generated samples.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from sklearn.datasets import load_iris, make_moons

# For the spiral dataset, we assume you have a dataset generator already defined.
# Here we import the function from your datasets module.
from src.datasets.synthetic import generate_spiral_dataset  # Adjust the path if needed

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
    diff_sq = np.sum(diff**2)
    
    # Compute the matrix square root of the product of covariances.
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
        np.ndarray: Real data samples (using the first two dimensions).
    """
    if dataset_type.lower() == 'iris':
        iris = load_iris()
        data = iris.data[:, :2]  # Use first two features
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

def evaluate_file(file_path: str, dataset_type: str) -> float:
    """
    Load generated samples from a file, compute the FID-like score against real data,
    and display a scatter plot of the samples.
    
    Args:
        file_path (str): Path to the .npy file containing generated samples.
        dataset_type (str): Dataset type to load real data ('iris', '2moon', or 'spiral').
    
    Returns:
        float: The computed FID-like score.
    """
    gen_samples = np.load(file_path)
    # If there are more than 2 columns (e.g. class labels appended), discard them.
    if gen_samples.shape[1] > 2:
        gen_samples_eval = gen_samples[:, :2]
    else:
        gen_samples_eval = gen_samples
    
    real_data = load_real_data(dataset_type)
    fid_score = calculate_fid(real_data, gen_samples_eval)
    
    plt.figure(figsize=(5, 5))
    plt.scatter(gen_samples_eval[:, 0], gen_samples_eval[:, 1], c='blue', alpha=0.5, label='Generated Samples')
    plt.title(f"{os.path.basename(file_path)}\nFID Score: {fid_score:.4f}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return fid_score

def main():
    # List of generated sample files to evaluate
    files = [
        "mps_2moon_legendre.npy",
        "mps_2moon_fourier.npy",
        "mps_iris_legendre.npy",
        "mps_iris_fourier.npy",
        "mps_spiral_fourier.npy",
        "mps_spiral_legendre.npy",
        "tgan_2moon_legendre.npy",
        "tgan_2moon_fourier.npy",
        "tgan_iris_legendre.npy",
        "tgan_iris_fourier.npy",
        "tgan_spiral_fourier.npy",
        "tgan_spiral_legendre.npy"
    ]
    
    # Folder where the .npy files are stored
    data_folder = "./tests/data"
    
    results = {}
    for file_name in files:
        file_path = os.path.join(data_folder, file_name)
        # Determine dataset type based on filename
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
    
    print("FID Scores:")
    for fname, fid in results.items():
        print(f"{fname}: {fid:.4f}")

if __name__ == '__main__':
    main()
