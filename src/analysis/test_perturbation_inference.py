"""
test_perturbation_inference.py

This script evaluates how adding inference noise (sigma) to the embedding
affects the accuracy of an MPS-based classifier on a synthetic spiral dataset.
It trains two separate models:
  - One using Fourier embedding.
  - One using Legendre embedding.
Then it replaces their embedding with new ones that add varying noise levels,
collects accuracy for each sigma, and plots the results for comparison.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from src.datasets.synthetic import generate_spiral_dataset
from src.models.mps_super_ensemble import MPSSuperEnsemble
from src.models.emb import Emb

def main():
    device = 'cpu'
    N = 8000
    # Generate synthetic spiral dataset
    X_train, Y_train, X_test, Y_test = generate_spiral_dataset(N)
    
    # Create DataLoaders
    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    Y_train_tensor = torch.from_numpy(Y_train).float().to(device)
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    
    X_test_tensor = torch.from_numpy(X_test).float().to(device)
    Y_test_tensor = torch.from_numpy(Y_test).float().to(device)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    n = 2  # Two features in the spiral dataset
    n_sigma = 100
    sigma_values = np.linspace(0, 2, n_sigma)
    acc_fourier = np.zeros(n_sigma)
    acc_legendre = np.zeros(n_sigma)
    
    # ------------------------
    # Train and evaluate Fourier model
    # ------------------------
    print("Training Fourier model...")
    model_fourier = MPSSuperEnsemble(n=n, D=50, d=20, C=2, stddev=0.1, family='fourier')
    model_fourier.to(device)
    model_fourier.train(train_loader, n_epochs=30, lr=5e-3, test_loader=test_loader, weight_decay=0e-3)
    
    # Evaluate inference with different noise levels (sigma) for Fourier embedding
    for j, sigma in enumerate(sigma_values):
        # Replace the embedding with one that adds noise sigma during inference.
        model_fourier.embedding = Emb(d=20, sigma=sigma, family='fourier')
        output = model_fourier(X_test_tensor)
        preds = output.argmax(-1)
        acc_fourier[j] = preds.eq(Y_test_tensor).float().mean().item()
        print(f"Fourier: sigma={sigma:.4f}, accuracy={acc_fourier[j]:.4f}")
    
    # ------------------------
    # Train and evaluate Legendre model
    # ------------------------
    print("Training Legendre model...")
    model_legendre = MPSSuperEnsemble(n=n, D=50, d=20, C=2, stddev=0.1, family='legendre')
    model_legendre.to(device)
    model_legendre.train(train_loader, n_epochs=30, lr=5e-3, test_loader=test_loader, weight_decay=0e-3)
    
    # Evaluate inference with different noise levels (sigma) for Legendre embedding
    for j, sigma in enumerate(sigma_values):
        model_legendre.embedding = Emb(d=20, sigma=sigma, family='legendre')
        output = model_legendre(X_test_tensor)
        preds = output.argmax(-1)
        acc_legendre[j] = preds.eq(Y_test_tensor).float().mean().item()
        print(f"Legendre: sigma={sigma:.4f}, accuracy={acc_legendre[j]:.4f}")
    
    # ------------------------
    # Plot results
    # ------------------------
    plt.figure(figsize=(5, 5))
    plt.ylim(0.5, 1)
    plt.plot(sigma_values, acc_fourier, marker='o', linestyle='-', label='Fourier')
    plt.plot(sigma_values, acc_legendre, marker='o', linestyle='-', label='Legendre')
    plt.xlabel('Sigma (Inference Noise Level)', fontsize=14)
    plt.ylabel('Validation Accuracy', fontsize=14)
    plt.title('Impact of Inference Noise on Accuracy', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
