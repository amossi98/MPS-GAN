"""
test_noise_training_comparison.py

This script examines the effect of introducing noise during training on model performance,
and compares two different embedding families—Fourier and Legendre—on a synthetic spiral dataset.
For a range of sigma values (noise levels), it trains separate models and records the validation accuracy.
Finally, it plots the comparison between the two embedding families.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from numpy import pi
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
    
    n = 2         # two features in the spiral dataset
    D = 50
    d = 20
    C = 2
    stddev = 0.1
    n_sigma = 10  # for demonstration; increase for finer granularity
    sigma_values = np.linspace(0, 2, n_sigma)
    acc_fourier = np.zeros(n_sigma)
    acc_legendre = np.zeros(n_sigma)
    
    for idx, sigma in enumerate(sigma_values):
        print(f"\n=== Sigma = {sigma:.4f} ===")
        # Train Fourier model
        print("Training Fourier model...")
        model_fourier = MPSSuperEnsemble(n=n, D=D, d=d, C=C, stddev=stddev, family='fourier', sigma=sigma)
        model_fourier.to(device)
        model_fourier.train(train_loader, n_epochs=30, lr=5e-3, test_loader=test_loader, weight_decay=0e-3)
        output_fourier = model_fourier(X_test_tensor)
        preds_fourier = output_fourier.argmax(-1)
        acc_fourier[idx] = preds_fourier.eq(Y_test_tensor).float().mean().item()
        print(f"Fourier model: sigma = {sigma:.4f}, accuracy = {acc_fourier[idx]:.4f}")
        
        # Train Legendre model
        print("Training Legendre model...")
        model_legendre = MPSSuperEnsemble(n=n, D=D, d=d, C=C, stddev=stddev, family='legendre', sigma=sigma)
        model_legendre.to(device)
        model_legendre.train(train_loader, n_epochs=30, lr=5e-3, test_loader=test_loader, weight_decay=0e-3)
        output_legendre = model_legendre(X_test_tensor)
        preds_legendre = output_legendre.argmax(-1)
        acc_legendre[idx] = preds_legendre.eq(Y_test_tensor).float().mean().item()
        print(f"Legendre model: sigma = {sigma:.4f}, accuracy = {acc_legendre[idx]:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(5, 5))
    plt.plot(sigma_values, acc_fourier, marker='o', linestyle='-', label='Fourier')
    plt.plot(sigma_values, acc_legendre, marker='o', linestyle='-', label='Legendre')
    plt.xlabel('Sigma (Noise Level)', fontsize=14)
    plt.ylabel('Validation Accuracy', fontsize=14)
    plt.title('Impact of Training Noise on Accuracy', fontsize=16)
    plt.ylim(0.5, 1)
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
