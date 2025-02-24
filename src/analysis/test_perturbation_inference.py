"""
test_perturbation_inference.py

This script investigates how varying the embedding parameter 'sigma' 
affects the inference accuracy of an MPS-based model on a synthetic spiral dataset.
It trains separate models with 'fourier' and 'legendre' embeddings and plots the validation accuracy.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from src.models.emb import Emb
import src.models.mps_super_ensemble as mps_super_ensemble # Assumes this module defines the MPSsuper class

def generate_spiral_dataset(N=8000):
    """Generate a synthetic spiral dataset."""
    pi = np.pi
    theta = np.sqrt(np.random.rand(N)) * 2 * pi
    r_a = 2 * theta + pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    x_a = data_a + np.random.randn(N, 2)
    
    r_b = -2 * theta - pi
    data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
    x_b = data_b + np.random.randn(N, 2)
    
    x_a = x_a / 20.
    x_b = x_b / 20.
    x = np.vstack((x_a, x_b))
    x = (x - x.min()) / (x.max() - x.min())
    y = np.vstack((np.zeros((N, 1)), np.ones((N, 1))))
    data = np.hstack((x, y))
    np.random.shuffle(data)
    
    X_train = data[:N // 2, :2]
    Y_train = data[:N // 2, 2]
    X_test = data[N // 2:, :2]
    Y_test = data[N // 2:, 2]
    return X_train, Y_train, X_test, Y_test

def main():
    device = 'cpu'
    N = 8000
    X_train, Y_train, X_test, Y_test = generate_spiral_dataset(N)

    # Create DataLoaders
    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    Y_train_tensor = torch.from_numpy(Y_train).float().to(device)
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)

    X_test_tensor = torch.from_numpy(X_test).float().to(device)
    Y_test_tensor = torch.from_numpy(Y_test).float().to(device)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    n = 2
    n_sigma = 100
    sigma_values = np.linspace(0, 2, n_sigma)
    acc_fourier = np.zeros(n_sigma)
    acc_legendre = np.zeros(n_sigma)

    # Evaluate models with Fourier embedding
    for j, sigma in enumerate(sigma_values):
        model_fourier = mps_super_ensemble.MPSsuper(n=n, D=50, d=20, C=2, stddev=0.1, family='fourier')
        model_fourier.to(device)
        model_fourier.train(train_loader, n_epochs=30, lr=5e-3, test_loader=test_loader, weight_decay=0e-3)
        output = model_fourier(X_test_tensor)
        preds = output.argmax(-1)
        acc_fourier[j] = preds.eq(Y_test_tensor).float().mean().item()
        print(f"Fourier sigma={sigma:.4f}, accuracy={acc_fourier[j]:.4f}")

    # Evaluate models with Legendre embedding
    for j, sigma in enumerate(sigma_values):
        model_legendre = mps_super_ensemble.MPSsuper(n=n, D=50, d=20, C=2, stddev=0.1, family='legendre')
        model_legendre.to(device)
        model_legendre.train(train_loader, n_epochs=30, lr=5e-3, test_loader=test_loader, weight_decay=0e-3)
        output = model_legendre(X_test_tensor)
        preds = output.argmax(-1)
        acc_legendre[j] = preds.eq(Y_test_tensor).float().mean().item()
        print(f"Legendre sigma={sigma:.4f}, accuracy={acc_legendre[j]:.4f}")

    # Plot validation accuracy vs. sigma
    plt.figure(figsize=(5, 5))
    plt.ylim(0.5, 1)
    plt.plot(sigma_values, acc_fourier, marker='o', linestyle='-', label='Fourier')
    plt.plot(sigma_values, acc_legendre, marker='o', linestyle='-', label='Legendre')
    plt.xlabel('Sigma')
    plt.ylabel('Validation Accuracy')
    plt.title('Impact of Embedding Perturbation on Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
