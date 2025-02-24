"""
test_noise_training.py

This script examines the effect of introducing noise during training on model performance 
and latent space trajectories. Using a synthetic spiral dataset, it trains an MPS model under 
different noise levels (controlled by sigma) and visualizes both the validation accuracy and 
the interpolation trajectories in the latent space.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from numpy import pi
from src.models.mps_super_ensemble import MPSsuper

def generate_spiral_dataset(N=8000):
    """Generate a synthetic spiral dataset."""
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

    # Prepare DataLoaders
    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    Y_train_tensor = torch.from_numpy(Y_train).float().to(device)
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    X_test_tensor = torch.from_numpy(X_test).float().to(device)
    Y_test_tensor = torch.from_numpy(Y_test).to(device).float()
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=400, shuffle=True)

    # Lists to store validation accuracy for different noise levels (sigma)
    accuracy_legendre = []
    sigmas = []
    n_sigmas = 2  # Adjust as needed for finer granularity

    for i in range(n_sigmas):
        sigma = i / n_sigmas * 2
        sigmas.append(sigma)
        # Create and train an MPS model with a given sigma and Legendre embedding
        model = MPSsuper(n=2, D=50, d=20, C=2, stddev=0.1, sigma=sigma, family='legendre')
        model.to(device)
        model.train(train_loader, n_epochs=30, lr=0.001, test_loader=test_loader, weight_decay=1e-3, early_stopping=False)
        acc = model(X_test_tensor).argmax(-1).eq(Y_test_tensor).float().mean().item()
        accuracy_legendre.append(acc)
        print(f"Legendre sigma = {sigma:.4f}, accuracy = {acc:.4f}")

        # Visualize latent space interpolation trajectories
        ntrajectories = 2
        steps = 500
        samples = torch.zeros(ntrajectories, steps, 2)
        samples2 = torch.zeros(ntrajectories, steps, 2)
        extremes = torch.zeros(ntrajectories, 3, 2)
        
        for j in range(ntrajectories):
            # Define fixed noise trajectories for reproducibility
            n1 = torch.tensor((0.1, 0.1))
            n2 = torch.tensor((0.1, 0.9))
            n3 = torch.tensor((0.9, 0.9))
            for k in range(steps):
                noise1 = n1 * (k / steps) + n2 * (1 - k / steps)
                samples[j, k] = model.mps[j % 2].sample(noise1)
                if k == 0:
                    extremes[j, 0, :] = samples[j, k]
                elif k == steps - 1:
                    extremes[j, 1, :] = samples[j, k]
                
                noise2 = n2 * (k / steps) + n3 * (1 - k / steps)
                samples2[j, k] = model.mps[j % 2].sample(noise2)
                if k == 0:
                    extremes[j, 1, :] = samples2[j, k]
                elif k == steps - 1:
                    extremes[j, 2, :] = samples2[j, k]
        
        samples_np = samples.view(-1, 2).detach().cpu().numpy()
        samples2_np = samples2.view(-1, 2).detach().cpu().numpy()
        extremes_np = extremes.view(-1, 2).detach().cpu().numpy()

        plt.figure(figsize=(5, 5))
        plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap='viridis', alpha=0.5)
        plt.scatter(samples_np[:, 0], samples_np[:, 1], c='blue', marker='o', label='Trajectory 1')
        plt.scatter(samples2_np[:, 0], samples2_np[:, 1], c='green', marker='x', label='Trajectory 2')
        plt.scatter(extremes_np[:, 0], extremes_np[:, 1], c='red', marker='D', label='Extremes')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.title(f"Latent Trajectories (sigma = {sigma:.2f})")
        plt.legend()
        plt.show()

    # Plot validation accuracy vs sigma for Legendre embedding
    plt.figure(figsize=(5, 5))
    plt.plot(sigmas, accuracy_legendre, marker='o', linestyle='-', label='Legendre')
    plt.xlabel('Sigma')
    plt.ylabel('Validation Accuracy')
    plt.title('Impact of Noise During Training (Legendre)')
    plt.ylim(0.5, 1)
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
