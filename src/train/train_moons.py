import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import os
from sklearn.datasets import make_moons

from src.models.mps_super_ensemble import MPSSuperEnsemble
from src.models.discriminator import Discriminator
from src.models.TGAN import TGAN

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running on {device}')

# Generate 2 Moons dataset
N = 8000  # Number of samples
X, Y = make_moons(n_samples=N, noise=0.1, random_state=42)

# Normalize to [0,1]
X_min, X_max = X.min(axis=0), X.max(axis=0)
X = (X - X_min) / (X_max - X_min)

# Split into train and test
split = int(0.8 * N)
X_train, Y_train = X[:split], Y[:split]
X_test, Y_test = X[split:], Y[split:]

# Convert to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).to(device)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define model hyperparameters
n = 2  # Two features for 2 Moons
model = MPSSuperEnsemble(n=n, D=10, d=30, C=2, stddev=0.05, family='fourier')  # Change family as needed
model.to(device)
include_class = True

# Function to generate and save samples
def generate_and_save_samples(model, nsamples, filename):
    samples_list = []
    noise_vectors = torch.rand(nsamples, n, device=device)
    class_indices = np.random.randint(0, model.C, nsamples)
    
    for i in range(nsamples):
        samples_list.append(model.mps[class_indices[i]].sample(noise_vectors[i]))
    
    samples = torch.stack(samples_list).detach().cpu().numpy()
    samples_with_labels = np.hstack((samples, class_indices.reshape(-1, 1)))

    directory = os.path.dirname(filename)

    # Ensure the directory exists before saving
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    np.save(filename, samples_with_labels)
    
    # Plot generated samples
    plt.figure(figsize=(5, 5))
    plt.scatter(samples[:, 0], samples[:, 1], c=class_indices, cmap='viridis')
    plt.title(f"Generated Samples ({'Before' if 'pretrain' in filename else 'After'} GAN Training)")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(filename.replace('.npy', '.png'))
    plt.close()

# Pretrain the generator
print("Pretraining the 2 Moons model...")
model.train(train_loader, n_epochs=10, lr=1e-2, test_loader=test_loader, weight_decay=0e-3)

# Generate and save pre-GAN samples
generate_and_save_samples(
    model,
    nsamples=2000,
    filename=f'./data/tgan_2moon_{model.embedding.family}_pretrain.npy'
)

# Initialize and pretrain the discriminator
discriminator = Discriminator(n, include_class=include_class, device=device)
print("Pretraining Discriminator on 2 Moon dataset...")
discriminator.train(model, train_loader, n_epochs=10, lr=1e-2, test_loader=test_loader)

gan = TGAN(model, discriminator, include_class=include_class)

print("Training TGAN on 2 Moons dataset...")
gan.train(train_loader, num_epochs=3, lr=1e-3, test_loader=test_loader)

# Generate and save post-GAN samples
generate_and_save_samples(
    model,
    nsamples=2000,
    filename=f'./data/tgan_2moon_{model.embedding.family}_postgan.npy'
)
