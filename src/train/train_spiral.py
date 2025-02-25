import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import os

from src.datasets.synthetic import generate_spiral_dataset
from src.models.mps_super_ensemble import MPSSuperEnsemble
from src.models.discriminator import Discriminator
from src.models.TGAN import TGAN

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running on {device}')

# Generate Spiral dataset
X_train, Y_train, X_test, Y_test = generate_spiral_dataset(N=8000)

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
n = 2  # two features for the spiral data
model = MPSSuperEnsemble(n=n, D=30, d=30, C=2, stddev=0.05, family='fourier')  # Change family for different embeddings
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
    
    # Optional plotting
    plt.figure(figsize=(5,5))
    plt.scatter(samples[:, 0], samples[:, 1], c=class_indices, cmap='viridis')
    plt.title(f"Generated Samples ({'Before' if 'pretrain' in filename else 'After'} GAN Training)")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(filename.replace('.npy', '.png'))
    plt.close()

# Pretrain the generator
print("Pretraining the Spiral model...")
model.train(train_loader, n_epochs=10, lr=1e-2, test_loader=test_loader, weight_decay=0e-3)

# Generate and save pre-GAN samples
generate_and_save_samples(
    model, 
    nsamples=2000,
    filename=f'./data/tgan_spiral_{model.embedding.family}_pretrain.npy'
)

# Create GAN and train
discriminator = Discriminator(n, include_class=include_class)
gan = TGAN(model, discriminator, include_class=include_class)

print("Training TGAN on Spiral dataset...")
gan.train_tgan(train_loader, num_epochs=5, lr=1e-3, test_loader=test_loader)

# Generate and save post-GAN samples
generate_and_save_samples(
    model,
    nsamples=2000,
    filename=f'./data/tgan_spiral_{model.embedding.family}_postgan.npy'
)