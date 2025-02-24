import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from datasets.synthetic import generate_moons_dataset
from models.mps_super_ensemble import MPSSuperEnsemble
from src.models.discriminator import Discriminator
from src.models.TGAN import TGAN

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running on {device}')

# Generate Two Moons dataset
X, y = generate_moons_dataset(n_samples=10000, noise=0.1)
# For testing, generate a smaller dataset
X_test, y_test = generate_moons_dataset(n_samples=1000, noise=0.1)

# Convert to torch tensors
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# Create DataLoaders
batch_size = 32
train_dataset = TensorDataset(X_tensor, y_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define model hyperparameters
n = 2  # two features for Two Moons
model = MPSSuperEnsemble(n=n, D=5, d=10, C=2, stddev=0.2, family='legendre')
model.to(device)

# Train the MPS-based classifier on Two Moons
print("Training the Two Moons model...")
model.train_model(train_loader, n_epochs=10, test_loader=test_loader, lr=0.01, weight_decay=0e-3)

# Generate samples from each class
nsamp = 1000
samples_list = []
noise_vectors = torch.rand(nsamp, n, device=device)
for i in range(nsamp):
    samples_list.append(model.mps[0].sample(noise_vectors[i]) if y_tensor[i % len(y_tensor)] == 0 
                        else model.mps[1].sample(noise_vectors[i]))
samples = torch.stack(samples_list).detach().cpu().numpy()
# For visualization, combine with class labels
classes = y[:nsamp].reshape(-1, 1)
plt.figure(figsize=(5,5))
plt.scatter(samples[:, 0], samples[:, 1], c=classes.squeeze(), cmap='viridis')
plt.title("Two Moons Generated Samples")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()

# Save generated samples
np.save('./data/tgan_2moon_legendre.npy', np.hstack((samples, classes)))
