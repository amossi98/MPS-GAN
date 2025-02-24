import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from src.models.mps_super_ensemble import MPSSuperEnsemble
from src.models.discriminator import Discriminator
from src.models.TGAN import TGAN
from src.datasets.synthetic import load_iris_dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running on {device}')

# Load and split Iris dataset
iris_data, iris_target = load_iris_dataset()
X_train, X_test, Y_train, Y_test = train_test_split(iris_data, iris_target, test_size=0.2, random_state=42)

# Convert to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.long).to(device)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define model hyperparameters
n = 4  # number of features
model = MPSSuperEnsemble(n=n, D=10, d=10, C=3, stddev=0.1, family='legendre')
model.to(device)

# Pretrain the generator (if desired) and then train the model
print("Training the Iris model...")
model.train_model(train_loader, n_epochs=50, test_loader=test_loader, lr=1e-2, weight_decay=0e-5)

# Optionally generate samples from the trained model
nsamp = 150
samples = torch.zeros((nsamp, n), device=device)
noise_vec = torch.rand(nsamp, n, device=device)
# Randomly choose a class for each sample
class_indices = np.random.randint(0, model.C, nsamp)
for i in range(nsamp):
    samples[i] = model.mps[class_indices[i]].sample(noise_vec[i])
samples_np = samples.cpu().detach().numpy()

# Plot results (using first two features for visualization)
plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap='viridis', label='Train Data')
plt.scatter(samples_np[:, 0], samples_np[:, 1], c=class_indices, marker='x', label='Generated Samples')
plt.legend()
plt.title("Iris Experiment")
plt.show()

# Save generated samples
np.save('./data/tgan_iris_legendre.npy', np.hstack((samples_np, class_indices.reshape(-1, 1))))
