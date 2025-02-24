import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from src.models.emb import Emb
from src.models.mps import MPS
from src.models.utils import EarlyStopper  # Assuming EarlyStopper is defined in utils

# Device selection
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print(f'running on {device}')

class MPSSuperEnsemble(nn.Module):
    """
    Super Ensemble of MPS modules for supervised classification.
    This model aggregates an ensemble of MPS modules (one per class).
    """
    def __init__(self, n, D, d=2, C=2, stddev=0.5, family='fourier', sigma=0):
        """
        Args:
            n (int): Input size.
            D (int): Bond dimension.
            d (int): Embedding dimension.
            C (int): Number of classes.
            stddev (float): Standard deviation for initializing MPS tensors.
            family (str): Embedding family.
            sigma (float): Additional parameter for embedding.
        """
        super(MPSSuperEnsemble, self).__init__()
        self.n = n
        self.D = D
        self.d = d
        self.C = C
        self.embedding = Emb(d, family=family, sigma=sigma)
        # Create one MPS instance per class
        self.mps = nn.ModuleList([MPS(n=n, D=D, d=d, stddev=stddev, family=family) for _ in range(C)])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        """
        Forward pass for classification.
        
        Args:
            inputs (Tensor): Input data.
        Returns:
            Tensor: Logits of shape (batch_size, C)
        """
        b = inputs.shape[0]
        # Flatten and embed the input using the shared embedding
        x = inputs.view(b, -1)
        x = self.embedding(x)
        y = torch.zeros(b, self.C, device=x.device)
        for i in range(self.C):
            y[:, i] = self.mps[i](x)
        y = torch.pow(y, 2)  # Square the outputs as per original design
        return y

    def train_model(self, data_loader, n_epochs, test_loader=None, lr=0.01, weight_decay=1e-5, early_stopping=False):
        """
        Train the ensemble model using cross-entropy loss.
        
        Args:
            data_loader: Training data loader.
            n_epochs (int): Number of epochs.
            test_loader: Validation data loader.
            lr (float): Learning rate.
            weight_decay (float): Regularization parameter.
            early_stopping (bool): Whether to use early stopping.
        Returns:
            Tuple: (training_accuracy, validation_accuracy)
        """
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, verbose=True, mode='min', min_lr=1e-6)
        early_stopper = EarlyStopper(patience=20, min_delta=1e-3)

        for epoch in range(n_epochs):
            epoch_accuracy = 0.0
            for i, (inputs, labels) in enumerate(data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device, dtype=torch.long)
                optimizer.zero_grad()
                output = self(inputs)
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()
                epoch_accuracy += (output.argmax(dim=1) == labels).sum().item() / len(labels)
                if i % 100 == 0 and i > 0:
                    print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}, Accuracy: {epoch_accuracy/(i+1):.4f}")
            epoch_accuracy /= len(data_loader)
            test_accuracy = 0.0
            if test_loader is not None:
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs = inputs.to(device)
                        labels = labels.to(device, dtype=torch.long)
                        output = self(inputs)
                        test_accuracy += (output.argmax(dim=1) == labels).sum().item() / len(labels)
                test_accuracy /= len(test_loader)
            else:
                test_accuracy = 0.0
            print(f"Epoch {epoch+1}, Training Accuracy: {epoch_accuracy:.4f}, Validation Accuracy: {test_accuracy:.4f}")
            scheduler.step(test_accuracy)
            if early_stopping and early_stopper.early_stop(test_accuracy):
                print("Early stopping triggered. Validation Accuracy:", test_accuracy)
                break
        return epoch_accuracy, test_accuracy

    def sample(self, n_samples=1, class_idx=0, noise=None):
        """
        Generate samples from a specific class (MPS module).
        
        Args:
            n_samples (int): Number of samples to generate.
            class_idx (int): Index of the class to sample from.
            noise (Tensor): Optional noise tensor.
        Returns:
            Tensor: Generated samples of shape (n_samples, n)
        """
        samples = torch.zeros(n_samples, self.n, device=self.mps[0].tensor.device)
        if noise is None:
            noise = torch.rand(n_samples, self.n)
        for i in range(n_samples):
            samples[i] = self.mps[class_idx].sample(noise[i])
        return samples
