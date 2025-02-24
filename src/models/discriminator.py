import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class Discriminator(nn.Module):
    """
    A simple fully connected Discriminator for GAN training.
    Classifies samples as real (1) or fake (0).
    """
    def __init__(self, input_size, hidden_size1=5, hidden_size2=5, include_class=False, device='cpu'):
        """
        Args:
            input_size (int): Dimension of input features.
            hidden_size1 (int): Number of neurons in the first hidden layer.
            hidden_size2 (int): Number of neurons in the second hidden layer.
            include_class (bool): Whether to include class labels as input features.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        super(Discriminator, self).__init__()
        self.device = device
        self.include_class = include_class
        
        if include_class:
            input_size += 1
        
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass for the Discriminator.

        Args:
            x (Tensor): Input data of shape (batch_size, input_size)

        Returns:
            Tensor: Sigmoid output indicating probability of real vs. fake
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Ensure output is between [0,1]
        return x
    
    def train_discriminator(self, mps_super, data_loader, n_epochs=10, lr=0.01, test_loader=None, weight_decay=1e-5):
        """
        Train the Discriminator on real and fake samples.

        Args:
            mps_super (MPSsuper): The generator model.
            data_loader (DataLoader): DataLoader for training data.
            n_epochs (int): Number of training epochs.
            lr (float): Learning rate.
            test_loader (DataLoader, optional): DataLoader for validation/test set.
            weight_decay (float): L2 weight regularization.
        """
        self.to(self.device)
        loss_fn = nn.BCEWithLogitsLoss()  # Use BCE with logits for better numerical stability
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        n = data_loader.dataset[0][0].shape[0]  # Input size
        b = data_loader.batch_size  # Batch size

        for epoch in range(n_epochs):  
            epoch_accuracy = 0.0
            for i, (inputs, labels) in enumerate(data_loader):
                x_real = inputs.to(self.device)
                labels_real = labels.to(self.device)
                y_real = torch.ones(b, 1, device=self.device)  # Real samples = 1

                # Generate fake samples
                x_fake = torch.zeros(b, n, device=self.device)
                labels_fake = torch.randint(0, mps_super.C, (b,), device=self.device)
                noise_vectors = torch.rand(b, n, device=self.device)

                for j in range(b):
                    x_fake[j] = mps_super.mps[labels_fake[j]].sample(noise_vectors[j])

                y_fake = torch.zeros(b, 1, device=self.device)  # Fake samples = 0

                # Combine real and fake samples
                x = torch.cat((x_real, x_fake), dim=0)
                y = torch.cat((y_real, y_fake), dim=0)
                labels = torch.cat((labels_real, labels_fake), dim=0)

                if self.include_class:
                    x = torch.hstack((x, labels.reshape(-1, 1)))

                # Shuffle data
                permuted_indices = torch.randperm(2 * b)
                x, y = x[permuted_indices], y[permuted_indices]

                # Train Discriminator
                optimizer.zero_grad()
                self.zero_grad()
                prediction = self(x)
                loss = loss_fn(prediction, y)
                loss.backward()
                optimizer.step()

                # Compute accuracy
                correct = ((prediction > 0.5).float() == y).sum().item()
                epoch_accuracy += correct / len(y)

                # Print intermediate progress
                if i % 100 == 0 and len(data_loader.dataset) / data_loader.batch_size > 10 and i > 0:
                    print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}, Accuracy: {epoch_accuracy/(i+1):.4f}")

            epoch_accuracy /= len(data_loader)
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Accuracy: {epoch_accuracy:.4f}")

