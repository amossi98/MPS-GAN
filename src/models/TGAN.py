import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set device (assuming CPU for now)
device = 'cpu'

class TGAN(nn.Module):
    """
    TGAN: A Tensor Network GAN model that combines a supervised MPS-based generator
    (MPSSuperEnsemble) with a discriminator to improve generative performance.
    
    Optionally, the generated samples can have class labels appended as extra features.
    """
    def __init__(self, mps_super, discriminator, include_class=False):
        """
        Args:
            mps_super (nn.Module): The generator model (e.g. an instance of MPSSuperEnsemble).
            discriminator (nn.Module): The discriminator network.
            include_class (bool): Whether to append class labels to the generated samples.
        """
        super(TGAN, self).__init__()
        self.mps_super = mps_super
        self.discriminator = discriminator
        self.include_class = include_class
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, noise_vector, class_label):
        """
        Forward pass: generate a sample using the generator and pass it through the discriminator.
        
        Args:
            noise_vector (Tensor): Noise tensor of shape (n,) used for sampling.
            class_label (int or Tensor): The target class index for generation.
        
        Returns:
            Tuple[Tensor, Tensor]: Generated sample and discriminator output.
        """
        # Generate a sample from mps_super (generator)
        generated_sample = self.mps_super.sample(n_samples=1, class_idx=class_label, noise=noise_vector)
        # Optionally append the class label as an extra feature
        if self.include_class:
            class_tensor = torch.tensor([class_label], dtype=generated_sample.dtype, device=generated_sample.device)
            generated_sample = torch.hstack((generated_sample, class_tensor))
        disc_output = self.discriminator(generated_sample)
        return generated_sample, disc_output

    def train(self, data_loader, num_epochs, lr=1e-2, test_loader=None, threshold_training=0.01):
        """
        Custom training routine for TGAN. Alternates between updating the generator (mps_super)
        and the discriminator. If the generator's classification performance (via mps_super)
        drops below a set threshold compared to its original test accuracy, the generator is retrained.
        
        Args:
            data_loader: DataLoader for training data.
            num_epochs (int): Number of training epochs.
            lr (float): Learning rate.
            test_loader: DataLoader for validation/test data.
            threshold_training (float): Acceptable relative drop in test accuracy before retraining.
        """
        loss_fn = nn.BCELoss()
        
        # Compute original test accuracy of mps_super on the classification task.
        original_test_accuracy = 0.0
        if test_loader is not None:
            for inputs, labels in test_loader:
                output = self.mps_super(inputs)
                original_test_accuracy += (output.argmax(dim=1) == labels).sum().item() / len(labels)
            original_test_accuracy /= len(test_loader)
            print(f"Original test accuracy: {original_test_accuracy:.4f}")
        
        batch_size = data_loader.batch_size
        
        optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr)
        optimizer_G = optim.Adam(self.mps_super.parameters(), lr=lr)
        
        for epoch in range(num_epochs):
            for batch_idx, (real_samples, real_classes) in enumerate(data_loader):
                # Create real and fake labels for BCE loss
                real_labels = torch.ones(batch_size, 1, device=device)
                fake_labels = torch.zeros(batch_size, 1, device=device)
                
                # Generate fake samples: for each batch element, choose a random class and generate a sample.
                fake_classes = np.random.randint(0, self.mps_super.C, batch_size)
                fake_samples = torch.zeros(batch_size, self.mps_super.n, device=device)
                noise_vectors = torch.rand(batch_size, self.mps_super.n, device=device)
                
                for i, cls in enumerate(fake_classes):
                    fake_samples[i] = self.mps_super.sample(n_samples=1, class_idx=cls, noise=noise_vectors[i].unsqueeze(0))
                
                if self.include_class:
                    fake_class_tensor = torch.tensor(fake_classes, device=device).unsqueeze(1).float()
                    real_class_tensor = torch.tensor(real_classes, device=device).unsqueeze(1).float()
                    fake_samples = torch.hstack((fake_samples, fake_class_tensor))
                    real_samples = torch.hstack((real_samples, real_class_tensor))
                
                # Discriminator output on real samples
                real_outputs = self.discriminator(real_samples)
                
                # Update Generator: try to fool the discriminator
                optimizer_G.zero_grad()
                gen_outputs = self.discriminator(fake_samples)
                loss_G = self.loss_fn(gen_outputs, real_labels)
                loss_G.backward()
                optimizer_G.step()
                
                # Update Discriminator: correctly classify real vs. fake samples
                optimizer_D.zero_grad()
                fake_outputs = self.discriminator(fake_samples.detach())
                loss_D_real = self.loss_fn(real_outputs, real_labels)
                loss_D_fake = self.loss_fn(fake_outputs, fake_labels)
                loss_D = (loss_D_real + loss_D_fake) / 2
                loss_D.backward(retain_graph=True)
                optimizer_D.step()
                
                # Evaluate test accuracy on mps_super (if test_loader provided)
                test_accuracy = 0.0
                if test_loader is not None:
                    for inputs, labels in test_loader:
                        output = self.mps_super(inputs)
                        test_accuracy += (output.argmax(dim=1) == labels).sum().item() / len(labels)
                    test_accuracy /= len(test_loader)
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(data_loader)}], "
                          f"Discriminator Loss: {loss_D.item():.4f}, Generator Loss: {loss_G.item():.4f}, "
                          f"Test Accuracy: {test_accuracy:.4f}")
            
            # Retrain mps_super if its test accuracy drops below the threshold.
            if test_loader is not None and test_accuracy < original_test_accuracy * (1 - threshold_training):
                print(f"Retraining generator (mps_super), Test Accuracy: {test_accuracy:.4f}")
                self.mps_super.train(data_loader, n_epochs=5, test_loader=test_loader, lr=lr, weight_decay=1e-5)
                # Optionally, retrain the discriminator if it has a similar train method.
                if hasattr(self.discriminator, 'train_model'):
                    self.discriminator.train(self.mps_super, data_loader, 1, lr=lr, test_loader=test_loader)
                # Recompute test accuracy after retraining.
                test_accuracy = 0.0
                for inputs, labels in test_loader:
                    output = self.mps_super(inputs)
                    test_accuracy += (output.argmax(dim=1) == labels).sum().item() / len(labels)
                test_accuracy /= len(test_loader)
            
            # Optionally retrain discriminator every epoch if such a method exists.
            if hasattr(self.discriminator, 'train_model'):
                self.discriminator.train(self.mps_super, data_loader, 3, lr=lr, test_loader=test_loader)
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Final Test Accuracy: {test_accuracy:.4f}, "
                  f"Real Acc: {(real_outputs > 0.5).float().mean().item():.4f}, "
                  f"Fake Acc: {(fake_outputs < 0.5).float().mean().item():.4f}")

