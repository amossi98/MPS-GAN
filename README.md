# MPS-GAN

**A Matrix Product State (MPS) Framework for Simultaneous Classification and Generation**

MPS-GAN is a research-focused project that implements a quantum-inspired Matrix Product State model capable of both classifying data and generating new samples. By leveraging a novel GAN-style training approach, the model improves generative performance while maintaining robust classification accuracy.

## Overview

This repository contains the implementation of a Matrix Product State model as described in the [master’s thesis](https://link.springer.com/epdf/10.1007/s42484-025-00272-6?sharing_token=O-VFAiCLkJylIHPWSRqVk_e4RwlQNchNByi7wbcMAY5EzDoWhNPtTqOhCRWP1_KKfZzuucRwp64qzGpENIEKYfF03TRv0M2W-_fUgTT0fuyDuGbeGkHYeiDtnoJPd5kPTJXQBmnVQAGa5dWgwC7NkV2TSRkHxchKmFGeci-8QVM%3D). The model:
- Uses tensor networks for data representation and processing.
- Implements advanced embedding functions (e.g., Fourier and Legendre embeddings).
- Trains with both standard cross-entropy loss for classification and adversarial loss in a GAN-style setup for generation.
- Provides an exact sampling method to generate data from non-normalized MPS outputs.

## Repository Structure

- **README.md**: Project overview, installation instructions, and usage examples.
- **docs/**: Documentation including the thesis.
- **src/**: Source code for the project.
  - **models/**: Core MPS model implementation.
  - **train/**: Scripts for training the model (both classification and GAN-style).
  - **utils/**: Helper functions and utilities.
  - **analysis/**: Further studies on the effect of some hyperparameters (bond dimension, number of bins, ...) on performances, quality, and speed.
- **tests/**: Unit tests to measure the quality of output samples in the data folder.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/amossi98/MPS-GAN.git
   cd MPS-GAN
   ```


3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv env
   source env/bin/activate # For Linux/MacOS
   env\Scripts\activate # For Windows
   pip install -r requirements.txt
   ```


## Usage

### Train Models
You can train different models using the provided scripts. The training process includes:

Pretraining an MPS-based generator
Training a TGAN (MPS + Discriminator GAN)
Saving generated samples before and after GAN training
1. Train on 2 Moons
   ```bash
   python -m src.train.train_moons
   ```
2. Train on Spiral
   ```bash
   python -m src.train.train_spiral
   ```
3. Train on Iris
   ```bash
   python -m src.train.train_iris
   ```
   
Each script will save intermediate results and generate .npy files with sample data.

### Evaluate Generated Samples
Once the models are trained, evaluate the quality of generated samples using an FID-like metric:

```bash
python -m src.tests.evaluate_generated_samples
```
This will compute FID scores and visualize scatter plots of generated vs. real samples.

### Perturbation & Noise Analysis
You can analyze how embedding perturbations affect inference performance:
```bash
python -m src.analysis.test_perturbation_inference
```

Or study how perturbations during training influences results:
```bash
python -m src.analysis.test_perturbations_training
```

Both scripts generate plots comparing different embedding families (e.g., Fourier vs. Legendre).

### Saving and loaging models

```python
import torch
from src.models.mps_super_ensemble import MPSSuperEnsemble

# Create a model instance with some example parameters
model = MPSSuperEnsemble(n=10, D=5, d=2, C=2, stddev=0.5, family='fourier', sigma=0)

# (Optional: Train your model here or perform some operations)

# Save the model to a checkpoint file
model.save("checkpoint.pth")
print("Model saved successfully.")

# Load the model from the checkpoint file
loaded_model = MPSSuperEnsemble.load("checkpoint.pth", device='cpu')
print("Model loaded successfully.")
```

### Sampling New Data
To generate new samples from a trained model:
```python
import torch
from src.models.mps_super_ensemble import MPSSuperEnsemble

model = MPSSuperEnsemble.load("path/to/checkpoint.pth")  # Load trained model
noise_vector = torch.rand(1, model.n)  # Generate noise
sample = model.sample()  # Generate a new sample
```


## Results

The results and comprehensive analysis, including visualizations of the generated samples and detailed evaluation metrics, are available in the original PDF file of the thesis. You can access the document via the provided [arXiv link](https://arxiv.org/html/2406.17441v1).


## Contact

For questions or further discussion, please contact **Alex Mossi**.
