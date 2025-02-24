# MPS-GAN

**A Matrix Product State (MPS) Framework for Simultaneous Classification and Generation**

MPS-GAN is a research-focused project that implements a quantum-inspired Matrix Product State model capable of both classifying data and generating new samples. By leveraging a novel GAN-style training approach, the model improves generative performance while maintaining robust classification accuracy.

## Overview

This repository contains the implementation of a Matrix Product State model as described in the [master’s thesis](docs/thesis.pdf). The model:
- Uses tensor networks for data representation and processing.
- Implements advanced embedding functions (e.g., Fourier and Legendre embeddings).
- Trains with both standard cross-entropy loss for classification and adversarial loss in a GAN-style setup for generation.
- Provides an exact sampling method to generate data from non-normalized MPS outputs.

## Repository Structure

- **README.md**: Project overview, installation instructions, and usage examples.
- **docs/**: Documentation including the thesis, supplementary materials, and experiment results.
- **src/**: Source code for the project.
  - **models/**: Core MPS model implementation.
  - **training/**: Scripts for training the model (both classification and GAN-style).
  - **sampling/**: Implementation of the exact sampling algorithm.
  - **data/**: Data preprocessing and loader scripts.
  - **utils/**: Helper functions and utilities.
- **experiments/**: Configuration files, logs, and generated figures.
- **notebooks/**: Jupyter notebooks for interactive exploration and visualization.
- **tests/**: Unit tests to ensure code reliability.

## Installation

1. Clone the repository:
git clone https://github.com/yourusername/MPS-GAN.git
cd MPS-GAN


2. Create a virtual environment and install dependencies:
python -m venv env
source env/bin/activate # For Linux/MacOS
env\Scripts\activate # For Windows
pip install -r requirements.txt


## Usage

### Training the Model:
To train the MPS model for classification:
python src/training/train_classification.py --config config/classification.yaml


### GAN-Style Training:
To perform adversarial training:
python src/training/train_gan.py --config config/gan.yaml


### Sampling New Data:
To generate new samples from a trained model:
python src/sampling/sample.py --model_path path/to/model.pt --num_samples 100


## Results

Results and analysis, including visualizations of generated samples and evaluation metrics, can be found in the `experiments/` directory. Refer to the thesis for detailed explanations and experimental setups.


## Contact

For questions or further discussion, please contact **Alex Mossi**.
