# Kuramoto-FedAvg

[![ArXiv](https://img.shields.io/badge/arXiv-2505.19605v1-blue.svg)](https://arxiv.org/abs/2505.19605v1)  
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Federated optimization with phase-based synchronization dynamics inspired by the Kuramoto model.  
This repo contains the full implementation of **Kuramoto-FedAvg**, along with scripts for data preprocessing, training, evaluation, and visualization.

---

## 📖 Table of Contents

1. [Introduction](#introduction)  
2. [Installation & Prerequisites](#installation--prerequisites)  
3. [Data Preparation](#data-preparation)  
4. [Configuration](#configuration)  
5. [Usage](#usage)  
   - [Training](#training)  
   - [Evaluation](#evaluation)  
   - [Visualization](#visualization)  
6. [Experiments & Results](#experiments--results)    
7. [Citation](#citation)  
---

## 🌟 Introduction

Kuramoto-FedAvg reframes federated weight aggregation as a **synchronization problem**:  
each client’s update is treated as an oscillator phase, and the server adaptively re-weights updates based on their phase alignment with the global direction.  
This leads to faster convergence and better accuracy under non-IID client data.

Key features:
- **Phase-based weighting** of client updates  
- Theoretical convergence guarantees under heterogeneity  
- Configurable coupling strength, decay schedules, and optimizers  
- Support for MNIST, Fashion-MNIST, CIFAR-10, and custom datasets  

---

## 🚀 Installation & Prerequisites

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/Kuramoto-FedAvg.git
   cd Kuramoto-FedAvg

2. **Create a Python environment (recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt


4. **Configure Weights & Biases (optional)**
   ```bash
   wandb login

   You can disable WANDB by setting --wandb False in your command.

---

## 📦 Data Preparation

This repo uses torchvision datasets by default:

- MNIST / Fashion-MNIST: auto-downloaded
- CIFAR-10: auto-downloaded

To use custom data, update `parameters.py`:

```python
data_setups = {
    "dataset": "custom",               # e.g. "mnist", "fmnist", "cifar10", "custom"
    "data_root": "/path/to/your/data",
    "num_clients": 10,
    "shards_per_client": 5,
    # for custom: supply a PyTorch Dataset class in train_tools.preprocessing
}

## 📊 Experiments & Results

Pre-computed results and plots are in `experiments/figures/`. To reproduce:

### Run the MNIST experiment
```bash
bash experiments/run_mnist.sh

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@misc{muhebwa2025kuramoto,
  title        = {Kuramoto-FedAvg: Using Synchronization Dynamics to Improve Federated Learning Optimization under Statistical Heterogeneity},
  author       = {Muhebwa, Aggrey and Selialia, Khotso and Anwar, Fatima and Osman, Khalid K.},
  year         = {2025},
  eprint       = {2505.19605},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG}
}
