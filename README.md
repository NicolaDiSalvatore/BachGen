# BachGen

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.8%2B-red)](https://pytorch.org/)

---

## Overview

**BachGen** is a deep learning project focused on generating Bach-style chorales using Transformer and VAE architectures.  
The project is built with modular, production-oriented code aiming to incorporate MLOps practices such as reproducible training and scalable deployment.

---

## Current Status

The project is currently in active development. The core components implemented so far include:

- Data handling: raw and processed datasets with preprocessing scripts  
- Model base interface (`base.py`)  
- Transformer model implementation (`transformer.py`)  
- Variational Autoencoder model implementation (`vae.py`)  
- Training pipeline (`train.py`)  
- Dataset and tokenization utilities (`dataset.py`, `data_preprocessing.py`, `utils.py`)  
- Utility scripts for converting JSON to MIDI and MIDI to WAV  
- Unit tests covering all core scripts and utility modules, including audio conversion scripts

---

## Next Steps

- Finalize training and evaluation loops  
- Develop generation and inference scripts  
- Implement REST API for serving models  
- Integrate experiment tracking with MLflow  
- Set up CI/CD pipelines  

---

## Getting Started

### Installation

```bash
git clone https://github.com/NicolaDiSalvatore/BachGen.git
cd BachGen
pip install -r requirements.txt
