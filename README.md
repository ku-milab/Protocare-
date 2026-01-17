# ProtoCare+: Knowledge Graph Guided Representation Learning for Diagnosis Prediction

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-2.3.0-orange.svg)](https://pytorch.org/)

This repository contains the official PyTorch implementation of the paper: **"ProtoCare+: Knowledge Graph Guided Representation Learning for Diagnosis Prediction"**, Under review, 2026.

## 📝 Abstract

Deep learning models utilizing Electronic Health Records (EHR) and Medical Knowledge Graphs (KGs) often face challenges regarding **KG incompleteness** and **noise** (task-irrelevant information). 

**ProtoCare+** is a novel framework designed to address these limitations by integrating:
1.  **Mixed Attention Module (MAM):** Captures both explicit KG relations and implicit data-driven dependencies.
2.  **Prototype Learning:** Extracts shared latent attributes across patient groups to improve robustness, especially for low-frequency diseases.
3.  **Multi-view Graph Representation Learning (GRL):** Filters noise through graph contrastive learning from both local (patient) and global (prototype) perspectives.

Extensive experiments on the **MIMIC-III** dataset demonstrate that ProtoCare+ consistently outperforms state-of-the-art baselines.

## 🏗️ Model Architecture

The overall framework of ProtoCare+ consists of four main components:

![Framework Overview](path/to/your/figure2.png)
*Figure 1: The overall framework of ProtoCare+ (Source: Figure 2 in the paper).*

1.  **Patient Feature Extraction:** Encodes visit sequences using GRUs and clinical embeddings (Diagnosis, Procedure, Drug) enhanced by the **Mixed Attention Module (MAM)**.
2.  **Prototype Learning:** Softly assigns patient embeddings to representative prototypes under diversity and clustering constraints.
3.  **Multi-view GRL:** Extracts task-relevant information via global (prototype) and local (patient) graph masking modules.
4.  **Diagnosis Prediction:** Fuses multi-view representations to predict future diagnoses.

### Mixed Attention Module (MAM)
![MAM Architecture](path/to/your/figure3.png)
*Figure 2: Detailed architecture of the MAM (Source: Figure 3 in the paper).*
MAM integrates structural signals from the medical KG (via Graph Attention Networks) and contextual co-occurrence patterns (via Transformer-based self-attention).

## 🛠️ Requirements

The code was developed and tested in the following environment:

* **Python** >= 3.8
* **PyTorch** == 2.3.0
* **CUDA** (Tested on NVIDIA RTX A6000)
* **Libraries:** `numpy`, `pandas`, `scikit-learn`, `dgl` (or `torch_geometric` depending on implementation)

To install dependencies:
```bash
pip install -r requirements.txt
