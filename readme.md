# ChemiconAI

An end-to-end vision-language framework for generating IUPAC nomenclature from hand-drawn molecule structure images.

---

## Overview

ChemiconAI tackles the problem of converting hand-drawn molecular structure sketches into their corresponding IUPAC names — a task that sits at the intersection of computer vision and chemical informatics. The pipeline is split into two sequential stages:

1. **Image to SMILES** — A visual encoder extracts structural features from a molecule sketch and predicts its SMILES representation.
2. **SMILES to IUPAC** — A fine-tuned sequence-to-sequence model translates the SMILES string into a valid IUPAC name.

---

## Pipeline Architecture

```
Hand-drawn Molecule Image
        |
        v
EfficientNet-B3 (Visual Encoder)
        |
        v
Transformer Encoder
        |
        v
    SMILES String
        |
        v
Fine-tuned BART Decoder (with custom BPE tokenization)
        |
        v
    IUPAC Name
```

---

## Results

| Stage | Metric | Score |
|---|---|---|
| Image to SMILES | Accuracy | 95.8% |
| SMILES to IUPAC | Token Accuracy | 97.48% |

---

## Dataset

- **100,000** RDKit-generated molecule sketches for image-to-SMILES training
- **150,000** SMILES-IUPAC pairs for sequence-to-sequence training
- Custom **Byte-Pair Encoding (BPE)** tokenization built from scratch to handle chemical syntax

---

## Repository Structure

```
ChemiconAI/
├── Images-to-SMILES/       # Visual encoder pipeline (EfficientNet-B3 + Transformer)
├── SMILES-to-IUPAC/        # Sequence generation pipeline (fine-tuned BART decoder)
└── README.md
```

---

## Tech Stack

- **Python**, **PyTorch**
- **EfficientNet-B3** — Visual feature extraction
- **Transformer Encoder** — Structural representation
- **BART** — Sequence-to-sequence IUPAC name generation
- **RDKit** — Molecule sketch generation and dataset curation
- **Custom BPE Tokenizer** — Chemical-syntax-aware tokenization

---

## Getting Started

### Prerequisites

```bash
pip install torch torchvision rdkit transformers efficientnet_pytorch
```

### Image to SMILES

```bash
cd Images-to-SMILES
# Follow notebook instructions for training and inference
```

### SMILES to IUPAC

```bash
cd SMILES-to-IUPAC
# Follow notebook instructions for training and inference
```

---

## Key Design Decisions

- **Modular pipeline** — Each stage is independently trainable and evaluable, making it easy to swap components or improve individual stages.
- **Custom BPE tokenization** — Off-the-shelf tokenizers do not handle IUPAC chemical syntax well. A custom tokenizer was built to preserve chemical nomenclature patterns.
- **RDKit-generated dataset** — Rather than relying on noisy real-world sketches, molecule images were programmatically generated using RDKit to ensure clean, labeled training data.

---

## Course Context

This project was developed as a research project for **CSCI 566 - Deep Learning** at the **University of Southern California (USC)**, Spring 2025.

---

## Authors

- **Aditya Ramachandran** — USC MS Computer Science
