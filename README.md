# ML from Scratch

## Introduction

This repository contains machine learning algorithms implemented from scratch using Python and NumPy. The implementations are developed while studying new concepts and may evolve step by step.

The goal of this project is to develop a deeper understanding of machine learning by implementing algorithms without relying on high-level machine learning libraries such as scikit-learn. In addition to the implementations themselves, the repository contains demo experiments, visualizations, and notes created during the learning process.

## Learning Resources

- Machine Learning Specialization (DeepLearning.AI, Andrew Ng)

## Topics

- Linear Regression
- Gradient Descent

## Getting Started

### Clone the repository

```bash
git clone https://github.com/MonasteryStudent/ml-from-scratch.git
cd ml-from-scratch
```

### Create a virtual environment

```bash
python3 -m venv .venv
```

### Activate the virtual environment

Linux / macOS:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

### Install the project

Install the project together with development dependencies:

```bash
python -m pip install -e ".[dev]"
```

### Run the tests

```bash
pytest
```

### Start Jupyter

```bash
jupyter lab
```

or

```bash
jupyter notebook
```