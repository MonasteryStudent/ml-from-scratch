# ML from Scratch

## Introduction

This repository contains machine learning algorithms implemented from scratch using Python and NumPy. The implementations are developed while studying new concepts and may evolve step by step.

The goal is to build a deeper understanding of machine learning without relying on high-level machine learning libraries such as scikit-learn.

The repository also includes notebooks with explanations, mathematical background, experiments, and visualizations created during the learning process.

## Learning Resources

The following resources are used throughout the learning process. This list may be updated as the project evolves.

- Machine Learning Specialization — DeepLearning.AI, Andrew Ng

## Topics

- Linear Regression
- Gradient Descent
- Multiple Linear Regression
- Feature Scaling
- Polynomial Regression
- Logistic Regression
- Neural Networks

## Project Structure

```text
ml-from-scratch/
├── notebooks/
│   ├── 01_linear_regression/
│   ├── 02_multiple_linear_regression/
│   └── 03_logistic_regression/
├── src/
│   └── ml_from_scratch/
│       ├── activation_functions.py
│       ├── logistic_regression.py
│       ├── multiple_linear_regression.py
│       ├── neural_network.py
│       └── univariate_linear_regression.py
├── tests/
├── pyproject.toml
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/MonasteryStudent/ml-from-scratch.git
cd ml-from-scratch
```

### 2. Set up a virtual environment

Create the virtual environment:

```bash
python3 -m venv .venv
```

Activate it (Linux / macOS):

```bash
source .venv/bin/activate
```

### 3. Install the project

Install the project together with its development dependencies:

```bash
python -m pip install -e ".[dev]"
```

### 4. Run the tests

```bash
pytest
```