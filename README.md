# ML from Scratch

## Introduction

This repository contains machine learning algorithms and selected mathematical foundations implemented from scratch using Python and NumPy. The implementations are developed while studying new concepts and may evolve step by step.

The goal is to build a deeper understanding of machine learning without relying on libraries that provide ready-made algorithm implementations.

The repository also includes explanatory notebooks, exercises, and unit tests.

## Learning Resources

The following resources are used throughout the learning process. This list may be updated as the project evolves.

- Machine Learning Specialization — DeepLearning.AI, Andrew Ng
- Mathematical Foundations of Reinforcement Learning — Shiyu Zhao

## Project Structure

```text
ml-from-scratch/
├── exercises/
│   └── stochastic_approximation/
├── notebooks/
│   └── supervised_learning/
├── src/
│   └── ml_from_scratch/
│       ├── reinforcement_learning/
│       ├── stochastic_approximation/
│       └── supervised_learning/
└── tests/
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