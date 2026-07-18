# ML from Scratch

## Introduction

This repository contains machine learning algorithms implemented from scratch using Python and NumPy. The implementations are developed while studying new concepts and may evolve step by step.

The goal is to build a deeper understanding of machine learning without relying on high-level machine learning libraries such as scikit-learn.

The repository also includes notebooks with explanations, mathematical background, experiments, and visualizations created during the learning process.

## How to Approach the Notebooks

Each topic is covered in a dedicated notebook that combines intuition, mathematical foundations, code examples, and visualizations.

## Learning Resources

The following resources are used throughout the learning process. This list may be updated as the project evolves.

- Machine Learning Specialization — DeepLearning.AI, Andrew Ng

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

### VS Code Notebook Setup

To work with the notebooks in VS Code, install the following extensions:

- Python
- Jupyter

Then select the project's virtual environment (`.venv`) as the notebook kernel. 
After that, you can run any notebook in the `notebooks/` directory.

### Run the tests

```bash
pytest
```