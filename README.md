# ML from Scratch

## Introduction

This repository contains machine learning algorithms implemented from scratch using Python and NumPy. The implementations are developed while studying new concepts and may evolve step by step.

The goal of this project is to develop a deeper understanding of machine learning by implementing algorithms without relying on high-level machine learning libraries such as scikit-learn.

The project also explores the mathematical foundations behind the algorithms. These sections provide additional technical depth, but readers may focus on the intuition and practical examples without working through every mathematical detail.

In addition to the implementations, the repository contains notebooks with explanations, experiments, and visualizations created during the learning process.

## How to Approach the Notebooks

Each topic is covered in a dedicated notebook that combines intuitive explanations, mathematical background, code examples, and visualizations.

## Learning Resources

Here are the resources I used to learn ML. This will be updated if I feel that I start using other resources more frequently.

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