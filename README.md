# ML from Scratch

## Introduction

This repository contains machine learning algorithms implemented from scratch using Python and NumPy. The implementations are developed while studying new concepts and may evolve step by step.

The goal of this project is to develop a deeper understanding of machine learning by implementing algorithms without relying on high-level machine learning libraries such as scikit-learn.
For me it is also about understanding the mathematical foundations of machine learning, but other learners can treat this part optionally as I will explain further in the next section below.

In addition to the ML implementations themselves, the repository contains demo experiments, visualizations, and notes created during the learning process.

## How to approach the Notebooks

In the notebooks section I split the demo experiments and the technical explanations for the ML-Algorithms in two categories. Before I tried combining everything in one notebook, but it gets to
heavy in my opinion. Like that the student can choose if he or she
just wants to build intuition with the demo or on the other hand build technical depth with the explanations of the algorithms and the mathematical concepts involved.

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