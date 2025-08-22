# Torch-PDE

Torch-PDE is a repository aimed at collecting implementations of partial differential equations (PDEs) rewritten in PyTorch.
The long-term goal is to build a library of efficient, GPU-accelerated solvers that can be used both for scientific simulations and for generating datasets in Scientific Machine Learning (SciML).

This first release includes an implementation of the Cahn–Hilliard equation, a classical PDE model widely studied in materials physics, especially for phase separation and pattern formation.

## Features

- General framework for implementing PDE solvers in PyTorch.

- Cahn–Hilliard solver using spectral methods (Fourier-based discretisation).

- GPU-accelerated computations for large-scale 2D/3D simulations.

- Ready-to-use for dataset generation and integration with ML pipelines.

  ## Getting Started

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/mariovozza/TorchPDE.git
cd TorchPDE
pip install -r requirements.txt
```
```bash
python run_cahn_hilliard.py
```

## Learn More

For a step-by-step explanation of the implementation, see the accompanying Medium article:

[How to Solve Partial Differential Equations with Spectral Methods in PyTorch](https://medium.com/@marvozzam/how-to-solve-partial-differential-equations-with-spectral-methods-in-pytorch-d009011e6567)
