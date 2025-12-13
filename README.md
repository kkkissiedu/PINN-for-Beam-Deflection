# ⚛️ Physics-Informed Neural Network for Structural Mechanics

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> A deep learning model that learns the governing laws of physics to solve the Euler-Bernoulli beam equation without traditional training data.

---

## 📌 Overview

This repository contains the implementation of a **Physics-Informed Neural Network (PINN)**, a novel machine learning paradigm that integrates physical laws directly into the training process. This project demonstrates how a PINN can solve a classic structural mechanics problem—the deflection of a simply supported beam under a uniform load—by learning the underlying differential equation.

Instead of relying on a large dataset of pre-solved examples, the model learns by minimizing a loss function that penalizes violations of the governing physics and boundary conditions. This approach showcases a powerful method for solving complex engineering problems where data is scarce but the physical laws are well-understood.

The repository also tackles the **inverse problem**: recovering the unknown distributed load `q` from a handful of sparse, noisy deflection measurements along the beam. The same PINN architecture treats `q` as a learnable parameter and jointly fits the network and the load so that the predicted deflection matches the sensors while obeying the Euler-Bernoulli equation at the collocation points.

---

## ⚙️ How It Works

The PINN is a standard feed-forward neural network that is trained to act as a universal function approximator. The core innovation lies in its custom loss function, which is composed of two parts:

1.  **Physics Loss:** Calculated at random "collocation points" along the beam. At each point, PyTorch's **automatic differentiation** is used to compute the derivatives of the network's output. These derivatives are plugged into the Euler-Bernoulli equation, and the "residual" (how far the solution is from satisfying the equation) is minimized.
2.  **Boundary Loss:** Enforces the known physical constraints at the beam's ends, such as zero deflection and zero bending moment.

By minimizing the combined loss, the network is forced to discover a solution that is not only consistent with the boundary conditions but also obeys the laws of physics across the entire domain.

---

## 📈 Results

| Problem  | Max Abs Error | Relative L2 Error |
|----------|---------------|-------------------|
| Forward  | TBD           | TBD               |
| Inverse  | TBD           | TBD               |

> Placeholder values — run `python evaluate.py` to populate.

![comparison](beam_deflection_comparison.png)

---

## 🛠️ Getting Started

### Prerequisites
Ensure you have a Python environment with the following packages installed.

```bash
# Install required packages
pip install torch numpy matplotlib tqdm
