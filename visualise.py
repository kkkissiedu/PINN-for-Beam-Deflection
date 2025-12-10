"""
visualise.py — generate plots for the PINN beam-deflection results.

Reads checkpoints from saved_models/, writes PNGs to outputs/.

  python visualise.py --problem forward
  python visualise.py --problem inverse
"""

# %% Imports
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from evaluate import evaluate_forward, evaluate_inverse


# %% Config
FORWARD_CKPT = "saved_models/forward.pt"
INVERSE_CKPT = "saved_models/inverse.pt"
OUTPUT_DIR   = "outputs"
N_EVAL       = 200


# %% Plot helpers
def plot_deflection(x, u_true, u_pred, save_path: str, x_sensor=None, u_sensor=None):
    plt.figure(figsize=(10, 6))
    plt.plot(x, u_true, label="Analytical",   color="C0", lw=2)
    plt.plot(x, u_pred, label="PINN",         color="C3", lw=2, ls="--")
    if x_sensor is not None:
        plt.scatter(x_sensor, u_sensor, color="k", s=50, zorder=5, label="Sensors")
    plt.xlabel("Position along beam (m)")
    plt.ylabel("Deflection (m)")
    plt.title("Beam deflection — PINN vs analytical")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_error(x, u_true, u_pred, save_path: str):
    err = np.abs(u_pred - u_true)
    plt.figure(figsize=(10, 4))
    plt.semilogy(x, err, color="C3", lw=2)
    plt.xlabel("Position along beam (m)")
    plt.ylabel("|u_pred − u_true|  (m)")
    plt.title("Pointwise absolute error")
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_loss_history(history: dict, save_path: str):
    plt.figure(figsize=(10, 6))
    for key, vals in history.items():
        if key == "q":
            continue
        plt.semilogy(vals, label=key, lw=1.5, alpha=0.85)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("Training loss components")
    plt.grid(alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_q_convergence(q_history, q_true: float, save_path: str):
    plt.figure(figsize=(10, 6))
    plt.plot(q_history, color="C0", lw=2, label="Learned q")
    plt.axhline(q_true, color="C3", lw=2, ls="--", label=f"True q = {q_true:.0f}")
    plt.xlabel("Epoch")
    plt.ylabel("q  (N/m)")
    plt.title("Inverse problem — load parameter convergence")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# %% Visualise forward
def visualise_forward(config: dict) -> None:
    os.makedirs(config["output_dir"], exist_ok=True)
    res = evaluate_forward({"ckpt_path": config["ckpt_path"], "n_eval": config["n_eval"]})

    plot_deflection(
        res["x"], res["u_true"], res["u_pred"],
        os.path.join(config["output_dir"], "forward_deflection.png"),
    )
    plot_error(
        res["x"], res["u_true"], res["u_pred"],
        os.path.join(config["output_dir"], "forward_error.png"),
    )
    plot_loss_history(
        res["history"],
        os.path.join(config["output_dir"], "forward_loss.png"),
    )
    print(f"Saved forward plots -> {config['output_dir']}/")


# %% Visualise inverse
def visualise_inverse(config: dict) -> None:
    os.makedirs(config["output_dir"], exist_ok=True)
    res = evaluate_inverse({"ckpt_path": config["ckpt_path"], "n_eval": config["n_eval"]})

    plot_deflection(
        res["x"], res["u_true"], res["u_pred"],
        os.path.join(config["output_dir"], "inverse_deflection.png"),
    )
    plot_error(
        res["x"], res["u_true"], res["u_pred"],
        os.path.join(config["output_dir"], "inverse_error.png"),
    )
    plot_loss_history(
        res["history"],
        os.path.join(config["output_dir"], "inverse_loss.png"),
    )
    plot_q_convergence(
        res["history"]["q"], res["q_true"],
        os.path.join(config["output_dir"], "inverse_q_convergence.png"),
    )
    print(f"Saved inverse plots -> {config['output_dir']}/")


# %% Entry point
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=["forward", "inverse"], default="forward")
    args = parser.parse_args()

    config = dict(
        ckpt_path  = FORWARD_CKPT if args.problem == "forward" else INVERSE_CKPT,
        output_dir = OUTPUT_DIR,
        n_eval     = N_EVAL,
    )

    if args.problem == "forward":
        visualise_forward(config)
    else:
        visualise_inverse(config)
