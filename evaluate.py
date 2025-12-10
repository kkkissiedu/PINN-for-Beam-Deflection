"""
evaluate.py — load a trained PINN checkpoint and report error vs the analytical
Euler-Bernoulli solution.

  python evaluate.py --problem forward
  python evaluate.py --problem inverse
"""

# %% Imports
import argparse

import numpy as np
import torch

from model   import PINN
from physics import analytical_solution, derivatives


# %% Config
FORWARD_CKPT = "saved_models/forward.pt"
INVERSE_CKPT = "saved_models/inverse.pt"
N_EVAL       = 200


# %% Forward evaluation
def evaluate_forward(config: dict) -> dict:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(config["ckpt_path"], map_location=device, weights_only=False)
    cfg    = ckpt["config"]
    L_     = cfg["L"]

    model = PINN(
        hidden_size = cfg["hidden_size"],
        n_hidden    = cfg["n_hidden"],
        hard_bc     = False,
        L           = L_,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x = torch.linspace(0, L_, config["n_eval"], device=device).view(-1, 1)
    x.requires_grad_(True)

    u_pred  = model(x).detach().cpu().numpy().flatten()
    u_true  = analytical_solution(
        x.detach().cpu().numpy(), cfg["q0"], cfg["E"], cfg["I"], L_,
    ).flatten()

    mse        = float(np.mean((u_pred - u_true) ** 2))
    max_err    = float(np.max(np.abs(u_pred - u_true)))
    rel_err    = max_err / float(np.max(np.abs(u_true)))

    # BC residuals (deflection + moment at supports)
    x_bc = torch.tensor([[0.0], [L_]], device=device, requires_grad=True)
    u_bc, _, d2u_bc, _, _ = derivatives(model, x_bc)
    moment_bc = (-cfg["E"] * cfg["I"] * d2u_bc).detach().cpu().numpy().flatten()

    print("─── Forward problem ───")
    print(f"MSE                 : {mse:.3e}")
    print(f"Max absolute error  : {max_err:.3e} m")
    print(f"Relative error      : {rel_err*100:.3f} %")
    print(f"u(0) = {u_bc[0].item(): .3e} m   u(L) = {u_bc[1].item(): .3e} m")
    print(f"M(0) = {moment_bc[0]: .3e} N·m   M(L) = {moment_bc[1]: .3e} N·m")

    return {
        "x": x.detach().cpu().numpy().flatten(),
        "u_pred": u_pred, "u_true": u_true,
        "mse": mse, "max_err": max_err,
        "history": ckpt["history"],
    }


# %% Inverse evaluation
def evaluate_inverse(config: dict) -> dict:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(config["ckpt_path"], map_location=device, weights_only=False)
    cfg    = ckpt["config"]
    L_     = cfg["L"]

    model = PINN(
        hidden_size = cfg["hidden_size"],
        n_hidden    = cfg["n_hidden"],
        hard_bc     = True,
        L           = L_,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x = torch.linspace(0, L_, config["n_eval"], device=device).view(-1, 1)
    with torch.no_grad():
        u_pred = model(x).cpu().numpy().flatten()

    u_true = analytical_solution(
        x.cpu().numpy(), cfg["q_true"], cfg["E"], cfg["I"], L_,
    ).flatten()

    q_final = ckpt["q_final"]
    q_err   = abs(q_final - cfg["q_true"])
    q_pct   = q_err / abs(cfg["q_true"]) * 100
    mse     = float(np.mean((u_pred - u_true) ** 2))
    max_err = float(np.max(np.abs(u_pred - u_true)))

    print("─── Inverse problem ───")
    print(f"True q              : {cfg['q_true']:.2f} N/m")
    print(f"Recovered q         : {q_final:.2f} N/m")
    print(f"Parameter error     : {q_err:.2f} N/m  ({q_pct:.2f} %)")
    print(f"Deflection MSE      : {mse:.3e}")
    print(f"Max deflection error: {max_err:.3e} m")

    return {
        "x": x.cpu().numpy().flatten(),
        "u_pred": u_pred, "u_true": u_true,
        "q_final": q_final, "q_true": cfg["q_true"],
        "history": ckpt["history"],
    }


# %% Entry point
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=["forward", "inverse"], default="forward")
    args = parser.parse_args()

    config = dict(
        ckpt_path = FORWARD_CKPT if args.problem == "forward" else INVERSE_CKPT,
        n_eval    = N_EVAL,
    )

    if args.problem == "forward":
        evaluate_forward(config)
    else:
        evaluate_inverse(config)
