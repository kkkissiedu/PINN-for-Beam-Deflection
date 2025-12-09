"""
train_inverse.py — inverse PINN: recover unknown uniform load q from sparse
deflection sensors on a simply-supported beam.

Two fixes vs the v0 notebooks (which got stuck at the initial guess):
  1. Hard-BC ansatz u = x²(L-x)²·NN(x) enforces both u=0 and u''=0 at the
     supports exactly, removing the BC tug-of-war from the loss.
  2. Nondimensionalise q: store it as q_norm ~ O(1) and reconstruct as
     q = Q_SCALE · q_norm. Without this, the gradient on the raw q parameter
     is dwarfed by the network's ~10k weights and never moves it.

Usage:
    conda activate cuda_pt
    python train_inverse.py
"""

# %% Imports
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model   import PINN
from physics import (
    make_sensors,
    pde_residual_loss,
    moment_bc_loss,
    data_loss,
)


# %% Config
E              = 210e9        # Young's modulus (Pa)
I              = 3.255e-6     # Second moment of area (m^4)
L              = 5.0          # Beam length (m)

Q_TRUE         = -15000.0     # Load to recover (N/m)
Q_INIT         = -5000.0      # Initial guess
Q_SCALE        = 1.0e4        # Nondimensionalisation factor — q = Q_SCALE * q_norm

N_SENSORS      = 7            # Interior deflection measurements
NOISE_STD      = 0.02         # Gaussian sensor noise (absolute, on deflection)

HIDDEN_SIZE    = 64
N_HIDDEN       = 3

N_COLLOCATION  = 800
N_EPOCHS       = 30000
LR             = 1e-3
LR_Q           = 1e-2         # Larger step for the scalar load parameter
LBFGS_MAX_ITER = 500          # LBFGS fine-tuning iterations after Adam
LOG_EVERY      = 2000

LAMBDA_DATA    = 1.0e6        # Sensor deflections are O(1e-3); upweight to compete with physics
LAMBDA_PHYS    = 1.0
LAMBDA_BC      = 1.0

SAVE_PATH      = "saved_models/inverse.pt"
SEED           = 42
MODEL_NAME     = "pinn-inverse"


# %% Training
def train(config: dict) -> dict:

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    EI       = config["E"] * config["I"]
    L_       = config["L"]
    q_scale  = config["q_scale"]

    # ── Model ─────────────────────────────────────────────────────────────────
    model = PINN(
        hidden_size = config["hidden_size"],
        n_hidden    = config["n_hidden"],
        hard_bc     = True,                                       # u and u'' = 0 at supports baked in
        L           = L_,
    ).to(device)

    # ── Learnable load (nondimensionalised) ───────────────────────────────────
    q_norm = nn.Parameter(torch.tensor([config["q_init"] / q_scale], device=device))

    # ── Sensors + boundary + collocation grid ─────────────────────────────────
    x_sensor, u_sensor = make_sensors(
        config["n_sensors"], config["q_true"], config["E"], config["I"], L_, device,
        noise_std = config["noise_std"],
    )
    x_bc = torch.tensor([[0.0], [L_]], device=device).requires_grad_(True)

    # ── Optimiser: separate LR for the load parameter ─────────────────────────
    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": config["lr"]},
        {"params": [q_norm],            "lr": config["lr_q"]},
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.7)

    run_dir = f"runs/{config['model_name']}_{datetime.now():%Y%m%d_%H%M%S}"
    writer  = SummaryWriter(run_dir)
    print(f"TensorBoard: {run_dir}")

    history = {"total": [], "data": [], "bc": [], "physics": [], "q": []}
    t0 = time.time()

    # ── Training loop ─────────────────────────────────────────────────────────
    pbar = tqdm(range(config["n_epochs"]), desc="train_inverse")
    for epoch in pbar:

        optimizer.zero_grad()

        # Resample collocation each step — better generalisation across the domain
        x_coll = (torch.rand(config["n_collocation"], 1, device=device) * L_)
        x_coll = x_coll.requires_grad_(True)

        q       = q_scale * q_norm                                # reconstruct physical q

        L_data  = data_loss(model, x_sensor, u_sensor)
        L_bc    = moment_bc_loss(model, x_bc)
        L_phys  = pde_residual_loss(model, x_coll, EI, q)

        total = (config["lambda_data"] * L_data
                 + config["lambda_bc"]   * L_bc
                 + config["lambda_phys"] * L_phys)
        total.backward()
        optimizer.step()
        scheduler.step()

        history["total"  ].append(total.item())
        history["data"   ].append(L_data.item())
        history["bc"     ].append(L_bc.item())
        history["physics"].append(L_phys.item())
        history["q"      ].append(q.item())

        # ── TensorBoard log ───────────────────────────────────────────────────
        writer.add_scalar("train_loss",   total.item(),               epoch)
        writer.add_scalar("loss_data",    L_data.item(),              epoch)
        writer.add_scalar("loss_bc",      L_bc.item(),                epoch)
        writer.add_scalar("loss_physics", L_phys.item(),              epoch)
        writer.add_scalar("q_recovered",  q.item(),                   epoch)
        writer.add_scalar("lr",           scheduler.get_last_lr()[0], epoch)

        if (epoch + 1) % config["log_every"] == 0:
            pbar.set_postfix(
                total=f"{total.item():.2e}",
                data =f"{L_data.item():.2e}",
                q    =f"{q.item():.1f}",
            )

    adam_time = time.time() - t0
    print(f"\nAdam time     : {adam_time/60:.1f} min")
    print(f"Adam q        : {(q_scale * q_norm).item():.2f} N/m")

    # ── LBFGS fine-tuning ─────────────────────────────────────────────────────
    t1 = time.time()
    x_coll_fixed = torch.linspace(0, L_, config["n_collocation"], device=device)
    x_coll_fixed = x_coll_fixed.view(-1, 1).requires_grad_(True)

    optimizer_lbfgs = torch.optim.LBFGS(
        list(model.parameters()) + [q_norm],
        max_iter         = config["lbfgs_max_iter"],
        tolerance_grad   = 1e-9,
        tolerance_change = 1e-9,
        line_search_fn   = "strong_wolfe",
    )

    def closure():
        optimizer_lbfgs.zero_grad()
        q       = q_scale * q_norm
        L_data  = data_loss(model, x_sensor, u_sensor)
        L_bc    = moment_bc_loss(model, x_bc)
        L_phys  = pde_residual_loss(model, x_coll_fixed, EI, q)
        total   = (config["lambda_data"] * L_data
                   + config["lambda_bc"]   * L_bc
                   + config["lambda_phys"] * L_phys)
        total.backward()
        history["total"  ].append(total.item())
        history["data"   ].append(L_data.item())
        history["bc"     ].append(L_bc.item())
        history["physics"].append(L_phys.item())
        history["q"      ].append(q.item())
        return total

    optimizer_lbfgs.step(closure)

    q_final = (q_scale * q_norm).item()
    err_pct = abs(q_final - config["q_true"]) / abs(config["q_true"]) * 100

    print(f"LBFGS time    : {(time.time() - t1)/60:.1f} min")
    print(f"True q        : {config['q_true']:.2f} N/m")
    print(f"Recovered q   : {q_final:.2f} N/m")
    print(f"Relative error: {err_pct:.2f} %")

    # ── Checkpoint ────────────────────────────────────────────────────────────
    torch.save({
        "model_state_dict": model.state_dict(),
        "q_norm"          : q_norm.detach().cpu(),
        "q_final"         : q_final,
        "history"         : history,
        "config"          : config,
    }, config["save_path"])
    print(f"Saved -> {config['save_path']}")

    writer.close()
    return {"model": model, "q_final": q_final, "history": history}


# %% Entry point
if __name__ == "__main__":

    config = dict(
        E              = E,
        I              = I,
        L              = L,
        q_true         = Q_TRUE,
        q_init         = Q_INIT,
        q_scale        = Q_SCALE,
        n_sensors      = N_SENSORS,
        noise_std      = NOISE_STD,
        hidden_size    = HIDDEN_SIZE,
        n_hidden       = N_HIDDEN,
        n_collocation  = N_COLLOCATION,
        n_epochs       = N_EPOCHS,
        lr             = LR,
        lr_q           = LR_Q,
        lbfgs_max_iter = LBFGS_MAX_ITER,
        lambda_data    = LAMBDA_DATA,
        lambda_phys    = LAMBDA_PHYS,
        lambda_bc      = LAMBDA_BC,
        log_every      = LOG_EVERY,
        save_path      = SAVE_PATH,
        seed           = SEED,
        model_name     = MODEL_NAME,
    )

    train(config)
