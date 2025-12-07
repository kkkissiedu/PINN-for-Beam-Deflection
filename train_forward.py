"""
train_forward.py — forward PINN for a simply-supported beam under uniform load.

Solves EI·u'''' = q with soft boundary conditions enforced through the loss.
Saves the trained model + loss history to saved_models/forward.pt.

Usage:
    conda activate cuda_pt
    python train_forward.py
"""

# %% Imports
import time
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model   import PINN
from physics import (
    pde_residual_loss,
    deflection_bc_loss,
    moment_bc_loss,
)


# %% Config
E              = 210e9        # Young's modulus (Pa) — steel
I              = 3.255e-6     # Second moment of area (m^4)
L              = 5.0          # Beam length (m)
Q0             = 1000.0       # Uniform distributed load (N/m)

HIDDEN_SIZE    = 50
N_HIDDEN       = 3

N_COLLOCATION  = 2000
N_EPOCHS       = 15000
LR             = 1e-3
LBFGS_MAX_ITER = 500          # LBFGS fine-tuning iterations after Adam
LOG_EVERY      = 1000

SAVE_PATH      = "saved_models/forward.pt"
SEED           = 42
MODEL_NAME     = "pinn-forward"


# %% Training
def train(config: dict) -> dict:

    torch.manual_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    EI = config["E"] * config["I"]
    L_ = config["L"]

    # ── Model ─────────────────────────────────────────────────────────────────
    model = PINN(
        hidden_size = config["hidden_size"],
        n_hidden    = config["n_hidden"],
        hard_bc     = False,
        L           = L_,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # ── Collocation + boundary points ─────────────────────────────────────────
    x_coll = torch.linspace(0, L_, config["n_collocation"], device=device)
    x_coll = x_coll.view(-1, 1).requires_grad_(True)              # (N, 1)

    x_bc = torch.tensor([[0.0], [L_]], device=device).requires_grad_(True)
    u_bc = torch.zeros_like(x_bc)                                 # zero deflection at supports

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    run_dir = f"runs/{config['model_name']}_{datetime.now():%Y%m%d_%H%M%S}"
    writer  = SummaryWriter(run_dir)
    print(f"TensorBoard: {run_dir}")

    history = {"total": [], "pde": [], "bc": [], "moment": []}
    t0 = time.time()

    # ── Training loop ─────────────────────────────────────────────────────────
    pbar = tqdm(range(config["n_epochs"]), desc="train_forward")
    for epoch in pbar:

        optimizer.zero_grad()

        loss_pde    = pde_residual_loss(model, x_coll, EI, config["q0"])
        loss_bc     = deflection_bc_loss(model, x_bc, u_bc)
        loss_moment = moment_bc_loss(model, x_bc)

        loss = loss_pde + loss_bc + loss_moment
        loss.backward()
        optimizer.step()

        history["total" ].append(loss.item())
        history["pde"   ].append(loss_pde.item())
        history["bc"    ].append(loss_bc.item())
        history["moment"].append(loss_moment.item())

        # ── TensorBoard log ───────────────────────────────────────────────────
        writer.add_scalar("train_loss",  loss.item(),        epoch)
        writer.add_scalar("loss_pde",    loss_pde.item(),    epoch)
        writer.add_scalar("loss_bc",     loss_bc.item(),     epoch)
        writer.add_scalar("loss_moment", loss_moment.item(), epoch)
        writer.add_scalar("lr",          config["lr"],       epoch)

        if (epoch + 1) % config["log_every"] == 0:
            pbar.set_postfix(
                loss=f"{loss.item():.2e}",
                pde =f"{loss_pde.item():.2e}",
                bc  =f"{loss_bc.item():.2e}",
            )

    print(f"\nAdam time    : {(time.time() - t0)/60:.1f} min")
    print(f"Adam loss    : {history['total'][-1]:.3e}")

    # ── LBFGS fine-tuning ─────────────────────────────────────────────────────
    t1 = time.time()
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        max_iter         = config["lbfgs_max_iter"],
        tolerance_grad   = 1e-9,
        tolerance_change = 1e-9,
        line_search_fn   = "strong_wolfe",
    )

    def closure():
        optimizer_lbfgs.zero_grad()
        loss_pde    = pde_residual_loss(model, x_coll, EI, config["q0"])
        loss_bc     = deflection_bc_loss(model, x_bc, u_bc)
        loss_moment = moment_bc_loss(model, x_bc)
        loss = loss_pde + loss_bc + loss_moment
        loss.backward()
        history["total" ].append(loss.item())
        history["pde"   ].append(loss_pde.item())
        history["bc"    ].append(loss_bc.item())
        history["moment"].append(loss_moment.item())
        return loss

    optimizer_lbfgs.step(closure)

    print(f"LBFGS time   : {(time.time() - t1)/60:.1f} min")
    print(f"Final loss   : {history['total'][-1]:.3e}")

    # ── Checkpoint ────────────────────────────────────────────────────────────
    torch.save({
        "model_state_dict": model.state_dict(),
        "history"         : history,
        "config"          : config,
    }, config["save_path"])
    print(f"Saved -> {config['save_path']}")

    writer.close()
    return {"model": model, "history": history}


# %% Entry point
if __name__ == "__main__":

    config = dict(
        E              = E,
        I              = I,
        L              = L,
        q0             = Q0,
        hidden_size    = HIDDEN_SIZE,
        n_hidden       = N_HIDDEN,
        n_collocation  = N_COLLOCATION,
        n_epochs       = N_EPOCHS,
        lr             = LR,
        lbfgs_max_iter = LBFGS_MAX_ITER,
        log_every      = LOG_EVERY,
        save_path      = SAVE_PATH,
        seed           = SEED,
        model_name     = MODEL_NAME,
    )

    train(config)
