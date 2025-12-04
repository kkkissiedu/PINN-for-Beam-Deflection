"""
physics.py — Euler-Bernoulli beam physics for the PINN.

Governing PDE for a simply-supported beam under distributed load q(x):
    EI · u''''(x) = q(x),     u(0) = u(L) = 0,   u''(0) = u''(L) = 0

Pure-function module: takes a model + tensors, returns scalar loss terms.
No state, no globals.
"""

# %% Imports
import numpy as np
import torch


# %% Analytical reference
def analytical_solution(
    x: np.ndarray | torch.Tensor,
    q: float,
    E: float,
    I: float,
    L: float,
):
    """Closed-form deflection for a simply-supported beam, uniform load q."""
    return (q * x / (24 * E * I)) * (L ** 3 - 2 * L * x ** 2 + x ** 3)


# %% Synthetic sensor data
def make_sensors(
    n:         int,
    q_true:    float,
    E:         float,
    I:         float,
    L:         float,
    device:    torch.device,
    noise_std: float = 0.0,                                       # absolute Gaussian noise on u
):
    """Sample n interior deflection sensors from the analytical solution."""
    x = torch.linspace(0, L, n + 2)[1:-1].view(-1, 1).to(device)  # exclude supports
    u = analytical_solution(x.cpu().numpy(), q_true, E, I, L)
    sensor_values = torch.from_numpy(u).float().to(device)
    sensor_values += torch.randn_like(sensor_values) * noise_std
    return x, sensor_values                                       # (N_s, 1), (N_s, 1)


# %% Derivative helper
def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y, x,
        grad_outputs = torch.ones_like(y),
        create_graph = True,
    )[0]


def derivatives(model: torch.nn.Module, x: torch.Tensor):
    """Return (u, u', u'', u''', u'''') at x. x must have requires_grad=True."""
    u    = model(x)                                               # (N, 1)
    du   = _grad(u,   x)                                          # (N, 1)
    d2u  = _grad(du,  x)                                          # (N, 1)
    d3u  = _grad(d2u, x)                                          # (N, 1)
    d4u  = _grad(d3u, x)                                          # (N, 1)
    return u, du, d2u, d3u, d4u


# %% Loss terms
def pde_residual_loss(
    model: torch.nn.Module,
    x_coll: torch.Tensor,                                         # (N, 1) collocation
    EI:    float,
    q:     torch.Tensor | float,                                  # scalar or learnable tensor
) -> torch.Tensor:
    _, _, _, _, d4u = derivatives(model, x_coll)
    residual = EI * d4u - q                                       # (N, 1)
    return torch.mean(residual ** 2)


def deflection_bc_loss(
    model:  torch.nn.Module,
    x_bc:   torch.Tensor,                                         # (2, 1) endpoints
    u_bc:   torch.Tensor,                                         # (2, 1) target deflections
) -> torch.Tensor:
    return torch.mean((model(x_bc) - u_bc) ** 2)


def moment_bc_loss(
    model: torch.nn.Module,
    x_bc:  torch.Tensor,                                          # (2, 1) endpoints
) -> torch.Tensor:
    _, _, d2u, _, _ = derivatives(model, x_bc)
    return torch.mean(d2u ** 2)                                   # M = -EI·u'' → u''(ends) = 0


def data_loss(
    model:    torch.nn.Module,
    x_sensor: torch.Tensor,                                       # (N_s, 1)
    u_sensor: torch.Tensor,                                       # (N_s, 1)
) -> torch.Tensor:
    return torch.mean((model(x_sensor) - u_sensor) ** 2)
