"""
model.py — PINN MLP for the Euler-Bernoulli beam.

Two output modes:
  - soft_bc=False : plain MLP. BCs enforced via loss terms (forward problem).
  - hard_bc=True  : output is multiplied by x²(L-x)², so u(0)=u(L)=0 and
                    u''(0)=u''(L)=0 both hold by construction. Removes the
                    BC tug-of-war and frees capacity for the physics term
                    (inverse problem).
"""

# %% Imports
import torch
import torch.nn as nn


# %% PINN
class PINN(nn.Module):

    def __init__(
        self,
        hidden_size: int = 50,
        n_hidden:    int = 3,
        hard_bc:     bool = False,
        L:           float = 1.0,
    ):
        super().__init__()
        self.hard_bc = hard_bc
        self.L       = L

        layers = [nn.Linear(1, hidden_size), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers += [nn.Linear(hidden_size, 1)]
        self.net = nn.Sequential(*layers)

        self.apply(_xavier_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:           # x: (N, 1)
        y = self.net(x)                                           # (N, 1)
        if self.hard_bc:
            y = (x ** 2) * ((self.L - x) ** 2) * y                # u and u'' both zero at x=0, x=L
        return y                                                  # (N, 1)


# %% Weight init
def _xavier_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
