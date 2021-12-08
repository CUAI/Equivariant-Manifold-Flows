import torch
import torch.nn as nn
import torch.autograd.functional as AF
import numpy as np

from flows.sun import SUN
from flows.potential import ComplexDeepTimeSet
from flows.logm import su3_to_eigs_cdesa


# maps SU(3) -> R in a conjugation invariant way
class TimePotentialSU3(nn.Module):
    def __init__(self):
        super(TimePotentialSU3, self).__init__()
        self.full_eigdecomp = su3_to_eigs_cdesa
        self.deepset = ComplexDeepTimeSet(1, 1, hidden_channels=64)

    def forward(self, t, x):
        x = self.full_eigdecomp(x)
        x = x.unsqueeze(-1)
        x = self.deepset(t, x)
        return x


class SU3TimeEquivariantVectorField(nn.Module):
    def __init__(self, func):
        super(SU3TimeEquivariantVectorField, self).__init__()
        self.func = func

    def forward(self, t, x):
        vfield = torch.autograd.grad(self.func(t, x).squeeze().sum(), x, create_graph=True, retain_graph=True)[0]
        return vfield


class AmbientProjNN(nn.Module):
    def __init__(self, func):
        super(AmbientProjNN, self).__init__()
        self.func = func
        self.man = SUN()

    def forward(self, t, x):
        x = self.man.proju(x, self.func(t, x))
        return x


class SU3_CNF(nn.Module):
    def __init__(self):
        super(SU3_CNF, self).__init__()

        self.solver = 'rk4'
        self.atol = 1e-4
        self.rtol = 1e-4
        self.solver_options = {'step_size': 1}
        self.man = SUN()

        self.potentialfn = TimePotentialSU3()
        self.func = AmbientProjNN(SU3TimeEquivariantVectorField(self.potentialfn))

    def forward(self, z, charts=1):
        return self.bf(z, reverse=False)

    def inverse(self, z, charts=1):
        return self.bf(z, reverse=True)

    def mani_div(self, dx, y):
        sum_diag = 0.
        for i in range(y.shape[1]):
            for j in range(y.shape[2]):
                sum_diag += self.man.proju(y, torch.autograd.grad(dx[:, i, j].sum(), y, create_graph=True)[0].contiguous())[:, i, j].contiguous()
        return sum_diag.contiguous()

    def bf(self, z, steps=3, reverse=False):
        y = z
        log_p = 0
        for t in range(steps):
            with torch.set_grad_enabled(True):
                y.requires_grad_(True)
                t = torch.tensor(t).to(y)
                t.requires_grad_(True)

                if reverse:
                    dy = -self.func(t / steps, y)
                    div = -self.mani_div(dy, y)
                else:
                    dy = self.func(t / steps, y)
                    div = self.mani_div(dy, y)

            y = self.man.exp(y, dy / steps)
            log_p += div / steps

        log_p = torch.real(log_p) + torch.imag(log_p)
        return y, log_p

    def get_regularization_states(self):
        return None

    def num_evals(self):
        return self.odefunc._num_evals.item()
