import numpy as np
import torch
import flows

from torch import matrix_exp as expm
from flows.logm import log3x3_cdesa


class SUN:
    def __init__(self):
        super(SUN, self).__init__()

    def exp(self, x, u):
        return x @ expm(x.conj().transpose(-2,-1) @ u)

    def log(self, x, y):
        _, n, _ = x.shape
        assert n == 3, "Operation supported only for SU(3)"

        return x @ log3x3_cdesa(x.conj().transpose(-2,-1) @ y)

    def proju(self, x, u, inplace=False):
        # arbitrary matrix C projects to skew-hermitian B := (C - C^H)/2
        # then make traceless with tr(B - trB/N * I) = trB - trB = 0

        _, n, _ = x.shape
        algebra_elem = torch.solve(u, x)[0] # X^{-1} u

        # do projection in lie algebra
        B = (algebra_elem - algebra_elem.conj().transpose(-2,-1)) / 2
        trace = torch.einsum('bii->b', B)
        B = B - 1/n * trace.unsqueeze(-1).unsqueeze(-1) * torch.eye(n).repeat(x.shape[0], 1, 1)

        # check zero trace
        assert torch.abs(torch.mean(torch.einsum('bii->b', B))) <= 1e-6

        
        return B
