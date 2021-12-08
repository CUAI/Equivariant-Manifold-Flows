import torch
import torch.distributions
import scipy
import scipy.linalg
import numpy as np

from flows.logm import su3_to_eigs_cdesa

class HaarSUN(torch.distributions.Distribution):

    support = torch.distributions.constraints.real
    has_rsample = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def rsample(self, n, dim):
        # produces n uniform samples over SU(dim)
        final = torch.zeros(n, dim, dim).to(torch.complex64)
        for k in range(n):
            z = (scipy.randn(dim, dim) + 1j*scipy.randn(dim, dim))/scipy.sqrt(2.0)
            q,r = scipy.linalg.qr(z)
            d = scipy.diagonal(r)
            ph = d/np.abs(d)
            q = scipy.multiply(q, ph, q)
            q = q / scipy.linalg.det(q)**(1/dim)

            final[k] = torch.tensor(q)
        return final

    def log_prob(self, z):
        # log(z) = log p(v) - log det [(\partial / \partial v) proj_{\mu}(v)]
        _, n, _ = z.shape

        assert n == 3, "Operation supported only for SU(3)"
        v = su3_to_eigs_cdesa(z)

        # recall that eigdecomp returns [real1, real2, ..., imag1, imag2, ...]
        # use haar formula from boyda,\prod_{i < j} |\lambda_i -\lambda_j|^2
        log_prob = 0
        for j in range(n):
            for i in range(j):
                log_prob += torch.log(torch.abs(v[:,i] - v[:,j])**2)

        return log_prob

    def rsample_log_prob(self, shape=torch.Size()):
        z = self.rsample(shape)
        return z, self.log_prob(z)
