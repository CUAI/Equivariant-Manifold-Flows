import torch
import numpy as np
from scipy.linalg import logm as logm_scipy

# 3x3 cdesa logm approach

# det(lambda*I - A) = lambda^3 + lambda^2 * c[3] + lambda * c[2] + c[1]
def charpoly3x3(A):
    elem1 = -(A[:,0,0] * (A[:,1,1] * A[:,2,2] - A[:,1,2] * A[:,2,1]) - A[:,1,0] * (A[:,0,1] * A[:,2,2] - A[:,0,2] * A[:,2,1]) + A[:,2,0] * (A[:,0,1] * A[:,1,2] - A[:,0,2] * A[:,1,1]))
    elem2 = A[:,0,0] * A[:,1,1] + A[:,0,0] * A[:,2,2] + A[:,1,1] * A[:,2,2] - A[:,1,0] * A[:,0,1] - A[:,2,0] * A[:,0,2] - A[:,2,1] * A[:,1,2]
    elem3 = -(A[:,0,0] + A[:,1,1] + A[:,2,2])

    return [elem1, elem2, elem3]

# largest-magnitude complex number
def cmax(x, y):
    return torch.where(torch.abs(x) > torch.abs(y), x, y)

# zeros of cubic polynomial
def cubic_zeros(p):
    a = 1
    b = p[2]
    c = p[1]
    d = p[0]
    D0 = b**2 - 3*a*c
    D1 = 2*b**3 - 9*a*b*c + 27*a**2*d
    L = torch.pow(1e-3 + D1**2 - 4 * D0**3, 0.5)
    V = cmax((D1 + L)/2, (D1 - L)/2)
    C = V**(1/3)
    w = np.exp(2*np.pi*1j/3)
    return [-(b+(w**k*C)+D0/(w**k*C))/(3*a) for k in range(3)]

# solve char poly for eigs
def su3_to_eigs_cdesa(x):
    p = charpoly3x3(x)
    zs = cubic_zeros(p)
    return torch.cat([x.unsqueeze(-1) for x in zs], dim=-1)

# log map cdesa, use poly interp
def log3x3_cdesa(x):
    eigs = su3_to_eigs_cdesa(x)
    q, _ = torch.solve(torch.log(eigs).unsqueeze(-1), 1e-6 * torch.eye(3).unsqueeze(0) + eigs.unsqueeze(-1)**(torch.tensor([0,1,2]).unsqueeze(0).unsqueeze(0)))
    q = q.unsqueeze(-1)
    return q[:,0] * torch.eye(x.shape[-1]).reshape(1,x.shape[-1],x.shape[-1]).repeat(x.shape[0], 1, 1) + q[:,1] * x + q[:,2] * x @ x
