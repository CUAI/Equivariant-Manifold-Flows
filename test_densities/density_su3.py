import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

from distributions.haarsun import HaarSUN
from flows.logm import su3_to_eigs_cdesa

## Data fetching/sample densities

def eval_density_log_su3(dataset, vals, c, beta):
    # general form of density from boyda : e^{\beta/3 \cdot Re tr [c_1 U^1 + c_2 U^2 + c_3 U^3]}

    # vals are on SU(3)
    v = su3_to_eigs_cdesa(vals)
    v = torch.cat((torch.real(v), torch.imag(v)), dim=1)

    # recall that eigdecomp returns [real1, real2, ..., imag1, imag2, ...]
    thetas = torch.acos(v[:,0:3].clamp(-1+1e-6,1-1e-6))

    # note vals are thetas
    if dataset == 'boyda':
        u_1 = torch.sum(torch.cos(thetas), dim=1)
        u_2 = torch.sum(torch.cos(2*thetas), dim=1)
        u_3 = torch.sum(torch.cos(3*thetas), dim=1)

        return beta / 3 * (c[0] * u_1 + c[1] * u_2 + c[2] * u_3)
    elif dataset == 'haar_su3':
        return HaarSUN().log_prob(vals)


def eval_density_log_su3_boyda(dataset, vals):
    if dataset == 'boyda1':
        return eval_density_log_su3('boyda', vals, c=[0.98,-0.63,-0.21], beta=9)
    elif dataset == 'boyda2':
        return eval_density_log_su3('boyda', vals, c=[0.17,-0.65,1.22], beta=9)
    elif dataset == 'boyda3':
        return eval_density_log_su3('boyda', vals, c=[1.00,0.00,0.00], beta=9)

### SU3 plotting methods

def plot_distr_su3(distr=None, res_npts=500, save_fig=True, model=None, device=None, namestr='su3_model'):
    on_mani, thetas, log_detjac = make_grid_eigs_su3(res_npts)

    if distr == 'boyda1' or distr == 'boyda2' or distr == 'boyda3':
        log_probs_baseline = eval_density_log_su3_boyda(distr, on_mani)
        log_probs_baseline += log_detjac
        probs = torch.exp(log_probs_baseline)
    if distr == 'model':
        probs = model_probs_su3(model, on_mani, log_detjac, device)

    plot_su3_density(thetas, probs, res_npts)

    if save_fig:
        print(f'Saved to: {namestr}.png')
        plt.savefig(f'{namestr}.png')


def plot_su3_density(thetas, probs, npts):
    # thetas are in [-pi,pi]
    theta1, theta2 = thetas

    # reshape back to grid form
    theta1 = theta1.cpu().numpy().reshape(npts, npts)
    theta2 = theta2.cpu().numpy().reshape(npts, npts)
    probs = probs.cpu().numpy().reshape(npts, npts)

    print(f'probs: {probs}')

    # some instability, may need to set infs and nans to 0
    probs[probs == np.inf] = 0
    probs[np.isnan(probs)] = 0

    # now clip and visualize
    probs = probs / probs.max()
    probs = np.clip(probs, a_min=1e-5, a_max=np.inf)
    print(f'max probs: {probs.max()}, min probs: {probs.min()}')
    log_probs = np.log(probs)

    fig = plt.figure(figsize=(3,3), dpi=200)
    ax = fig.add_subplot(111)

    pcm = ax.pcolormesh(theta1, theta2, probs, cmap='viridis', shading='auto', norm=colors.LogNorm(vmin=probs.min(), vmax=probs.max()))
    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([-np.pi, np.pi])

    plt.grid(False)
    ax.axis('off')


def model_probs_su3(model, on_mani, log_detjac, device):
    if device:
        on_mani = on_mani.to(device)
        log_detjac = log_detjac.to(device)

    z, logprob = model.inverse(on_mani)

    val = HaarSUN().log_prob(z) # do z, since uniformly sampling target density
    val += logprob.detach()
    probs = torch.exp(val)
    return probs.detach()


def make_grid_eigs_su3(npts):
    t1 = torch.linspace(-np.pi, np.pi, npts)
    t2 = torch.linspace(-np.pi, np.pi, npts)

    theta1, theta2 = torch.meshgrid((t1, t2))
    theta1 = theta1.flatten()
    theta2 = theta2.flatten()

    # on_mani construction is diagonal matrices
    on_mani = torch.zeros(theta1.shape[0], 3, 3, dtype=torch.complex64)
    on_mani[:,0,0] = torch.exp(1j * theta1)
    on_mani[:,1,1] = torch.exp(1j * theta2)
    on_mani[:,2,2] = 1 / (on_mani[:,0,0]*on_mani[:,1,1])

    # detjac is just log prob, look at boyda et al. page 7 for reference
    log_detjac = HaarSUN().log_prob(on_mani)

    return on_mani, (theta1, theta2), log_detjac
