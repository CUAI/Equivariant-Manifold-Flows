import torch
from torch import optim
import numpy as np
import matplotlib.pyplot as plt

import os
import glob
import argparse

from flows.utils import check_mkdir
from flows.mcnf_su3 import SU3_CNF

from distributions.haarsun import HaarSUN

from test_densities.density_su3 import eval_density_log_su3_boyda, plot_distr_su3


parser = argparse.ArgumentParser()
parser.add_argument('--example_density', type=str, default='boyda1',
                    help='example density to learn. Options are: boyda1 | boyda2 | boyda3 (different densities defined on SU(3))')
parser.add_argument('--num_drops', default=2, type=int, help='number of times to drop the learning rate')
parser.add_argument('--save_viz', action='store_true', default=True, help='Save a visualization of the learned density once training is completed')
parser.add_argument('--save_model', action='store_true', default=True, help='Save a visualization of the learned sampler')
parser.add_argument('--dev', type=str, default='cpu',
                    help='Default device; cpu preferred due to poor support of complex tensor differentiation on cuda')
parser.add_argument('--M', type=str, default='SU3', choices=['SU3'], help='Manifold over which to learn')
parser.add_argument('--contsave', action='store_true', default=False, help='Continuously save intermediate flow visualization in contsave/')
parser.add_argument('--save_freq', type=int, default=5, help='frequency of continuous saving of intermediate flows')
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--batch_size', default=200, type=int)
parser.add_argument('--lr', type=float, default='1e-3')
parser.add_argument('--weight_decay', type=float, default=1e-5)

args = parser.parse_args()

if args.dev == 'cuda':
    args.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    args.dev = torch.device(args.dev)

# set constructs based on specified manifold
assert args.M in ['SU2', 'SU3'], "Manifold not supported"

if args.M == 'SU3':
    base_distr = HaarSUN()
    make_model = SU3_CNF
    plot_distr = plot_distr_su3
    data_target_density = eval_density_log_su3_boyda
else:
    raise Error('Manifold not supported')


# x should be a set of uniform samples over the manifold
def compute_loss(args, model, x):
    z, delta_logp = model.inverse(x)
    logpz = base_distr.log_prob(z) + delta_logp
    logq = data_target_density(args.example_density, x)

    # normalize to make logq, logpz into true "empirical ditributions"
    pz = torch.exp(logpz)
    pz = pz / torch.mean(pz)
    logpz = torch.log(pz)

    q = torch.exp(logq)
    q = q / torch.mean(q)
    logq = torch.log(q)

    loss = (torch.exp(logq) * (logq - logpz)).mean()

    return loss


def main(args):
    model = make_model().to(args.dev)

    print(f'Running with: {vars(args)}')

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    batch_size = args.batch_size

    # number of times to drop learning rate
    num_drops = args.num_drops
    lr_milestones = [j*args.epochs//(num_drops+1) for j in range(1,num_drops+1)]
    scheduler = optim.lr_scheduler.MultiStepLR(opt, lr_milestones, gamma=.1)

    if args.contsave:
        check_mkdir(f'contsave/{args.example_density}_{args.M}/')
        files = glob.glob(f'contsave/{args.example_density}_{args.M}/*')
        for f in files:
            os.remove(f)

    for epoch in range(0, args.epochs):
        if args.M == 'SU3':
            # SU(3) sampling from prior
            samples = HaarSUN().rsample(batch_size, 3).to(args.dev)

        opt.zero_grad()

        loss = compute_loss(args, model, samples)
        loss.backward()

        opt.step()
        scheduler.step()
        train_loss = loss.item()/batch_size

        # print update
        if epoch % 2 == 0:
            print(f'Epoch: {epoch}, Loss: {train_loss}')

        if args.contsave and epoch % args.save_freq == 0:
            try:
                print('Trying to save...')
                namestr = f'contsave/{args.example_density}_{args.M}/{str(epoch).zfill(4)}'
                plot_distr(distr='model', model=model, device=args.dev, save_fig=True, namestr=namestr, res_npts=200)
                plt.close()
            except:
                print('Could not save')

    # plot density evaluated on grid
    namestr = f'{args.example_density}_{args.M}'
    plot_distr(distr='model', model=model, device=args.dev, save_fig=args.save_viz, namestr=namestr)

    if args.save_model:
        torch.save(model.state_dict(), f'model_{args.M}_for_{args.example_density}.pth')

    return model


if __name__ == '__main__':
    model = main(args)
