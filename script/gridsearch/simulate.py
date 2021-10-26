import argparse

import os
import pickle

import torch
import numpy as np

from torchdiffeq import odeint_adjoint as odeint
from pdes.models import LandauCahnHilliard, FloryHugginsCahnHilliard
from pdes.util import safe_cast, scattering


def priors(landau=True):
    ''' priors for landau and flory huggins '''
    return {
        'a': torch.tensor([5.6006147e-6, 8.9186578e-6]),
        'b': torch.tensor([1.1691e-4, 1.2309e-4]),
        'k': torch.tensor([6.832446e-1, 7.2680455e-1])} if landau else {
        'N': torch.tensor([5e4, 1.5e5]),
        'chi': torch.tensor([2e-5, 2.9e-5]),
        'k': torch.tensor([6.832446e-1, 7.2680455e-1])}


def simulate(params, args, device, simulator=LandauCahnHilliard):
    ''' '''
    dx, NX = args.PARAM_DX, args.PARAM_MESH_RES_SPACE

    ts = torch.arange(args.PARAM_MESH_RES_TIME, device=device)

    all_obs, all_theta = [], []
    NP, mini_batch_size = args.PARAM_GRID_RES**3, args.PARAM_BATCH_SIZE
    for mb in range(0, NP, mini_batch_size):
        batch_params = {}
        for k,v in params.items():
            batch_params[k] = v[mb:mb+mini_batch_size]

        sim_pde = simulator(
            params = batch_params,
            M      = args.PARAM_DT,
            dx     = dx,
            device = device)

        phi0 = 0.2 * torch.rand((NX, NX), device=device).view(-1,1,NX,NX)
        sim_phis = odeint(sim_pde, phi0, ts, method='euler')

        t_ = ts[99::900]
        phis_ = sim_phis[99::900,:,0,...]

        Ss = []
        for i in range(len(t_)):
            if i > 0:
                S_ = scattering(phis_[i,...], dx)
            else:
                q, S_ = scattering(phis_[i,...], dx, True)
            Ss.append(S_)

        Ss = torch.stack(Ss, 0)

        obs = {'t': t_.numpy(), 'q': q.detach().numpy(), 'Ss': Ss.detach().numpy()}
        theta = {}
        for k,v in params.items():
            theta[k] = v.detach().numpy()

        if mini_batch_size > 1:
            out_name = 'simulation_%03d-%03d.p' % (mb, mb+mini_batch_size-1)
        else:
            out_name = 'simulation_%03d.p' % mb
        out_path = os.path.join(args.OUT_DIR, out_name)
        with open(out_path, 'wb') as fid:
            pickle.dump((theta, obs), fid)

def generate_params(args, device):
    ''' Generates grid of parameters to simulate '''
    priors_uniform = priors(args.FLAG_MODEL_LANDAU)

    grid = torch.meshgrid(
        *[torch.linspace(0., 1., steps=args.PARAM_GRID_RES, device=device) for _ in range(3)])

    params = {}
    for i,k in enumerate(priors_uniform):
        v = priors_uniform[k]
        params[k] = (v[1]-v[0])*(grid[i].reshape(-1)) + v[0]

    return params


def run(args, device='cpu'):
    ''' Run main body '''
    device = safe_cast(torch.device, device)

    params = generate_params(args, device)

    simulate(
        params, args, device,
        simulator=LandauCahnHilliard if args.FLAG_MODEL_LANDAU else
            FloryHugginsCahnHilliard)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-d', '--out_dir', type=str, dest='OUT_DIR',
        help='Filename for pickled simulations')

    parser.add_argument(
        '--gpu', action='store_true', default=False, dest='FLAG_USE_GPU',
        help='Indicate whether to use CUDA to generate simulations')

    parser.add_argument(
        '-p', '--param_res', type=int, default=5, dest='PARAM_GRID_RES',
        help='Number of each parameter to simulate (total simulations p**3)')

    parser.add_argument(
        '-dx', default=5e2, type=int, dest='PARAM_DX',
        help='Spatial mesh scaling parameter')

    parser.add_argument(
        '-dt', default=1e8, type=int, dest='PARAM_DT',
        help='Temporal mesh scaling parameter')

    parser.add_argument(
        '-NX', default=256, type=int, dest='PARAM_MESH_RES_SPACE',
        help='Spatial mesh size (Hx x Hx)')

    parser.add_argument(
        '-NT', action='store', default=1e4, type=int, dest='PARAM_MESH_RES_TIME',
        help='Temporal mesh size')

    parser.add_argument(
        '--no_seed', action='store_false', default=True, dest='FLAG_SEED',
        help='Flag to turn off random seed')

    parser.add_argument(
        '-nmb', default=1, type=int, dest='PARAM_BATCH_SIZE',
        help='Mini-batch size for concurrent simulation (may get OOM if too high)')

    parser.add_argument(
        '-L', '--landau', action='store_true', default=False, dest='FLAG_TMP_LANDAU',
        help='set model to landau (default behaviour)')

    parser.add_argument(
        '-F', '--floryhuggins', action='store_true', default=False,
        dest='FLAG_TMP_FH',
        help='set model to flory huggins')

    # DEFAULT PARAMS
    # FLAG_USE_GPU         = False
    # PARAM_GRID_RES       = 5
    # PARAM_MESH_RES_SPACE = 256
    # PARAM_MESH_RES_TIME  = 1e4
    # PARAM_DX, PARAM_DT   = 5e2, 1e8
    # FLAG_MODEL_LANDAU    = True
    # FLAG_SEED            = True
    # PARAM_BATCH_SIZE     = 1

    args = parser.parse_args()

    if args.FLAG_TMP_LANDAU:
        args.FLAG_MODEL_LANDAU = True
        if args.FLAG_TMP_FH:
            print(
                'Warning: both Landau and Flory Huggins set',
                ' defaulting to Landau')
    elif args.FLAG_TMP_FH:
        args.FLAG_MODEL_LANDAU = False
    else:
        args.FLAG_MODEL_LANDAU = True

    if args.OUT_DIR is None:
        cwd = os.path.abspath(os.getcwd())
        args.OUT_DIR = os.path.join(
            cwd, 'simulations', 'gridsearch',
            'landau' if args.FLAG_MODEL_LANDAU else 'floryhuggins')
        print('No output directory specified. Writing out to ')
        print(args.OUT_DIR)

    if not os.path.exists(args.OUT_DIR):
        os.makedirs(args.OUT_DIR)

    device = torch.device('cuda' if args.FLAG_USE_GPU else 'cpu')
    print('Using device: {}'.format(device))

    run(args, device)
