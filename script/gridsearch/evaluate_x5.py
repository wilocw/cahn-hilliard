import argparse

import pickle

import torch
import numpy as np

from torchdiffeq import odeint_adjoint as odeint
from pdes.models import LandauCahnHilliard, FloryHugginsCahnHilliard
from pdes.util import safe_cast, scattering

FLAG_USE_GPU         = False
PARAM_MESH_RES_SPACE = 256
PARAM_MESH_RES_TIME  = 1e4
PARAM_DX, PARAM_DT   = 5e2, 1e8
FLAG_MODEL_LANDAU    = True
FLAG_SEED            = True
PARAM_BATCH_SIZE     = 1

def priors():
    return {
        'a': torch.tensor([5.6006147e-6, 8.9186578e-6]),
        'b': torch.tensor([1.1691e-4, 1.2309e-4]),
        'k': torch.tensor([6.832446e-1, 7.2680455e-1])
    }


def run(obs, params_true, device='cpu'):
    device = safe_cast(torch.device, device)

    dx, NX = PARAM_DX, PARAM_MESH_RES_SPACE

    ts = torch.arange(PARAM_MESH_RES_TIME, device=device)

    priors_uniform = priors()

    params = {}

    grid = torch.meshgrid(
        *[torch.linspace(0., 1., steps=5, device=device) for _ in range(3)]
    )

    for i,k in enumerate(priors_uniform):
        v = priors_uniform[k]
        params[k] = (v[1]-v[0])*(grid[i].reshape(-1)) + v[0]

    all_obs, all_theta = [], []
    NP, mini_batch_size = 125, PARAM_BATCH_SIZE
    for mb in range(0, NP, mini_batch_size):

        batch_params = {}
        for k,v in params.items():
            batch_params[k] = v[mb:mb+mini_batch_size]

        print(batch_params)

        sim_pde = LandauCahnHilliard(
            params = batch_params,
            M      = PARAM_DT,
            dx     = dx,
            device = device
        )

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

        with open('.tmp\simulation_%03d.p' % mb, 'wb') as fid:
            pickle.dump((theta, obs), fid)
    # return all_theta, all_obs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'data_file', action='store', help='Filename for pickled data file'
    )

    parser.add_argument(
        '--gpu', action='store_true', default=False, dest='FLAG_USE_GPU',
        help='Indicate whether to use CUDA to generate simulations'
    )
    parser.add_argument(
        '-dx', action='store', default=5e2, dest='PARAM_DX',
        help='Spatial mesh scaling parameter'
    )
    parser.add_argument(
        '-dt', action='store', default=1e8, dest='PARAM_DT',
        help='Temporal mesh scaling parameter'
    )
    parser.add_argument(
        '-NX', action='store', default=256, dest='PARAM_MESH_RES_SPACE',
        help='Spatial mesh size (Hx x Hx)'
    )
    parser.add_argument(
        '--no_seed', action='store_false', default=True, dest='FLAG_SEED',
        help='Flag to turn off random seed'
    )
    parser.add_argument(
        '-nmb', action='store', default=1, dest='PARAM_BATCH_SIZE',
        help='Mini-batch size for concurrent simulation (may get OOM if too high)'
    )


    args = parser.parse_args()

    FLAG_USE_GPU         = args.FLAG_USE_GPU
    PARAM_MESH_RES_SPACE = args.PARAM_MESH_RES_SPACE
    PARAM_MESH_RES_TIME  = 1e4
    PARAM_DX, PARAM_DT   = args.PARAM_DX, args.PARAM_DT
    FLAG_MODEL_LANDAU    = True
    FLAG_SEED            = args.FLAG_SEED
    PARAM_BATCH_SIZE     = args.PARAM_BATCH_SIZE

    device = torch.device('cuda' if FLAG_USE_GPU else 'cpu')
    # gen_data = run(device)

    # with open(args.data_file, 'rb') as fid:
    #     params_true, obs = pickle.load(fid)

    results = run([], [], device=device)

    with open(args.data_file,'wb') as fid:
        pickle.dump(results, fid)
