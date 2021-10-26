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

def priors():
    return {
        'N': torch.tensor([5e4, 1.5e5]),
        'chi': torch.tensor([2e-5, 2.9e-5]),
        'k': torch.tensor([6.832446e-1, 7.2680455e-1])
    }

def run(device='cpu'):
    device = safe_cast(torch.device, device)

    dx, NX = PARAM_DX, PARAM_MESH_RES_SPACE

    ts = torch.arange(PARAM_MESH_RES_TIME, device=device)

    priors_uniform = priors()

    if FLAG_SEED:
        torch.manual_seed(125329)

    params_true = {}
    for k,v in priors_uniform.items():
        params_true[k] = (v[1]-v[0])*torch.rand(size=(1,), device=device) + v[0]

    sim_pde = FloryHugginsCahnHilliard(
        params = params_true,
        M      = PARAM_DT,
        dx     = dx
    )

    phi0 = 0.2 * torch.rand((NX, NX), device=device).view(-1,1,NX,NX) + 0.5
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
    for k,v in params_true.items():
        theta[k] = v.detach().numpy()
    return theta, obs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'output_file', action='store', help='Filename for pickled data file'
    )

    parser.add_argument(
        '--gpu', action='store_true', default=False, dest='FLAG_USE_GPU',
        help='Indicate whether to use CUDA to generate data'
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
    args = parser.parse_args()

    FLAG_USE_GPU         = args.FLAG_USE_GPU
    PARAM_MESH_RES_SPACE = args.PARAM_MESH_RES_SPACE
    PARAM_MESH_RES_TIME  = 1e4
    PARAM_DX, PARAM_DT   = args.PARAM_DX, args.PARAM_DT
    FLAG_MODEL_LANDAU    = True
    FLAG_SEED            = args.FLAG_SEED

    device = torch.device('cuda' if FLAG_USE_GPU else 'cpu')
    gen_data = run(device)

    with open(args.output_file, 'wb') as fid:
        pickle.dump(gen_data, fid)
