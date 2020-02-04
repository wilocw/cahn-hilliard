import argparse

import pickle

import torch
import numpy as np

from torchdiffeq import odeint_adjoint as odeint
from pdes.models import LandauCahnHilliard, FloryHugginsCahnHilliard
from pdes.util import safe_cast, scattering

from gpytorch.models import ExactGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal

from pyDOE import lhs # latin-hypercube sampling

FLAG_USE_GPU         = False
PARAM_MESH_RES_SPACE = 256
PARAM_MESH_RES_TIME  = 1e4
PARAM_DX, PARAM_DT   = 5e2, 1e8
FLAG_MODEL_LANDAU    = True
FLAG_SEED            = True
PARAM_INIT_EVAL      = 5

def sqdiff(y,y_):
    ''' (y-y')**2 '''
    return ((y - y_)**2).view(-1)
def mse(*args):
    return sqdiff(*args).mean()
def loss(true_y, sim_y):
    return mse(true_y, sim_y)

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

    y = torch.tensor(obs['Ss'], device=device)

    def simulate(params):
        _theta = {'a':params[0], 'b':params[1], 'k':params[2]}
        sim_pde = LandauCahnHilliard(
            params = _theta,
            M      = PARAM_DT,
            dx     = dx,
            device = device
        )

        phi0 = torch.nn.Parameter(0.2 * torch.rand((NX, NX), device=device)).view(-1,1,NX,NX)

        print(phi0)
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

        return loss(y, Ss), sim_phis

    pgd = lhs(3, samples=PARAM_INIT_EVAL) # calculate initial samples from latin hypercube
    xs, ys = [],[]

    for j in range(PARAM_INIT_EVAL):
        xk = torch.stack(
            [torch.nn.Parameter((priors_uniform[k][1]-priors_uniform[k][0])*torch.tensor(pgd[j,i], device=device, dtype=torch.float32) + priors_uniform[k][0]) for i,k in enumerate(('a','b','k'))],
            0
        )
        xs.append(xk)

    params = xs[0]
    print(params[0])

    ell, phis = simulate(params)
    print('simulated')
    ell.backward()
    print('calculated gradients')
    #[print(p.grad) for p in params];
    print(params.grad)

    #
    # params = {}
    # for k,v in priors_uniform.items():
    #     params[k] = (v[1]-v[0])*torch.rand(size=(1,), device=device) + v[0]
    #
    # print(simulate(params))



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
        '-e', action='store', default=5, dest='PARAM_INIT_EVAL',
        help='Number of (quasi-uniform) evaluations to train GP before beginning search'
    )


    args = parser.parse_args()

    FLAG_USE_GPU         = args.FLAG_USE_GPU
    PARAM_MESH_RES_SPACE = args.PARAM_MESH_RES_SPACE
    PARAM_MESH_RES_TIME  = 1e4
    PARAM_DX, PARAM_DT   = args.PARAM_DX, args.PARAM_DT
    FLAG_MODEL_LANDAU    = True
    FLAG_SEED            = args.FLAG_SEED
    PARAM_INIT_EVAL      = args.PARAM_INIT_EVAL
    # PARAM_BATCH_SIZE     = args.PARAM_BATCH_SIZE

    device = torch.device('cuda' if FLAG_USE_GPU else 'cpu')
    # gen_data = run(device)

    with open(args.data_file, 'rb') as fid:
        params_true, obs = pickle.load(fid)

    results = run(obs, params_true, device=device)
    #
    # with open('test_simulation','wb') as fid:
    #     pickle.dump(results, fid)