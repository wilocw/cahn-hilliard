import argparse

import pickle

from scipy.optimize import minimize

import torch
import numpy as np

from torchdiffeq import odeint_adjoint as odeint
from pdes.models import LandauCahnHilliard, FloryHugginsCahnHilliard
from pdes.util import safe_cast, scattering

global k
global OPT_STATE
global phi0

FLAG_USE_GPU         = False
PARAM_MESH_RES_SPACE = 256
PARAM_MESH_RES_TIME  = 1e4
PARAM_DX, PARAM_DT   = 5e2, 1e8
FLAG_MODEL_LANDAU    = True
FLAG_SEED            = True
PARAM_INIT_EVAL      = 5
PARAM_SEARCH_RES     = 100
PARAM_MAX_EVAL       = 50

class Loss(torch.nn.Module):
    def __init__(self, sim_y):
        super(Loss, self).__init__()
        self._y = torch.nn.Parameter(sim_y)

    def forward(self, true_y):
        return loss(true_y, self._y)

class Evaluator(torch.nn.Module):
    def __init__(self, pde, loss):
        super(Evaluator, self).__init__()
        self._pde  = pde
        self._loss = loss
        # for idx, m in enumerate(self.modules()):
            # print(idx, '->', m)
#
    def forward(self, x0, ts, y, dx = 1.):
        sim_phis = odeint(self._pde, x0, ts, method='euler')

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
        return self._loss(y, Ss)

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

def print_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            print(param.data)


def run(obs, params_true, device='cpu', optimiser='lbfgs'):
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
        loss_fn = Evaluator(sim_pde, loss)

        return loss_fn


    if optimiser is 'lbfgs':
        param0 = torch.stack([
            priors_uniform[k][0] + (priors_uniform[k][1]-priors_uniform[k][0])*torch.rand(1,device=device, dtype=torch.float32) for k in ('a','b','k')
        ],0)
        # loss_fn = simulate(param0)
        # opt = torch.optim.LBFGS(
        #     [{'params': [
        #         loss_fn._pde._a, loss_fn._pde._b, loss_fn._pde._k
        #     ]}],
        #     max_iter=125
        # )
        # print(param0)
        global k
        k = 0
        # def closure():
        #     global k
        #     print(k)
        #     phi0 = (0.2 * torch.rand((NX, NX), device=device)).view(-1,1,NX,NX)
        #     ell  = loss_fn(phi0, ts, y, dx)
        #     opt.zero_grad()
        #     ell.backward()
        #     # print(i)
        #     k+=1
        #     return ell
        # opt.step(closure)
        # print(param0)
        # return (param0, opt.param_groups[0]['params'])
##
        global phi0
        phi0 = (0.2*torch.rand((NX, NX), device=device)).view(-1,1,NX,NX)
        def _f(params):
            global OPT_STATE
            global phi0
            loss_fn = simulate(torch.tensor(params,dtype=torch.float32).unsqueeze(1))
            ell = loss_fn(phi0, ts, y, dx)
            ell.backward()
            grads = np.array([
                loss_fn._pde._a.grad.item(),
                loss_fn._pde._b.grad.item(),
                loss_fn._pde._b.grad.item()
            ])
            OPT_STATE = (ell.detach().item(), grads)
            # print(OPT_STATE[0])
            return OPT_STATE[0]

        def _df(params):
            global OPT_STATE
            # print(OPT_STATE[1])
            return OPT_STATE[1]

        def _callback(xk):
            global k
            global OPT_STATE
            print('%d l=%9.5f | a:%8.6fe-6   b: %8.6fe-4   k: %8.6fe-1' % (k,OPT_STATE[0],xk[0]*1e6,xk[1]*1e4,xk[2]+1e1))
            k+=1

        # print(param0.numpy())
        # print(_f(param0.numpy()))
        # print([
        #     (priors_uniform[p][0].item(), priors_uniform[p][1].item()) for p in priors_uniform
        # ])
        # print(OPT_STATE)

        res = minimize(
            _f, param0.numpy().ravel(),
            method='L-BFGS-B', jac=_df,
            bounds=[
                (priors_uniform[p][0].item(), priors_uniform[p][1].item()) for p in priors_uniform
            ], callback=_callback, options={'maxiter':125})
        return res

    elif optimiser is 'adam':
        pass


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
        '-dx', action='store', type=int, default=5e2, dest='PARAM_DX',
        help='Spatial mesh scaling parameter'
    )
    parser.add_argument(
        '-dt', action='store', type=int, default=1e8, dest='PARAM_DT',
        help='Temporal mesh scaling parameter'
    )
    parser.add_argument(
        '-NX', action='store', type=int, default=256, dest='PARAM_MESH_RES_SPACE',
        help='Spatial mesh size (Hx x Hx)'
    )
    parser.add_argument(
        '--no_seed', action='store_false', default=True, dest='FLAG_SEED',
        help='Flag to turn off random seed'
    )
    parser.add_argument(
        '-e', action='store', type=int, default=5, dest='PARAM_INIT_EVAL',
        help='Number of (quasi-uniform) evaluations to train GP before beginning search'
    )
    parser.add_argument(
        '-m', action='store', type=int, default=100, dest='PARAM_SEARCH_RES',
        help='Resolution of GP prediction during surrogate modelling'
    )

    parser.add_argument(
        '-max_e', action='store', type=int, default=50, dest='PARAM_MAX_EVAL',
        help='Maximum number of evaluations of simulator (includes initial evaulations)'
    )

    args = parser.parse_args()

    FLAG_USE_GPU         = args.FLAG_USE_GPU
    PARAM_MESH_RES_SPACE = args.PARAM_MESH_RES_SPACE
    PARAM_MESH_RES_TIME  = 1e4
    PARAM_DX, PARAM_DT   = args.PARAM_DX, args.PARAM_DT
    FLAG_MODEL_LANDAU    = True
    FLAG_SEED            = args.FLAG_SEED
    PARAM_INIT_EVAL      = args.PARAM_INIT_EVAL
    PARAM_SEARCH_RES     = args.PARAM_SEARCH_RES
    PARAM_MAX_EVAL       = args.PARAM_MAX_EVAL
    # PARAM_BATCH_SIZE     = args.PARAM_BATCH_SIZE
    device = torch.device('cuda' if FLAG_USE_GPU else 'cpu')
    # gen_data = run(device)

    with open(args.data_file, 'rb') as fid:
        params_true, obs = pickle.load(fid)

    results = run(obs, params_true, device=device)

    with open('lbfgs_result.p','wb') as fid:
        pickle.dump(results, fid)
    #
    # with open('test_simulation','wb') as fid:
    #     pickle.dump(results, fid)
