import argparse

import pickle

import torch
import numpy as np

from torchdiffeq import odeint_adjoint as odeint
from pdes.models import LandauCahnHilliard, FloryHugginsCahnHilliard
from pdes.util import safe_cast, scattering

from gpytorch.models import ExactGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import MultitaskGaussianLikelihood, FixedNoiseGaussianLikelihood
from gpytorch.means import ConstantMeanGrad
from gpytorch.kernels import ScaleKernel, RBFKernelGrad
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.settings import fast_pred_var, lazily_evaluate_kernels

from pyDOE import lhs # latin-hypercube sampling

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

def optimise(model, method='lbfgs', max_iter=None):
    if method is 'lbfgs': optimise_lbfgs(model)
    if method is 'adam': optimise_adam(model) if max_iter is None else optimise_adam(model, max_iter)

def optimise_lbfgs(model):
    model.train()
#     model.likelihood.train()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    def closure():
        output = model(model.train_inputs[0])
        loss   = -mll(output, model.train_targets)
        opt.zero_grad()
        loss.backward()
        return loss
    opt = torch.optim.LBFGS([{'params':model.parameters()}], max_iter=1000)
    opt.step(closure)

def optimise_adam(model, max_iter=50, fix_likelihood=True):
    model.train() # set model to train (gpytorch state)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    opt = torch.optim.Adam([ # use Adam optimiser (recommended by gpytorch) on parameters (ignore likelihood for bayesopt)
        {'params': [v for k,v in model.named_parameters() if 'likelihood' not in k] if fix_likelihood else model.parameters()}
    ], lr=0.1)
    for _ in range(max_iter): # no convergence criteria implemented but 1000 generally more than enough
        output = model(model.train_inputs[0]) # predicted f*
        loss   = -mll(output, model.train_targets) # MLE(f*, y)
        opt.zero_grad() # reset optimiser for this step
        loss.backward() # calculate gradients of MLE
        opt.step()      # iterate optimiser

class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMeanGrad()
        self.base_kernel = RBFKernelGrad(arg_num_dims=3)
        self.covar_module = ScaleKernel(self.base_kernel)
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)
#
# def acq(fo, model, x_eval=None):
#     model.eval()
#     if x_eval is None: x_eval = torch.linspace(0,1,100)
#     with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.lazily_evaluate_kernels(True):
#         f_ = model(x_eval)
#         mu, sig = f_.mean, f_.variance#covariance_matrix
#
#     _cdf = 0.5*(1+torch.erf((fo-mu)/(torch.sqrt(sig*2.))))
#     _pdf = torch.exp(-(fo-mu)**2/(2*sig))/torch.sqrt(sig*2*3.141593)
#     return (fo-mu)*_cdf + sig*_pdf
#

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

def acq(fo, model, x_eval=None):
    model.eval()
    if x_eval is None: x_eval = torch.linspace(0,1,100).unsqueeze(-1)
    print(x_eval.shape)
    batch_sz = 100
    with torch.no_grad(), fast_pred_var(), lazily_evaluate_kernels(True):
        mu, sig = [], []
        for i in range(0, x_eval.shape[0], batch_sz):
            f_ = model(x_eval[i:i+batch_sz,:])
            mu.append(f_.mean[:,0])
            sig.append(f_.variance[:,0])#covariance_matrix
        mu = torch.cat(mu,0)
        sig = torch.cat(sig,0)
        print(mu.shape)

    _cdf = 0.5*(1+torch.erf((fo-mu)/(torch.sqrt(sig*2.))))
    _pdf = torch.exp(-(fo-mu)**2/(2*sig))/torch.sqrt(sig*2*3.141593)
    return (fo-mu)*_cdf + sig*_pdf


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
        loss_fn = Evaluator(sim_pde, loss)

        return loss_fn

    pgd = lhs(3, samples=PARAM_INIT_EVAL) # calculate initial samples from latin hypercube
    xs, ys = [],[]

    for j in range(PARAM_INIT_EVAL):
        xk = torch.stack(
            [(priors_uniform[k][1]-priors_uniform[k][0])*torch.tensor(pgd[j,i], device=device, dtype=torch.float32) + priors_uniform[k][0] for i,k in enumerate(('a','b','k'))],
            0
        )
        xs.append(xk)
#    ell, params = simulate(params)

    phi0 = (0.2 * torch.rand((NX, NX), device=device)).view(-1,1,NX,NX)

    # with torch.no_grad():
    for j in range(PARAM_INIT_EVAL):
        params = xs[j]
        loss_fn = simulate(params)
        # print(loss_fn._pde)
        # for n,p in loss_fn.named_parameters():
        #     print(n + '->' + str(p))
        ell = loss_fn(phi0, ts, y, dx)
        ell.backward()
        grads = (
            loss_fn._pde._a.grad,
            loss_fn._pde._b.grad,
            loss_fn._pde._k.grad
        )
        ys.append(torch.stack([ell.detach(), *grads]).unsqueeze(0))
        print('init sample %d/%d' % (j, PARAM_INIT_EVAL))

    x_init, y_init = torch.stack(xs), torch.cat(ys,0)

    #
    # print(y_init)
    N = PARAM_SEARCH_RES
    x_eval = torch.cat([x.reshape(-1,1) for x in torch.meshgrid(
        *[torch.linspace(priors_uniform[k][0], priors_uniform[k][1], N)\
            for k in priors_uniform]
    )],1)

    print(x_init.shape)
    print(x_eval.shape)
    x_train = x_init
    y_train = y_init

    print(x_init)
    print(y_init)

    jit = 1e-2

    lik = MultitaskGaussianLikelihood(num_tasks=4)
    lik.noise_covar.noise = jit*torch.ones(4)
    lik.noise = torch.tensor(jit).sqrt()

    for i in range(PARAM_MAX_EVAL - PARAM_INIT_EVAL):
        for ntry in range(5):
            model = ExactGPModel(x_train, y_train, lik)
            try:
                optimise(model, method='adam', max_iter=1000)
                break
            except Exception as err:
                print('attempt %d failed' % ntry)
                if ntry == 4:
                    raise err

        u = acq(y_train[:,0].min(), model, x_eval)
        idx = u.argmax()
        xn = x_eval[idx,:]
        loss_fn = simulate(xn)

        ell = loss_fn(phi0, ts, y, dx)
        ell.backward()
        grads = (
            loss_fn._pde._a.grad,
            loss_fn._pde._b.grad,
            loss_fn._pde._k.grad
        )
        #ys.append(torch.stack([ell.detach(), *grads]).unsqueeze(0))

        yn = torch.stack([ell.detach(), *grads],-1).unsqueeze(0)

        x_eval = torch.cat([x_eval[0:idx,:], x_eval[idx+1:,:]],0)
        x_train = torch.cat([x_train, xn.reshape(1,-1)])

        # y_train = torch.stack([*y_train, yn.detach()])
        y_train = torch.cat([y_train, yn], 0)
        print(x_train)
        print(y_train)

        print(i)
    #
    return (x_train, y_train)
    # ell.backward()
    # print('calculated gradients')
    # for name, param in loss_fn.named_parameters():
    #     print(name)
    #     print(param.grad)
    #
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

    print('pred.')
    print(results[0][results[1].argmin(),:])
    print('true.')
    print(params_true)

    with open('bayesopt_result.p','wb') as fid:
        pickle.dump(results, fid)
    #
    # with open('test_simulation','wb') as fid:
    #     pickle.dump(results, fid)
