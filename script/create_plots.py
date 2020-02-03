import sys

import torch as tp
import pyro
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm as cm
plt.style.use('ggplot')

from GPyOpt.methods import BayesianOptimization

import pickle

sys.path.insert(1, '..\\src\\cahnhilliard.py')
## Cahn Hilliard code
from demixing_2d import *

LogNormal = pyro.distributions.LogNormal
Normal    = pyro.distributions.Normal

GLOBAL_DEVICE = tp.device('cpu')

def load_true_theta(run=0):
    fname = [
        'ch_landau_0511141003_run_0',
        'ch_landau_0611031012_run_1',
        'ch_landau_0611083851_run_2',
        'ch_landau_0611140804_run_3',
        'ch_landau_0611193745_run_4',
        'ch_landau_0511214221_run_0',
    ][run]
    with open(fname,"rb") as fid:
        experiment = pickle.load(fid)
    return experiment['true']['theta']

def load_true_y(run=0):
    fname = [
        'ch_landau_0511141003_run_0',
        'ch_landau_0611031012_run_1',
        'ch_landau_0611083851_run_2',
        'ch_landau_0611140804_run_3',
        'ch_landau_0611193745_run_4',
        'ch_landau_0511214221_run_0',
    ][run]
    with open(fname,"rb") as fid:
        experiment = pickle.load(fid)
    return experiment['true']['y']


def simulate(theta=None, nt=9, max_iter=1e4, seed=None, device=GLOBAL_DEVICE):
    params = Variables(
            theta = sample_priors(
                 {
                    'a': LogNormal(-11.86, 0.1),
                    'b': Normal(1.2e-4, 1e-6),
                    'k': LogNormal(-0.35, 0.01)
                },
                device=device
            ) if theta is None else theta
        )
    # Construct Cahn Hilliard simulation model
    model = CahnHilliard(params=params, dim=2, seed=seed, device=device)

    ts = np.linspace(0, max_iter, nt).astype(np.int)

    q, S0 = model.scattering()
    S = tp.zeros(S0.shape[0], nt)
    S[:,0] = S0

    t = 1
    for k in range(int(max_iter)):
        model.iterate()
        if (k+1) in ts:
            _, St = model.scattering()
            S[:,t] = St
            t+=1
    return ts, q, S


def sqdiff(y,y_):
    ''' (y-y')**2 '''
    return ((y - y_)**2).view(-1)
def mse(*args):
    return sqdiff(*args).mean()
def rmse(*args):
    return tp.sqrt(mse(*args))
def sse(*args):
    return sqdiff(*args).sum()

def loss(true_y, sim_y, metric=mse):
    return metric(true_y[2], sim_y[2])

def run(_run=0):

    true_theta = load_true_theta(_run)
    true_y = load_true_y(_run)

    theta_priors = {
        'a': LogNormal(-11.86, 0.1),
        'b': Normal(1.2e-4, 1e-6),
        'k': LogNormal(-0.35, 0.01)
    }

    def f_of_(X, device=GLOBAL_DEVICE):
        a = theta_priors['a'].icdf(tp.tensor(X[0,0]).to(device))
        b = theta_priors['b'].icdf(tp.tensor(X[0,1]).to(device))
        k = theta_priors['k'].icdf(tp.tensor(X[0,2]).to(device))
        sample = simulate({'a':a, 'b':b, 'k':k}, device=device)
        return loss(true_y, sample).cpu().item()

    domain = [
        {'name': 'a', 'type':'continuous','domain':(0.0001, 0.9999)},
        {'name': 'b', 'type':'continuous','domain':(0.0001, 0.9999)},
        {'name': 'k', 'type':'continuous','domain':(0.0001, 0.9999)}
    ]

    experiment = BayesianOptimization(f = lambda x: f_of_(x), domain=domain)
    experiment.run_optimization(max_iter=35)

    return {'y':true_y, 'theta':true_theta}, experiment

def generate_animation_data(run=0, max_iter=1e4, seed=None):
    theta = load_true_theta(run)
    params = Variables(theta=theta)
    model = CahnHilliard(params=params, seed=seed, dim=2, device=GLOBAL_DEVICE)

    animation_data = []
    frame_capture_rate = 50 # every 50 iterations

    for k in range(int(max_iter+1)):
        if (k % 50) is 0:
            animation_data.append(model.phi.cpu().numpy().copy())
        model.iterate()
    return animation_data

def animate(data):
    NotImplemented

def maptotheta(x)
    theta_priors = {
        'a': LogNormal(-11.86, 0.1),
        'b': Normal(1.2e-4, 1e-6),
        'k': LogNormal(-0.35, 0.01)
    }
    keys = ('a','b','k')
    theta = []
    for i,v in enumerate(x):
        theta.append(theta_priors[keys[i]].icdf(tp.tensor(x[i])))
    return theta
