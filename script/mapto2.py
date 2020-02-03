import sys

import torch as tp
import pyro
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm as cm
plt.style.use('ggplot')

from GPyOpt.methods import BayesianOptimization


LogNormal = pyro.distributions.LogNormal
Normal    = pyro.distributions.Normal


import pickle

sys.path.insert(1, '..\\src\\cahnhilliard.py')
## Cahn Hilliard code
from demixing_2d import *

def maptotheta(x):
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
