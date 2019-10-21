import torch as tp
import pyro
import numpy as np

from matplotlib import pyplot as plt
plt.style.use("ggplot")

class Variables():
    ''' Hyperparameters and parameters '''
    def __init__(self, delta=None, phibar=None, theta=None, M=None):
        self.delta  = delta  if delta  is not None else self._defaults('delta' )
        self.phibar = phibar if phibar is not None else self._defaults('phibar')
        self.theta  = theta  if theta  is not None else self._defaults('theta' )
        self.M      = M if M is not None else 1.

    def _defaults(self, name):
        ''' Returns defaults for given parameter name '''
        return {
            'delta': {'x': 5e2, 't': 1e8, 'phi': 2e-1},
            'phibar': 0.,
            'theta' : {'a': 7.07e-6, 'b': 1.2e-4, 'k': 0.71}
        }[name]


class Laplacian():
    ''' Finite difference Laplacian with predefined shape for fast indexing '''
    def __init__(self, shape, delta, boundary='periodic'):
        self.shape = shape
        self.delta = delta
        self._boundary = 'periodic'
        self._create_indexset()

    def _create_indexset(self):
        self.ndims = len(self.shape)
        indexset = []
        for n in self.shape:
            if self._boundary is 'periodic':
                indexset.append(list(range(1,n)) + [0])
                indexset.append([n-1] + list(range(n-1)))
            elif self._boundary is 'mirror':
                indexset.append(list(range(1,n)) + [n-2])
                indexset.append([1] + list(range(n-1)))
            else:
                raise Exception('invalid boundary condition')
        self.indexset = indexset

    def __call__(self, x):
        ''' Calculates Laplacian of x '''
        return self.calculate(x)

    def calculate(self, x):
        ''' Approximates Laplacian of x '''
        lapx = - self.ndims * 2.0 * x
        for i,ix in enumerate(self.indexset):
            d = i//2
            perm     = [d, *list(range(d)), *list(range(d+1,self.ndims))]
            inv_perm = [*list(range(1,d+1)), 0, *list(range(d+1,self.ndims))]
            lapx += x.permute(*perm)[ix,...].permute(*inv_perm)
        return lapx / (self.delta**2)

class CahnHilliard():
    _LENGTH = 256


    def __init__(self, params=None, dim=2, seed=None, device='cpu'):

        self.params = Variables() if params is None else params
        self.ndims  = dim
        self.device = device if type(device) is tp.device else tp.device(device)
        self._initialise()
        self._phi0 = self.phi.clone()
        self.lapl_op = Laplacian(self._phi0.shape, self.params.delta['x'])

    def _initialise(self, seed=None):
        if seed is not None:
            rng_state = tp.random.get_rng_state().clone()
            tp.manual_seed(seed)

        delta  = self.params.delta['phi']
        phibar = self.params.phibar
        self.phi = phibar + tp.rand([self._LENGTH]*self.ndims, device=self.device)*delta
        self.mu  = tp.zeros([self._LENGTH]*self.ndims, device=self.device)

        self._flag_reinitialise = False

        if seed is not None:
            tp.random.set_rng_state(rng_state)

    def _diffuse_phi(self):
        ''' Finite difference update for ϕ
            dϕ = Mδt∇²μ
            ϕ+ = ϕ + dϕ
        '''
        M, dt = self.params.M, self.params.delta['t']

        nabla2_mu = self.lapl_op(self.mu)
        self.phi += M*dt*nabla2_mu

    def _calculate_mu(self):
        ''' Calculate μ = dF/dϕ
                μ = -αϕ + βϕ³ - κ∇²ϕ
        '''
        theta = self.params.theta

        a, b, k = theta['a'], theta['b'], theta['k']

        nabla2_phi = self.lapl_op(self.phi)
        self.mu = -a*self.phi + b*tp.pow(self.phi,3) - k*nabla2_phi

    def iterate(self, N=1):
        ''' Iterates evolution of ϕ in discrete time '''
        for _ in range(N):
            self._calculate_mu()
            self._diffuse_phi()

    def plot(self):
        ''' Plot contour of ϕ '''
        plt.contourf(self.numpy())
        plt.show()

    def numpy(self):
        ''' Returns current evaluation of ϕ as numpy array '''
        return self.phi.cpu().numpy()

    def scattering(self):
        ''' Get scattering plot for current ϕ '''

        F = tp.fft(complex_tensor(self.phi), self.ndims)
        F = fftshift(F)

        absF = tp.sqrt(F[...,0]**2 + F[...,1]**2)

        q, S = radial_avg(absF, self.params, device=self.device)
        return q, S

####### UTILITY
def complex_tensor(X):
    ''' converts real to complex '''
    ndims = len(X.shape)
    return tp.cat(
        [X[...,None], tp.zeros_like(X[...,None])],
        ndims
    )

def conjugate(X):
    ''' calculates the complex conjugate '''
    X[...,1] = -X[...,1]
    return X

def radial_avg(X, params, device=None):
    ''' [Assumes 2D] '''

    if device is None:
        device = tp.device('cpu')

    n = X.shape[0]

    xi, yi = np.indices((n,n))
    r = tp.tensor(
        np.sqrt((xi - 0.5*n)**2 + (yi - 0.5*n)**2).astype(np.int),
        device=device
    )

    qs = tp.linspace(
        -1/(2*params.delta['x']), 1/(2*params.delta['x']), n,
        device=device
    )
    qr = tp.tensor(
        [[tp.sqrt((qx**2 + qy**2)) for qx in qs] for qy in qs],
        device=device
    )

    q  = tp.bincount(r.view(-1), qr.view(-1)) / tp.bincount(r.view(-1))
    S  = tp.bincount(r.view(-1), X.view(-1)) / tp.bincount(r.view(-1))
    return q[1:], S[1:]

def fftshift(X):
    dims = X.shape

    ndims = len(dims)-1

    if ndims is 2:
        nx, ny = dims[0] //2 ,dims[1]//2
        Y = tp.zeros_like(X)
        Y[0:nx, 0:ny, :] = X[nx:,ny:,:]
        Y[0:nx, ny:,:]   = X[nx:,0:ny, :]
        Y[nx:,ny:,:]     = X[0:nx, 0:ny, :]
        Y[nx:,0:ny, :]   = X[0:nx, ny:,:]
    else:
        raise NotImplementedError
    return Y

def verify_numerics(device=None):
    ''' Utility code to confirm Laplacian is implemented correctly '''
    if device is 'gpu' or device is 'cuda':
        device = tp.device('cuda')
    elif device is 'cpu' or device is None:
        device = tp.device('cpu')
    else:
        raise Exception('invalid device')

    delta = 1/32

    xs = tp.arange(-2., 2., step=delta, device=device)

    test_f = lambda x: tp.exp(-x.T@x/0.5)/(3.141592/2.0)
    true_lapf = lambda x: (x.T@x - 0.5)*tp.exp(-x.T@x/0.5)/(2*3.141592*0.015625)

    vec = lambda x1,x2: tp.tensor([[x1],[x2]], device=device)

    F = tp.tensor([[test_f(vec(x1,x2)) for x1 in xs] for x2 in xs], device=device)
    lapF = tp.tensor([[true_lapf(vec(x1,x2)) for x1 in xs] for x2 in xs], device=device).cpu().numpy()

    lapl_op = Laplacian(F.shape, delta)

    num_lapF = lapl_op(F).cpu().numpy()

    maxs, mins = (np.max(num_lapF),np.max(lapF)),(np.min(num_lapF),np.min(lapF))
    diff    = num_lapF - lapF
    rel_err = (num_lapF / lapF) - 1.0

    print("        |    numeric  |   true\n")
    print("max val |   %7.4f   |  %7.4f\n" % maxs)
    print("min val |   %7.4f   |  %7.4f\n" %  mins)
    print("--------|    min      |  median    | max\n")
    print("abs dif |  %7.4f    |  %7.4f   | %7.4f\n" %
            (np.min(np.abs(diff)), np.median(np.abs(diff)), np.max(np.abs(diff)))
    )
    print("sqr dif |  %7.4f    |  %7.4f   | %7.4f\n" %
            (np.min(diff**2), np.median(diff**2), np.max(diff**2))
    )
    print("rel err |  %7.4f    |  %7.4f   | %7.4f\n" %
            (np.min(rel_err), np.median(rel_err), np.median(rel_err))
    )
