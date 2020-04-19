import unittest
import numpy as np
import matplotlib.pylab as plt
import autograd.numpy as npa
import sys
sys.path.append('../')
from ceviche import fdfd_hz, fdfd_ez, jacobian
from ceviche.utils import imarr
import scipy.sparse.linalg as spl
import scipy.sparse as sp
from ceviche.utils import make_sparse
from ceviche.fdfd import compute_derivative_matrices
import autograd.numpy as npa
from ceviche.constants import *
# make parameters
omega = 2 * np.pi * 200e12*1.5/1.834  # lambda = 1.843 um
dL = 2e-8
def solve_mode_hz(eps_vec):
    N_shape = eps_vec.shape
    N = np.prod(N_shape)
    eps_vec = np.reshape(eps_vec, [N, 1])
    npml = 0
    Dxf, Dxb, Dyf, Dyb = compute_derivative_matrices(omega, (N, 1), [npml, 0], dL=dL)
    eps_vec_xx = eps_vec
    eps_vec_yy = 1 / 2 * (eps_vec + npa.roll(eps_vec, axis=0, shift=1))
    eps_vec_xx_inv = 1/(eps_vec_xx)
    eps_vec_yy_inv = 1/(eps_vec_yy)
    eps_vec_xx_inv = sp.diags(np.squeeze(eps_vec_xx_inv), shape=[N, N])
    eps_vec_yy_inv = sp.diags(np.squeeze(eps_vec_yy_inv), shape=[N, N])
    dxepsydx = (Dxf.dot(eps_vec_yy_inv)).dot(Dxb)
    dyepsxdy = (Dyf.dot(eps_vec_xx_inv)).dot(Dyb)
    A = sp.diags(np.squeeze(eps_r_input)).dot((dxepsydx)*(C_0/omega)**2)+sp.diags(np.squeeze(eps_r_input))
    guess_value = eps_vec.max()
    vals, vecs = spl.eigs(A, sigma=12, v0=None, which='LM')
    return vals, vecs
eps_r_input=np.zeros((1, 250))
eps_r_input[0, :50]=1.44**2
eps_r_input[0, 50:-50]=3.477**2
eps_r_input[0, -50:]=1.44**2
vals, vecs = solve_mode_hz(eps_r_input)
