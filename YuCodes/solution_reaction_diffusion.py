# the 1D solution of the reaction diffusion system 

# assuming the model to be dim N * (2 * N)
# data(i,j) = [A(ti,xj), S(ti, xj)]

import torch 
from numpy import array, prod, sum, zeros, exp, max, log, sqrt
from torch.nn.functional import relu
from torch.nn import Softplus

torch.set_default_tensor_type('torch.DoubleTensor')
h = 0.001
left = 0
right = 10

# constants needs to be sampled 
# to be fixed later 
rho = 0.5
mu = 1.5
D = 0.0002

# the point-wise Du: Input x is a sampling point of d variables ; Output is a numpy vector which means the result of Du(x))
def Du(model, x_batch):

	N = x_batch.shape[1]

	data = model(x_batch)
	A = data[:,0]
	S = data[:,1]

	# laplace term
	laplace = (model(tensor_x_batch+h*ei)-2*model(tensor_x_batch)+model(tensor_x_batch-h*ei))/h/h
	laplace_A = laplace[:,0]
	laplace_S = laplace[:,1]

	# rhs
	rhs1 = D * laplace_A + S * (A ** 2) - A + rho 
	rhs2 = laplace_S + mu * (1 - S * (A ** 2) )

	# t-derivative 
