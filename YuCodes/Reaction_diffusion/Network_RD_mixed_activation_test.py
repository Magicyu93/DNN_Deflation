import torch
from torch import Tensor, optim
import numpy as  np
from numpy import array, arange, zeros, sum, sqrt, linspace, ones, absolute, meshgrid
from useful_tools import generate_uniform_points_in_cube, generate_uniform_points_on_cube,\
    generate_learning_rates, generate_deflation_alpha, generate_x_plot, generate_learning_rates_stage
import Network_React_Diff_general as network_file
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show, clf, pause, subplot, xlim, semilogy, contourf
import pickle
import time

from solution_RD_eq8 import res1, res2, res, res_bd, steady_state_sol, domain_shape, domain_parameter, \
    time_dependent_type, sol_plot

torch.set_default_tensor_type('torch.DoubleTensor')

# Set parameters #########################################
# training paramters
method = 'B'        # choose methods: B(basic), S(SelectNet), RS (reversed SelectNet)
m = 100        # number of nodes in each layer of solution network
n_epoch = 100000    # number of outer iterations
N_inside_train = 1000    # number of training sampling points inside the domain in each epoch (batch size)
N_inside_test = 1000        # number of test sampling points inside the domain
N_pts_deflation = 1000      # number of deflation sampling points inside the domain
n_update_each_batch = 1         # number of iterations in each epoch (for the same batch of points)

flag_IBC_in_loss = True
N_each_face_train = 1       # this should depends on the dimension of the problem...
lambda_term = 10000       # the boundary contribution for loss

# Network Setting ########################################
activation = 'mixed'        # activation function for the solution net
boundary_control = 'none'        # if the solution net architecture satisfies the boundary condition automatically
initial_constant = 'none'
structure_probing_num = 0

# Problem parameters #####################################
d = 1       # dimension of problem
domain_intervals = domain_parameter(d)

# Initializing the network #########################################
net_A = network_file.Network(d, m, activation_type=activation, boundary_control_type=boundary_control,
                             initial_constant=initial_constant, structure_probing_num=structure_probing_num)
net_S = network_file.Network(d, m, activation_type=activation, boundary_control_type=boundary_control,
                             initial_constant=initial_constant, structure_probing_num=structure_probing_num)

# Points
x1_train = generate_uniform_points_in_cube(domain_intervals, 1)
tensor_x1_train = torch.ones(x1_train.shape)
tensor_x1_train.requires_grad = False

# linear = torch.nn.Linear(1, m)
# y = linear(tensor_x1_train)
# test = network_file.mix(y)

# test1 = net_A(tensor_x1_train)
# print(test1)

N_plot_each_dim = 1001
x_plot = generate_x_plot(domain_intervals, N_plot_each_dim)
sol_plot(net_A, net_S, x_plot, 1)

print(1)