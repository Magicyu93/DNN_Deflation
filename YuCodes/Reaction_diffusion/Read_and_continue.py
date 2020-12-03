import torch
from torch import Tensor, optim
import numpy as  np
from numpy import array, arange, zeros, sum, sqrt, linspace, ones, absolute, meshgrid
from useful_tools import generate_uniform_points_in_cube, generate_uniform_points_on_cube,\
    generate_learning_rates, generate_deflation_alpha, generate_x_plot, generate_learning_rates_stage
import Network_React_Diff_general as network_file
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show, clf, pause, subplot, xlim, semilogy
import pickle
import time

from solution_RD_eq14 import res1, res2, res, res_bd, steady_state_sol, domain_shape, \
    domain_parameter, time_dependent_type, sol_plot


torch.set_default_tensor_type('torch.DoubleTensor')
prev_data_ID = ['12_2_19_50']
dir_text = './Results/'

# Read and load the corresponding network
data=pickle.load(open(dir_text + 'React_Diff_d_2_B_'+prev_data_ID[0]+'.data', 'rb'))

net_A = network_file.Network(data['d'],
                             data['m'],
                             activation_type = data['activation'],
                             boundary_control_type = data['boundary_control'],
                             initial_constant = data['initial_constant'],
                             structure_probing_num= data['structure_probing_num'])
net_A.load_state_dict(torch.load(dir_text + 'netApara_'+prev_data_ID[0]+'.pkl'))
net_A.c = data['net_A.c']
net_A.c.requires_grad = True

net_S = network_file.Network(data['d'],
                             data['m'],
                             activation_type = data['activation'],
                             boundary_control_type = data['boundary_control'],
                             initial_constant = data['initial_constant'],
                             structure_probing_num= data['structure_probing_num'])
net_S.load_state_dict(torch.load(dir_text + 'netSpara_'+prev_data_ID[0]+'.pkl'))
net_S.c = data['net_S.c']
net_S.c.requires_grad = True

# For the moment, only need the dimensional information
d = data['d']
domain_intervals = domain_parameter(d)

# drawing the plot from the data
x_plot = data['x_plot']
net_A_plot = data['net_A_plot']
net_S_plot = data['net_S_plot']

if d == 1:
    plt.clf()
    plt.figure(1)
    plt.plot(x_plot[:, 0], net_A_plot, 'r')
    plt.plot(x_plot[:, 0], net_S_plot, 'b')
    plt.legend(["Net_A", "Net_S"])
    plt.show()
pause(0.02)

# drawing the plot from the imported network
# sol_plot(net_A, net_S, x_plot)

# To finish the rest continue computing part