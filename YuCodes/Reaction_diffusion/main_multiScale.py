import torch
from torch import Tensor, optim
import numpy as  np
from numpy import array, arange, zeros, sum, sqrt, linspace, ones, absolute, meshgrid
from useful_tools import generate_uniform_points_in_cube, generate_uniform_points_on_cube,\
    generate_learning_rates, generate_deflation_alpha, generate_x_plot, generate_learning_rates_stage, \
    normal_init1, normal_init2, uniform_init, zero_init
import Network_React_Diff_general as network_file
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show, clf, pause, subplot, xlim, semilogy, contourf
import pickle
import time

from solution_RD_eq8 import res1, res2, res, res_bd, steady_state_sol, domain_shape, domain_parameter, \
    time_dependent_type, sol_plot, Mscale_evalueate, Mscale_res1, Mscale_res2, Mscale_res, Mscale_res_bd, \
    Mscale_sol_plot, Mscale_gradient

torch.set_default_tensor_type('torch.DoubleTensor')

# setting the directory where the solution will be saved
dir_text = './Results'
print('The goal for this code')

# Set parameters #########################################
# training paramters
method = 'B'        # choose methods: B(basic), S(SelectNet), RS (reversed SelectNet)
m = 50        # number of nodes in each layer of solution network
n_epoch = 100000    # number of outer iterations
N_inside_train = 1000    # number of training sampling points inside the domain in each epoch (batch size)
N_inside_test = 1000        # number of test sampling points inside the domain
N_pts_deflation = 1000      # number of deflation sampling points inside the domain
n_update_each_batch = 1         # number of iterations in each epoch (for the same batch of points)

flag_IBC_in_loss = True
N_each_face_train = 1       # this should depends on the dimension of the problem...
lambda_term = 100      # the boundary contribution for loss

# Network Setting ########################################
activation = 'ReLU3'        # activation function for the solution net
boundary_control = 'none'        # if the solution net architecture satisfies the boundary condition automatically
initial_constant = 'none'
structure_probing_num = 0
MscaleNN_num = 4

# Problem parameters #####################################
d = 1       # dimension of problem
domain_intervals = domain_parameter(d)

# Solution Parameter
rho = 0.01
D = 0.1
mu = 1

# Steady State solution
A_s, S_s = steady_state_sol(rho)

# deflation operator parameters
p = 3
alpha = generate_deflation_alpha(3, 1, n_epoch)

# Interface parameters #########################################
flag_compute_loss_each_epoch = True
n_epoch_show_info = 100
flag_show_sn_info = False
flag_show_plot = True
flag_output_results = True

# Print the parameters ########################################
print("d = %d, "
      "n_epoch = %d, "
      "N_inside_train = %d, "
      "N_each_face_train = %d, "
      "m = %d, "
      "p = %.1f, "
      "lambda_term = %.1f, " 
      "structure_probing_num = %d"
      % (d, n_epoch, N_inside_train, N_each_face_train, m, p, lambda_term, structure_probing_num)
      )

# Initializing the network #########################################
net_dict_A = {}
net_dict_S = {}
for i in range(MscaleNN_num):
    if i == 0:
        structure_probing = structure_probing_num
    else:
        structure_probing = 0
    net_dict_A[i] = network_file.Network(d, m, activation_type=activation, boundary_control_type=boundary_control,
                                         initial_constant = initial_constant, structure_probing_num=structure_probing_num)
    net_dict_A[i].apply(normal_init1)
    net_dict_S[i] = network_file.Network(d, m, activation_type=activation, boundary_control_type=boundary_control,
                                         initial_constant = initial_constant, structure_probing_num=structure_probing_num)
    net_dict_S[i].apply(normal_init1)
# Start ###########################################
# optimizer setting
# lrseq = generate_learning_rates(-1, -4, n_epoch)
lrseq = generate_learning_rates_stage(-1, -4, n_epoch, 0.995, 100)

params = []
for i in range(MscaleNN_num):
    params = params + list(net_dict_A[i].parameters())
    params = params + list(net_dict_S[i].parameters())
optimizer = optim.Adam(params,lr=lrseq[0])
if structure_probing_num != 0:
    optimizer.add_param_group({'params': net_dict_A[0].c, 'lr': 20*lrseq[0]})
    optimizer.add_param_group({'params': net_dict_S[0].c, 'lr': 20*lrseq[0]})

# Plot parameters ##########################################
N_plot_each_dim = 1001
x_plot = generate_x_plot(domain_intervals, N_plot_each_dim)

lossseq = zeros((n_epoch, ))
resseq1 = zeros((n_epoch, ))
resseq2 = zeros((n_epoch, ))
res_boundary = zeros((n_epoch, ))

# Training #########################################
k = 0
# # drawing the initial plot
Mscale_sol_plot(net_dict_A, net_dict_S, x_plot, 0)

while k < n_epoch:
    # sample the training points
    x1_train = generate_uniform_points_in_cube(domain_intervals, N_inside_train)
    tensor_x1_train = Tensor(x1_train)
    tensor_x1_train.requires_grad = False

    x1_test = generate_uniform_points_in_cube(domain_intervals, N_inside_test)
    tensor_x1_test = Tensor(x1_test)
    tensor_x1_test.requires_grad = False

    x_deflation = generate_uniform_points_in_cube(domain_intervals, N_pts_deflation)
    tensor_x_deflation = Tensor(x_deflation)
    tensor_x_deflation.requires_grad = False

    if flag_IBC_in_loss:
        x2_train = generate_uniform_points_on_cube(domain_intervals, N_each_face_train)
        tensor_x2_train = Tensor(x2_train)
        tensor_x2_train.requires_grad = False

    # set learning rate
    params = []
    for i in range(MscaleNN_num):
        params = params + list(net_dict_A[i].parameters())
        params = params + list(net_dict_S[i].parameters())
    optimizer = optim.Adam(params, lr=lrseq[k])
    if structure_probing_num != 0:
        optimizer.add_param_group({'params': net_dict_A[0].c, 'lr': 10 * lrseq[k]})
        optimizer.add_param_group({'params': net_dict_S[0].c, 'lr': 10 * lrseq[k]})

    for i_update in range(n_update_each_batch):
        # compute the deflation term
        ratio = 5

        deflation_l2_A = 1 / ((torch.sum((Mscale_evalueate(net_dict_A, tensor_x1_train) - A_s) ** 2))
                            /tensor_x1_train.shape[0])**(p/2)
        deflation_h1_A = 1 / (torch.sum((Mscale_gradient(net_dict_A, tensor_x1_train) - 0) ** 2)
                              / tensor_x1_train.shape[0]) ** (p / 2)

        deflation_l2_S = 1 / ((torch.sum((Mscale_evalueate(net_dict_S, tensor_x1_train) - S_s) ** 2))
                            /tensor_x1_train.shape[0])**(p/2)
        deflation_h1_S = 1 / (torch.sum((Mscale_gradient(net_dict_S, tensor_x1_train) - 0) ** 2)
                              / tensor_x1_train.shape[0]) ** (p / 2)

        # deflation = deflation_l2_A  + deflation_l2_S
        deflation = deflation_l2_A + deflation_l2_S + ratio * (deflation_h1_A + deflation_h1_S)

        # compute the loss
        # residual1 = res1(net_A, net_S, tensor_x1_train)
        # residual2 = res2(net_A, net_S, tensor_x1_train)
        residual_sq1 = 1/N_inside_train * torch.sum(Mscale_res(net_dict_A, net_dict_S, tensor_x1_train) **2)

        # test = res_bd(net_A, tensor_x2_train)
        residual_bd_A = 1/tensor_x2_train.shape[0] * torch.sum(Mscale_res_bd(net_dict_A, tensor_x2_train) ** 2)
        residual_bd_S = 1/tensor_x2_train.shape[0] * torch.sum(Mscale_res_bd(net_dict_S, tensor_x2_train) ** 2)

        residual_sq2 = residual_bd_A + residual_bd_S

        loss = (deflation + alpha[k]) * (residual_sq1 + lambda_term * residual_sq2)
        # loss = residual_sq1 + lambda_term * residual_sq2

        # update the network
        optimizer.zero_grad()
        loss.backward(retain_graph=not flag_compute_loss_each_epoch)
        optimizer.step()

    # save loss and l2 error
    lossseq[k] = loss.item()
    resseq1[k] = np.sqrt(1 / N_inside_train * torch.sum(Mscale_res1(net_dict_A, net_dict_S, tensor_x1_train) ** 2).detach().numpy())
    resseq2[k] = np.sqrt(1 / N_inside_train * torch.sum(Mscale_res2(net_dict_A, net_dict_S, tensor_x1_train) ** 2).detach().numpy())
    res_boundary[k] = np.sqrt(1/N_each_face_train *
                    (torch.sum(Mscale_res_bd(net_dict_A, tensor_x2_train) ** 2) + torch.sum(Mscale_res_bd(net_dict_S, tensor_x2_train)**2)).detach().numpy())

    # show information
    if k % n_epoch_show_info == 0:
        if flag_show_plot & i_update%10 == 0:
            # drawing the plot
            Mscale_sol_plot(net_dict_A, net_dict_S, x_plot, k)

        if flag_compute_loss_each_epoch:
            print("epoch = %d, "
                  "deflation_term = %2.5f, "
                  "loss = %2.5f, "
                  "residual1 = %2.5f, "
                  "residual2 = %2.5f, " 
                  "res_boundary = %2.5f, " 
                  "lr = %.5f"
                  % (k, deflation, lossseq[k], resseq1[k], resseq2[k], res_boundary[k], lrseq[k]))
        else:
            print("epoch = %d" % k)
        print("\n")

    # update to next step
    k = k + 1

# compute plot at x_plot
net_A_plot = Mscale_evalueate(net_dict_A, x_plot).detach().numpy()
net_S_plot = Mscale_evalueate(net_dict_S, x_plot).detach().numpy()

# Save net for deflation
localtime = time.localtime(time.time())
time_text = str(localtime.tm_mon)+'_'+str(localtime.tm_mday)+'_'+str(localtime.tm_hour)+'_'+str(localtime.tm_min)
for i in range(MscaleNN_num):
    torch.save(net_dict_A[i].state_dict(), dir_text + 'MscaleNN' + str(i) + 'netApara_'+time_text+'.pkl')
    torch.save(net_dict_S[i].state_dict(), dir_text + 'MscaleNN' + str(i) + 'netSpara_'+time_text+'.pkl')

# Output results
if flag_output_results == True:
    # save the data
    main_file_name = 'main_React_Diff'
    data = {'main_file_name': main_file_name,
            'ID': time_text,
            # important solution information
            'net_A.c': net_dict_A[0].c,
            'net_S.c': net_dict_S[0].c,
            # network information
            'method': method,
            'm': m,
            'activation': activation,
            'boundary_control': boundary_control,
            'initial_constant': initial_constant,
            'structure_probing_num': structure_probing_num,
            'MscaleNN_num': MscaleNN_num,
            # problem information
            'time_dependent_type': time_dependent_type,
            'd': d,
            'domain_shape': domain_shape,
            # training information
            'N_inside_train': N_inside_train,
            'flag_IBC_in_loss': flag_IBC_in_loss,
            'N_each_face_train': N_each_face_train,
            'lambda_term': lambda_term,
            'n_epoch': n_epoch,
            'n_update_each_batch': n_update_each_batch,
            # residual and loss information
            'lossseq': lossseq,
            'lrseq': lrseq,
            'resseq1': resseq2,
            'resseq2': resseq2,
            # plot information
            'net_A_plot': net_A_plot,
            'net_S_plot': net_S_plot,
            'N_plot_each_dim': N_plot_each_dim,
            'x_plot': x_plot,
            }
    filename = dir_text + 'React_Diff_d_' + str(d) + '_' + method + '_' + time_text + '.data'
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()