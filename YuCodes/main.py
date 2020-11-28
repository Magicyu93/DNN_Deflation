import torch
from torch import Tensor, optim
import numpy as  np
from numpy import array, arange, zeros, sum, sqrt, linspace, ones, absolute, meshgrid
from useful_tools import generate_uniform_points_in_cube, generate_uniform_points_on_cube,\
    generate_learning_rates, generate_deflation_alpha
import Network_React_Diff_1D as network_file
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show, clf, pause, subplot, xlim, semilogy
import pickle
import time

from solution_React_Diff_1D import res1, res2, res, res_bd, domain_shape, domain_parameter, time_dependent_type, time_interval

# torch.set_default_tensor_type('torch.cuda.DoubleTensor')
torch.set_default_tensor_type('torch.DoubleTensor')

# Set parameters #########################################

method = 'B'        # choose methods: B(basic), S(SelectNet), RS (reversed SelectNet)
d = 1       # dimension of problem
m = 100        # number of nodes in each layer of solution network
n_epoch = 50000     # number of outer iterations
N_inside_train = 1000     # number of trainning sampling points inside the domain in each epoch (batch size)
N_inside_test = 1000        # number of test sampling points inside the domain
N_pts_deflation = 1000      # number of deflation sampling points inside the domain
n_update_each_batch = 1         # number of iterations in each epoch (for the same batch of points)
lrseq = generate_learning_rates(-1.5, -4, n_epoch)

# the boundary contribution for loss
lambda_term = 1

# deflation operator parameters
p = 3
alpha = generate_deflation_alpha(3, 0, n_epoch)

# Solution Parameter #########################################
rho = 0.01
D = 0.1
mu = 1

# Steady State solution########################################
A_s = 1+rho
S_s = (1+rho)**(-2)

left = 0
right = 10
scale = 10
right_scaled = 1

# Network Setting ########################################
activation = 'ReLU3'        # activation function for the solution net
boundary_control = 'none'        # if the solution net architecture satisfies the boundary condition automatically
initial_constant = 'none'

# Problem parameters   #
domain_intervals = domain_parameter(d)

# Interface parameters #########################################
flag_compute_loss_each_epoch = True
n_epoch_show_info = 100
flag_show_sn_info = False
flag_show_plot = True
flag_output_results = True

# Depending parameters #########################################
net_A = network_file.Network(d, m, activation_type=activation, boundary_control_type=boundary_control,
                             initial_constant=initial_constant)
net_S = network_file.Network(d, m, activation_type=activation, boundary_control_type=boundary_control,
                             initial_constant=initial_constant)

flag_IBC_in_loss = True
N_each_face_train = 1

# Start ###########################################
optimizer_A = optim.Adam(net_A.parameters(), lr=lrseq[0])
optimizer_S = optim.Adam(net_S.parameters(), lr=lrseq[0])

losssq = zeros((n_epoch, ))

resseq1 = zeros((n_epoch, ))
resseq2 = zeros((n_epoch, ))

N_plot = 1001
x_plot = np.zeros((N_plot, d))
x_plot[:,0] = np.linspace(domain_intervals[0, 0], domain_intervals[0, 1], N_plot)

# Training #########################################
k = 0
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
    for param_group in optimizer_A.param_groups:
        param_group['lr'] = lrseq[k]
    for param_group in optimizer_S.param_groups:
        param_group['lr'] = lrseq[k]


    for i_update in range(n_update_each_batch):
        # # compute the deflation term
        # deflation_A = 1 / (torch.sum((net_A(tensor_x_deflation) - A_s)**2)/tensor_x_deflation.shape[0])**(p/2)
        # deflation_S = 1 / (torch.sum((net_S(tensor_x_deflation) - S_s)**2)/tensor_x_deflation.shape[0])**(p/2)
        # deflation = 1/(1/deflation_A + 1/deflation_S)
        #
        # # compute the loss
        # residual_sq1 = 1/N_inside_train * torch.sum(res(net_A, net_S, tensor_x1_train)**2)
        #
        # residual_bd_A = torch.sum(res_bd(net_A, tensor_x2_train)**2)
        # residual_bd_S = torch.sum(res_bd(net_S, tensor_x2_train)**2)
        # residual_sq2 = residual_bd_A + residual_bd_S
        #
        # loss = (deflation + alpha[k]) * (residual_sq1 + lambda_term * residual_sq2)
        # # loss = residual_sq1 + lambda_term * residual_sq2      # used for computing the homogeneous solution
        #
        # # update the network
        # optimizer_A.zero_grad()
        # optimizer_S.zero_grad()
        # loss.backward(retain_graph=not flag_compute_loss_each_epoch)
        # optimizer_A.step()
        # optimizer_S.step()

        # trying another way of doing optimization
        # updating based on A equation loss
        deflation_A = 1 / (torch.sum((net_A(tensor_x_deflation) - A_s)**2)/tensor_x_deflation.shape[0])**(p/2)
        residual_sq1_A = 1/N_inside_train * torch.sum(res1(net_A, net_S, tensor_x1_train)**2)
        residual_bd_A = torch.sum(res_bd(net_A, tensor_x2_train) ** 2)
        loss1 = (deflation_A + alpha[k]) * (residual_sq1_A + lambda_term * residual_bd_A)

        optimizer_A.zero_grad()
        optimizer_S.zero_grad()
        loss1.backward(retain_graph=not flag_compute_loss_each_epoch)
        optimizer_A.step()
        optimizer_S.step()

        # updating based on S equation loss
        deflation_S = 1 / (torch.sum((net_S(tensor_x_deflation) - S_s) ** 2) / tensor_x_deflation.shape[0]) ** (p / 2)
        residual_sq1_S = 1 / N_inside_train * torch.sum(res2(net_A, net_S, tensor_x1_train) ** 2)
        residual_bd_S = torch.sum(res_bd(net_S, tensor_x2_train) ** 2)
        loss2 = (deflation_S + alpha[k]) * (residual_sq1_S + lambda_term * residual_bd_S)
        optimizer_A.zero_grad()
        optimizer_S.zero_grad()
        loss2.backward(retain_graph=not flag_compute_loss_each_epoch)
        optimizer_A.step()
        optimizer_S.step()

        deflation_A = 1 / (torch.sum((net_A(tensor_x_deflation) - A_s)**2)/tensor_x_deflation.shape[0])**(p/2)
        deflation_S = 1 / (torch.sum((net_S(tensor_x_deflation) - S_s)**2)/tensor_x_deflation.shape[0])**(p/2)
        deflation = 1/(1/deflation_A + 1/deflation_S)

        # compute the loss
        residual_sq1 = 1/N_inside_train * torch.sum(res(net_A, net_S, tensor_x1_train)**2)

        residual_bd_A = torch.sum(res_bd(net_A, tensor_x2_train)**2)
        residual_bd_S = torch.sum(res_bd(net_S, tensor_x2_train)**2)
        residual_sq2 = residual_bd_A + residual_bd_S

        loss = (deflation + alpha[k]) * (residual_sq1 + lambda_term * residual_sq2)
        # loss = residual_sq1 + lambda_term * residual_sq2      # used for computing the homogeneous solution



    # save loss and l2 error
    losssq[k] = loss.item()
    resseq1[k] = np.sqrt(1/N_inside_train * torch.sum(res1(net_A, net_S, tensor_x1_train)**2).detach().numpy())
    resseq2[k] = np.sqrt(1/N_inside_train * torch.sum(res2(net_A, net_S, tensor_x1_train)**2).detach().numpy())


    # show information
    if k % n_epoch_show_info == 0:
        if flag_show_plot & i_update%10 == 0:
            # Plot the slice for xd
            clf()

            plt.figure(1)
            plt.plot(x_plot[:, 0], net_A(torch.tensor(x_plot)).detach().numpy(), 'r')
            plt.plot(x_plot[:, 0], net_S(torch.tensor(x_plot)).detach().numpy(), 'b')
            plt.legend(["Net_A", "Net_S"])

            # plt.figure(2)
            # plt.plot(x_plot[:, 0], res1(net_A, net_S, torch.tensor(x_plot)).detach().numpy(), 'r')
            # plt.plot(x_plot[:, 0], res2(net_A, net_S, torch.tensor(x_plot)).detach().numpy(), 'b')
            # plt.legend(["Net_A", "Net_S"])

            plt.show(block=False)

            pause(0.02)

        if flag_compute_loss_each_epoch:
            print(
                "epoch = %d, deflation_term = %2.5f, loss = %2.5f, residual1 = %2.5f, residual2 = %2.5f" % (
                k, deflation, loss.item(), resseq1[k], resseq2[k]), end='')
        else:
            print("epoch = %d" % k, end='')
        print("\n")

    # update to next
    k = k + 1


# compute plot at x_plot
net_A_plot = net_A.predict(x_plot)
net_S_plot = net_S.predict(x_plot)

# Save net for deflation
localtime = time.localtime(time.time())
time_text = str(localtime.tm_mon)+'_'+str(localtime.tm_mday)+'_'+str(localtime.tm_hour)+'_'+str(localtime.tm_min)

# Output results
if flag_output_results == True:
    # compute the plot
    N_plot = 201

    net_A_plot = net_A.predict(x_plot)
    net_S_plot = net_S.predict(x_plot)

    # save the data
    main_file_name = 'main_React_Diff'
    data = {'main_file_name': main_file_name, \
            'ID': time_text, \
            'N_inside_train': N_inside_train, \
            'N_plot': N_plot, \
            'activation': activation, \
            'boundary_control': boundary_control, \
            'd': d, \
            'domain_shape': domain_shape, \
            'initial_constant': initial_constant, \
            'lambda_term': lambda_term, \
            'lossseq': losssq, \
            'lrseq': lrseq, \
            'm': m, \
            'method': method, \
            'n_epoch': n_epoch, \
            'n_update_each_batch': n_update_each_batch, \
            'resseq1': resseq2, \
            'resseq2': resseq2, \
            'time_dependent_type': time_dependent_type, \
            'net_A_plot': net_A_plot, \
            'net_S_plot': net_S_plot, \
            'x_plot': x_plot, \
            }
    filename = 'React_Diff_d_' + str(d) + '_' + method + '_' + time_text + '.data'
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()