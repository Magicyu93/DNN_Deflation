import torch
from torch import Tensor, optim
import numpy as  np
from numpy import array, arange, zeros, sum, sqrt, linspace, ones, absolute, meshgrid
from useful_tools import generate_uniform_points_in_cube, generate_uniform_points_on_cube,\
    generate_learning_rates, generate_deflation_alpha, generate_x_plot
import Network_React_Diff_general as network_file
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show, clf, pause, subplot, xlim, semilogy, contourf
import pickle
import time

from solution_RD_eq14 import res1, res2, res, res_bd, steady_state_sol, domain_shape, domain_parameter, time_dependent_type

torch.set_default_tensor_type('torch.DoubleTensor')

# Set parameters #########################################
method = 'B'        # choose methods: B(basic), S(SelectNet), RS (reversed SelectNet)
d = 2       # dimension of problem
m = 100        # number of nodes in each layer of solution network
n_epoch = 50000     # number of outer iterations
N_inside_train = 1000    # number of training sampling points inside the domain in each epoch (batch size)
N_inside_test = 1000        # number of test sampling points inside the domain
N_pts_deflation = 1000      # number of deflation sampling points inside the domain
n_update_each_batch = 1         # number of iterations in each epoch (for the same batch of points)
lrseq = generate_learning_rates(-1, -4, n_epoch)

# the boundary contribution for loss
lambda_term = 100

# deflation operator parameters
p = 3
alpha = generate_deflation_alpha(3, 1, n_epoch)

# Solution Parameter #########################################
rho = 0.01
D = 0.1
mu = 1

# Steady State solution########################################
A_s, S_s = steady_state_sol(rho)

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
N_each_face_train = 1000

# Start ###########################################
optimizer = optim.Adam(net_A.parameters(), lr = lrseq[0])
optimizer.add_param_group({'params': net_S.parameters()})

losssq = zeros((n_epoch, ))

resseq1 = zeros((n_epoch, ))
resseq2 = zeros((n_epoch, ))

# giving the x_plot, grid_x, grid_y
N_plot_each_dim = 101
x_plot = generate_x_plot(domain_intervals, N_plot_each_dim)
grid_x, grid_y = np.meshgrid(np.linspace(domain_intervals[0, 0], domain_intervals[0, 1], N_plot_each_dim),
                             np.linspace(domain_intervals[1, 0], domain_intervals[1, 1], N_plot_each_dim))

a_plot = np.reshape(net_A(torch.tensor(x_plot)).detach().numpy(), (N_plot_each_dim, N_plot_each_dim))
s_plot = np.reshape(net_S(torch.tensor(x_plot)).detach().numpy(), (N_plot_each_dim, N_plot_each_dim))

fig_a, ax_a = plt.subplots()
contour_a = ax_a.contourf(grid_x, grid_y, a_plot)
c_bar_a = fig_a.colorbar(contour_a)
ax_a.set_title("plot for A")

fig_s, ax_s = plt.subplots()
contour_s = ax_s.contourf(grid_x, grid_y, s_plot)
c_bar_s = fig_s.colorbar(contour_s)
ax_s.set_title("plot for S")

plt.show(block=False)
pause(0.01)

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
    for param_group in optimizer.param_groups:
        param_group['lr'] = lrseq[k]

    for i_update in range(n_update_each_batch):
        # compute the deflation term
        deflation_A = 1 / (torch.sum((net_A(tensor_x_deflation) - A_s) ** 2) / tensor_x_deflation.shape[0]) ** (
                    p / 2)
        deflation_S = 1 / (torch.sum((net_S(tensor_x_deflation) - S_s) ** 2) / tensor_x_deflation.shape[0]) ** (
                    p / 2)
        deflation = 1 / (1 / deflation_A + 1 / deflation_S)
        # deflation = deflation_A + deflation_S

        # compute the loss
        residual_sq1 = 1 / N_inside_train * torch.sum(res(net_A, net_S, tensor_x1_train) ** 2)

        residual_bd_A = 1/N_each_face_train * torch.sum(res_bd(net_A, tensor_x2_train) ** 2)
        residual_bd_S = 1/N_each_face_train * torch.sum(res_bd(net_S, tensor_x2_train) ** 2)
        residual_sq2 = residual_bd_A + residual_bd_S

        loss = (deflation + alpha[k]) * (residual_sq1 + lambda_term * residual_sq2)

        # update the network
        optimizer.zero_grad()
        loss.backward(retain_graph=not flag_compute_loss_each_epoch)
        optimizer.step()

    # save loss and l2 error
    losssq[k] = loss.item()
    resseq1[k] = np.sqrt(1 / N_inside_train * torch.sum(res1(net_A, net_S, tensor_x1_train) ** 2).detach().numpy())
    resseq2[k] = np.sqrt(1 / N_inside_train * torch.sum(res2(net_A, net_S, tensor_x1_train) ** 2).detach().numpy())

    # show information
    if k % n_epoch_show_info == 0:
        if flag_show_plot & i_update%10 == 0:
            # Plot the slice for xd
            a_plot = np.reshape(net_A(torch.tensor(x_plot)).detach().numpy(), (N_plot_each_dim, N_plot_each_dim))
            s_plot = np.reshape(net_S(torch.tensor(x_plot)).detach().numpy(), (N_plot_each_dim, N_plot_each_dim))

            plt.close(fig_a)
            plt.close(fig_s)

            fig_a, ax_a = plt.subplots()
            contour_a = ax_a.contourf(grid_x, grid_y, a_plot)
            c_bar_a = fig_a.colorbar(contour_a)
            ax_a.set_title("plot for A")

            fig_s, ax_s = plt.subplots()
            contour_s = ax_s.contourf(grid_x, grid_y, s_plot)
            c_bar_s = fig_s.colorbar(contour_s)
            ax_s.set_title("plot for S")

            plt.show(block=False)
            pause(0.02)

        if flag_compute_loss_each_epoch:
            print(
                "epoch = %d, deflation_term = %2.5f, loss = %2.5f, residual1 = %2.5f, residual2 = %2.5f" % (
                k, deflation, loss.item(), resseq1[k], resseq2[k]), end='')
        else:
            print("epoch = %d" % k, end='')
        print("\n")

    # update to next step
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
    data = {'main_file_name': main_file_name,
            'ID': time_text,
            'N_inside_train': N_inside_train,
            'N_plot': N_plot,
            'activation': activation,
            'boundary_control': boundary_control,
            'd': d,
            'domain_shape': domain_shape,
            'initial_constant': initial_constant,
            'lambda_term': lambda_term,
            'lossseq': losssq,
            'lrseq': lrseq,
            'm': m,
            'method': method,
            'n_epoch': n_epoch,
            'n_update_each_batch': n_update_each_batch,
            'resseq1': resseq2,
            'resseq2': resseq2,
            'time_dependent_type': time_dependent_type,
            'net_A_plot': net_A_plot,
            'net_S_plot': net_S_plot,
            'x_plot': x_plot,
            }
    filename = 'React_Diff_d_' + str(d) + '_' + method + '_' + time_text + '.data'
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()