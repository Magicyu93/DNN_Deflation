# testing the deflation code using same point calculating the deflation and loss

import torch
from torch import Tensor, optim
from numpy import array, arange, zeros, sum, sqrt, linspace, ones, absolute, meshgrid
from useful_tools import generate_uniform_points_in_cube, generate_uniform_points_in_cube_time_dependent,\
    generate_uniform_points_on_cube, generate_uniform_points_on_cube_time_dependent,\
    generate_uniform_points_in_sphere, generate_uniform_points_in_sphere_time_dependent,\
    generate_uniform_points_on_sphere, generate_uniform_points_on_sphere_time_dependent,\
    generate_learning_rates
import network_3_Painleve as network_file
import network_3_Painleve as deflation_net_file
# from selection_network_setting import selection_network
from matplotlib.pyplot import plot, show, clf, pause, subplot, xlim, semilogy
import time
import pickle

from solution_Painleve import Du, Du_ft, Bu_ft, f, g, h0, h1, domain_shape, domain_parameter, time_dependent_type, time_interval

torch.set_default_tensor_type('torch.DoubleTensor')

########### Set parameters #############
method = 'B' # choose methods: B(basic), S(SelectNet), RS (reversed SelectNet)
d = 1  # dimension of problem
m = 100  # number of nodes in each layer of solution network
n_epoch = 100000  # number of outer iterations
N_inside_train = 1000 # number of trainning sampling points inside the domain in each epoch (batch size)
n_update_each_batch = 1 # number of iterations in each epoch (for the same batch of points)
lrseq = generate_learning_rates(-1,-2,n_epoch,0.995,1000)

alpha = 1
p = 2
N_pts_deflation = N_inside_train

activation = 'ReLU3'  # activation function for the solution net
boundary_control = 'Dirichlet'  # if tcd he solution net architecture satisfies the boundary condition automatically 
initial_constant = 1.0
x1_train_distribution = 'uniform'

########### Problem parameters   #############
time_dependent_type = time_dependent_type()   ## If this is a time-dependent problem
domain_shape = domain_shape()  ## the shape of domain 
domain_intervals = domain_parameter(d)  

########### Interface parameters #############
flag_compute_loss_each_epoch = True
n_epoch_show_info = 100
flag_show_sn_info = False
flag_show_plot = True
flag_output_results = True

########### Depending parameters #############
u_net = network_file.network(d,m, activation_type = activation, boundary_control_type = boundary_control, initial_constant = initial_constant)

#################### Start ######################
# not sure about the meaning of this file ID ... 
dir = '../Results/'
deflation_file_ID = ['11_17_1_39'] #'zero' must be the first

data=pickle.load(open(dir + 'Painleve_d_1_B_'+deflation_file_ID[0]+'.data', 'rb'))
net_deflation_1 = deflation_net_file.network(data['d'],data['m'], activation_type = data['activation'], boundary_control_type = data['boundary_control'], initial_constant = data['initial_constant'])
net_deflation_1.load_state_dict(torch.load(dir + 'networkpara_'+deflation_file_ID[0]+'.pkl'))

optimizer = optim.Adam(u_net.parameters(),lr=lrseq[0])
lossseq = zeros((n_epoch,))
resseq = zeros((n_epoch,))

N_plot = 1001
x_plot = zeros((N_plot,d))
x_plot[:,0] = linspace(domain_intervals[0,0],domain_intervals[0,1],N_plot)

N_plot = 1001
x_plot = zeros((N_plot,d))
x_plot[:,0] = linspace(domain_intervals[0,0],domain_intervals[0,1],N_plot)

clf()
plot(x_plot[:,0],u_net.predict(x_plot),'r')
plot(x_plot[:,0],net_deflation_1.predict(x_plot),'b')
show()
pause(5.0)

# Training 
k = 0

while k < n_epoch:
    
    # sampling the training points, with uniform distribution 
    x1_train = generate_uniform_points_in_cube(domain_intervals,N_inside_train)
    
    tensor_x1_train = Tensor(x1_train)
    tensor_x1_train.requires_grad = False
    
    tensor_f1_train = Tensor(f(x1_train))
    tensor_f1_train.requires_grad = False
    
    # compute the deflation using same sampling as u_net
    x1_deflation = generate_uniform_points_in_cube(domain_intervals,N_inside_train)
#     x1_deflation = x1_train
    tensor_x1_deflation = Tensor(x1_deflation) 
    tensor_x1_deflation.requires_grad = False

    # Set learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lrseq[k]
        
    # train the network 
    for i_update in range(n_update_each_batch):
        
        # residue 
        residual_sq = 1/N_inside_train * torch.sum( Du_ft(u_net, tensor_x1_train)**2 )
        
        #compute the deflation term
        deflation_term = 1/( 1/N_inside_train * torch.sum( (u_net(tensor_x1_deflation) - net_deflation_1(tensor_x1_deflation))**2 ) )**(p/2)
        
        deflation_term = deflation_term + alpha 
        
        loss = deflation_term * residual_sq 
        
        ## Update the network
        optimizer.zero_grad()
        loss.backward(retain_graph=not flag_compute_loss_each_epoch)
        optimizer.step()
        
    #save loss and l2 error 
    lossseq[k] = loss.item()
    resseq[k] = sqrt( 1/N_inside_train * sum(Du(u_net, x1_train)**2) )
    
    #show information 
    if k%n_epoch_show_info==0:
        if flag_show_plot == True:
            # Plot the slice for xd
            clf()
            plot(x_plot[:,0],u_net.predict(x_plot),'r')
            plot(x_plot[:,0],net_deflation_1.predict(x_plot),'b')
            show()
            pause(0.02)
        if flag_compute_loss_each_epoch:
            print("epoch = %d, loss = %2.5f, deflation_term = %2.5f, residual = %2.5f" %(k,loss.item(),deflation_term.item(),resseq[k]), end='')
        else:
            print("epoch = %d" % k, end='')
        print("\n")
        
    k = k + 1 
    
# compute u_plot at x_plot
u_net_plot = u_net.predict(x_plot)

# Save u_net at x for deflation
localtime = time.localtime(time.time())
time_text = str(localtime.tm_mon)+'_'+str(localtime.tm_mday)+'_'+str(localtime.tm_hour)+'_'+str(localtime.tm_min)
torch.save(u_net.state_dict(),'networkpara_'+time_text+'.pkl')


