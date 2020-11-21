import torch
from torch import Tensor, optim
from numpy import array, arange, zeros, sum, sqrt, linspace, ones, absolute, meshgrid
from useful_tools import generate_uniform_points_in_cube, generate_uniform_points_on_cube,\
    generate_learning_rates
import network_3_Painleve as network_file
# from selection_network_setting import selection_network
from matplotlib.pyplot import plot, show, clf, pause, subplot, xlim, semilogy
import pickle
import time

from solution_Painleve import Du, Du_ft, Bu_ft, f, g, h0, h1, domain_shape, domain_parameter, time_dependent_type, time_interval

torch.set_default_tensor_type('torch.DoubleTensor')

########### Set parameters #############
method = 'B' # choose methods: B(basic), S(SelectNet), RS (reversed SelectNet)
d = 1  # dimension of problem
m = 100  # number of nodes in each layer of solution network
n_epoch = 10000  # number of outer iterations
N_inside_train = 1000 # number of trainning sampling points inside the domain in each epoch (batch size)
n_update_each_batch = 1 # number of iterations in each epoch (for the same batch of points)
lrseq = generate_learning_rates(-2,-3,n_epoch,0.995,1000)

# Is this the lambda in the NN? 
lambda_term = 10

activation = 'ReLU3'  # activation function for the solution net
boundary_control = 'Dirichlet'  # if the solution net architecture satisfies the boundary condition automatically 
flag_preiteration_by_small_lr = False  # If pre iteration by small learning rates
# lr_pre = 1e-4
# n_update_each_batch_pre = 100
# h_Du_t = 0.01  # time length for computing the first derivative of t by finite difference (for the hyperbolic equations)
# flag_reset_select_net_each_epoch = False  # if reset selection net for each outer iteration
# selectnet_initial_constant = 1  # if selectnet is initialized as constant one
initial_constant = 'none'

########### Problem parameters   #############
time_dependent_type = time_dependent_type()   ## If this is a time-dependent problem
domain_shape = domain_shape()  ## the shape of domain 
if domain_shape == 'cube':  
    domain_intervals = domain_parameter(d)
elif domain_shape == 'sphere':
    R = domain_parameter(d)
    
# if not time_dependent_type == 'none':    
#     time_interval = time_interval()
#     T0 = time_interval[0]
#     T1 = time_interval[1]

########### Interface parameters #############
flag_compute_loss_each_epoch = True
n_epoch_show_info = 100
flag_show_sn_info = False
flag_show_plot = True
flag_output_results = True


########### Depending parameters #############
u_net = network_file.network(d,m, activation_type = activation, boundary_control_type = boundary_control, initial_constant = initial_constant)
if u_net.if_boundary_controlled == False:
    flag_boundary_term_in_loss = True  # if loss function has the boundary residual
else:
    flag_boundary_term_in_loss = False
if time_dependent_type == 'none':
    flag_initial_term_in_loss = False  # if loss function has the initial residual
else:
    if u_net.if_initial_controlled == False:
        flag_initial_term_in_loss = True
    else:
        flag_initial_term_in_loss = False
if flag_boundary_term_in_loss == True or flag_initial_term_in_loss == True:
    flag_IBC_in_loss = True  # if loss function has the boundary/initial residual
    N_IBC_train = 0  # number of boundary and initial training points
else:
    flag_IBC_in_loss = False
if flag_boundary_term_in_loss == True:
    if domain_shape == 'cube':
        if d == 1 and time_dependent_type == 'none':
            N_each_face_train = 1
        else:
            N_each_face_train = max([1,int(round(N_inside_train/2/d))]) # number of sampling points on each domain face when trainning
        N_boundary_train = 2*d*N_each_face_train
    elif domain_shape == 'sphere':
        if d == 1 and time_dependent_type == 'none':
            N_boundary_train = 2
        else:
            N_boundary_train = N_inside_train # number of sampling points on each domain face when trainning
    N_IBC_train = N_IBC_train + N_boundary_train
else:
    N_boundary_train = 0
if flag_initial_term_in_loss == True:          
    N_initial_train = max([1,int(round(N_inside_train/d))]) # number of sampling points on each domain face when trainning
    N_IBC_train = N_IBC_train + N_initial_train

########### Set functions #############
# if not time_dependent_type == 'none':   
#     def Du_t_ft(model, tensor_x_batch):
#         h = 0.01 # step length ot compute derivative
#         s = torch.zeros(tensor_x_batch.shape[0])
#         ei = torch.zeros(tensor_x_batch.shape)
#         ei[:,0] = 1
#         s = (3*model(tensor_x_batch+2*h*ei)-4*model(tensor_x_batch+h*ei)+model(tensor_x_batch))/2/h
#         return s

#################### Start ######################
optimizer = optim.Adam(u_net.parameters(),lr=lrseq[0])
lossseq = zeros((n_epoch,))
resseq = zeros((n_epoch,))

N_plot = 1001
x_plot = zeros((N_plot,d))
x_plot[:,0] = linspace(domain_intervals[0,0],domain_intervals[0,1],N_plot)

# Training 
k = 0
while k < n_epoch:
    ## generate training and testing data (the shape is (N,d)) or (N,d+1) 
    ## label 1 is for the points inside the domain, 2 is for those on the bondary or at the initial time
    x1_train = generate_uniform_points_in_cube(domain_intervals, N_inside_train)
    
#     if flag_IBC_in_loss == True:
#         x2_train = generate_uniform_points_on_cube(domain_intervals,N_each_face_train)  
    
    tensor_x1_train = Tensor(x1_train)
    tensor_x1_train.requires_grad = False
    
    tensor_f1_train = Tensor(f(x1_train))
    tensor_f1_train.requires_grad = False
    
#     if flag_boundary_term_in_loss == True:
#         tensor_x2_train = Tensor(x2_train)
#         tensor_x2_train.requires_grad=False
#         tensor_g2_train = Tensor(g(x2_train))
#         tensor_g2_train.requires_grad=False

    ## Set learning rate 
    
    # meaning as training, the learning rate decrease in exp rate 
    for param_group in optimizer.param_groups:
        if flag_preiteration_by_small_lr == True and k == 0:
            param_group['lr'] = lr_pre
        else:
            param_group['lr'] = lrseq[k]
            
    # Train the solution net 
    
    for i_update in range(n_update_each_batch):
        if flag_compute_loss_each_epoch == True or i_update == 0:
            # computing the loss function 
            residual_sq = 1 / N_inside_train * torch.sum((Du_ft(u_net,tensor_x1_train)-tensor_f1_train)**2) 
            
            loss = residual_sq
        
        ## Update the network
        optimizer.zero_grad()
        
        # question about the parameters here... 
        loss.backward(retain_graph=not flag_compute_loss_each_epoch)
        optimizer.step()
        
    # Save loss and L2 error
    lossseq[k] = loss.item()
    resseq[k] = sqrt( 1/N_inside_train * sum( (Du(u_net,x1_train)-f(x1_train))**2 ))
    
    ## show information 
    if k * n_epoch_show_info == 0:
        if flag_show_plot == True:
            if i_update % 10 == 0:
                # Plot the slice for xd
                clf()
                plot(x_plot[:,0],u_net.predict(x_plot),'r')
                show()
                pause(0.02)
        if flag_compute_loss_each_epoch:
            print("epoch = %d, loss = %2.5f, residual = %2.5f" %(k,loss.item(),resseq[k]), end='')
            print('')
        else:
            print("epoch = %d" % k, end='')
        
        print("\n")
        
    # update to next 
    k = k + 1


# compute u_plot at x_plot
u_net_plot = u_net.predict(x_plot)

# Save u_net for deflation
localtime = time.localtime(time.time())
time_text = str(localtime.tm_mon)+'_'+str(localtime.tm_mday)+'_'+str(localtime.tm_hour)+'_'+str(localtime.tm_min)
torch.save(u_net.state_dict(),'networkpara_'+time_text+'.pkl')

# Output results
if flag_output_results == True:
    # compute the plot 
    N_plot = 201
    
    x0_plot = linspace(0,1,N_plot)
    x_plot = zeros((N_plot,d))
    x_plot[:,0] = x0_plot
    u_net_plot = u_net.predict(x_plot)

    #save the data
    main_file_name = 'main_Painleve'
    data = {'main_file_name':main_file_name,\
                                'ID':time_text,\
                                'N_inside_train':N_inside_train,\
                                'N_plot':N_plot,\
                                'activation':activation,\
                                'boundary_control':boundary_control,\
                                'd':d,\
                                'domain_shape':domain_shape,\
                                'flag_preiteration_by_small_lr':flag_preiteration_by_small_lr,\
                                #'flag_reset_select_net_each_epoch':flag_reset_select_net_each_epoch,\
                                #'selectnet_initial_constant':selectnet_initial_constant,\
                                #'h_Du_t':h_Du_t,\
                                'initial_constant':initial_constant,\
                                'lambda_term':lambda_term,\
                                'lossseq':lossseq,\
                                'lr_pre':lr_pre,\
                                'lrseq':lrseq,\
                                'm':m,\
                                'method':method,\
                                'n_epoch':n_epoch,\
                                'n_update_each_batch':n_update_each_batch,\
                                #'n_update_each_batch_pre':n_update_each_batch_pre,\
                                'resseq':resseq,\
                                'time_dependent_type':time_dependent_type,\
                                'u_net_plot':u_net_plot,\
                                'x_plot':x_plot,\
                                }
    filename = 'Painleve_d_'+str(d)+'_'+method+'_'+time_text+'.data'
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()
