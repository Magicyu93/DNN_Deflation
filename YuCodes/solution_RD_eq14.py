import torch
from numpy import array, zeros

torch.set_default_tensor_type('torch.DoubleTensor')

h = 0.001

# domain
d = 1
left = 0
right = 10
scale = 10
right_scaled = 1

# Solution Parameter
rho = 0.01
D = 0.1
mu = 1


# output should have the same dimension as the input
def res1(net_A, net_S, tensor_x_batch):
    laplace_a = torch.zeros(tensor_x_batch.shape[0], )

    for i in range(tensor_x_batch.shape[1]):
        ei = torch.zeros(tensor_x_batch.shape)
        ei[:, i] = 1
        laplace_a = laplace_a + 1/scale/scale *(net_A(tensor_x_batch+h*ei) - 2*net_A(tensor_x_batch) + net_A(tensor_x_batch-h*ei))/h/h

    s = net_S(tensor_x_batch)
    a = net_A(tensor_x_batch)

    res1 = D * laplace_a + s * a**2 /(1 + a**2) - a + rho
    return res1


def res2(net_A, net_S, tensor_x_batch):
    laplace_s = torch.zeros(tensor_x_batch.shape[0], )

    for i in range(tensor_x_batch.shape[1]):
        ei = torch.zeros(tensor_x_batch.shape)
        ei[:, i] = 1
        laplace_s = laplace_s + 1/scale/scale * (net_S(tensor_x_batch+h*ei)
                                                 - 2*net_S(tensor_x_batch) + net_S(tensor_x_batch-h*ei))/h/h

    s = net_S(tensor_x_batch)
    a = net_A(tensor_x_batch)

    res2 = laplace_s + mu * (1 - s * a**2 / (1 + a**2))
    return res2


def res(net_A, net_S, tensor_x_batch):
    return (res1(net_A, net_S, tensor_x_batch)**2 + res2(net_A, net_S, tensor_x_batch)**2)**0.5


# computing the boundary normal derivative for each given points
# based on the output of 'generate_uniform_points_on_cube' in 'useful_tools'
# domain: cube
# time independent
def res_bd(model, tensor_x_batch):
    dim = tensor_x_batch.shape[1]
    n_each_face = int(tensor_x_batch.shape[0] / dim / 2)

    res_bd = torch.zeros(tensor_x_batch.shape)

    for i in range(dim):
        count_start = i * 2 * n_each_face
        count_end = (i + 1) * 2 * n_each_face - 1

        ei = torch.zeros(tensor_x_batch[count_start:count_end, :].shape)
        ei[:, i] = 1

        res_bd[count_start:count_end, i] = 1/scale * (model(tensor_x_batch[count_start:count_end, :] + h*ei)
                                                       - model(tensor_x_batch[count_start:count_end, :] - h*ei))/2/h
    return res_bd


def steady_state_sol(rho):
    A_s = 1 + rho
    S_s = 1 + (1 + rho) ** (-2)
    return A_s, S_s


def domain_shape():
    return 'cube'


def domain_parameter(d):
    intervals = zeros((d, 2))
    for i in range(d):
        intervals[i, :] = array([left, right_scaled])
    return intervals


# If this is a time-dependent problem
def time_dependent_type():
    return 'none'


# output the time interval
def time_interval():
    return None