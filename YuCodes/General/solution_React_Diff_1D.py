#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 13:57:41 2020

@author: haoli
"""

import torch
from numpy import array, zeros, sqrt

torch.set_default_tensor_type('torch.DoubleTensor')
h = 0.001

########### Solution Parameter ############
rho = 0.01
D = 0.1
mu = 1

# Steady State solution
A_s = 1+rho
S_s = (1+rho)**(-2)

left = 0
right = 10
scale = 10
right_scaled = right / scale


# the point-wise residual: Input x is a batch of sampling points of d variables (tensor); Output is tensor vector
def res1(net_A, net_S, tensor_x_batch):
    s = torch.zeros((tensor_x_batch.shape[0],))

    for i in range(tensor_x_batch.shape[1]):
        ei = torch.zeros(tensor_x_batch.shape)
        ei[:, i] = 1
        s = s + scale * scale * D*(net_A(tensor_x_batch+h*ei)-2*net_A(tensor_x_batch)+net_A(tensor_x_batch-h*ei))/h/h

    s = s + net_S(tensor_x_batch)*net_A(tensor_x_batch)**2 - net_A(tensor_x_batch) + rho
    return s


def res2(net_A, net_S, tensor_x_batch):
    s = torch.zeros((tensor_x_batch.shape[0],))

    for i in range(tensor_x_batch.shape[1]):
        ei = torch.zeros(tensor_x_batch.shape)
        ei[:, 0] = 1
        s = s + scale * scale * (net_S(tensor_x_batch+h*ei)-2*net_S(tensor_x_batch)+net_S(tensor_x_batch-h*ei))/h/h

    s = s + mu*(1 - net_S(tensor_x_batch)*net_A(tensor_x_batch)**2)
    return s


def res(net_A, net_S, tensor_x_batch):
    return (res1(net_A, net_S, tensor_x_batch)**2 + res2(net_A, net_S, tensor_x_batch)**2)**0.5


# the point-wise Bu for tensor (N sampling points of d variables)
# right now only works for the 1d case
def res_bd(model, tensor_x2_train):
    ei = torch.zeros(tensor_x2_train.shape, )
    ei[:, 0] = 1
    res_bd = scale * (model.forward(tensor_x2_train + h*ei) - model.forward(tensor_x2_train - h*ei))/2/h
    return res_bd


# specify the domain type
def domain_shape():
    return 'cube'


# output the domain parameters
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
