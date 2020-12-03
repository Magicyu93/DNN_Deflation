# build the neural network to approximate the solution
import torch
import math
from torch import tanh, squeeze, sin, sigmoid, autograd, cos
from torch.nn.functional import relu

# torch.set_default_tensor_type('torch.cuda.DoubleTensor')
torch.set_default_tensor_type('torch.DoubleTensor')


# a 2-layer feed forward network for the PDE solution
class Network(torch.nn.Module):
    def __init__(self, d, m, activation_type='ReLU', boundary_control_type='none', initial_constant='none', structure_probing_num=0):
        super(Network, self).__init__()
        self.layer1 = torch.nn.Linear(d, m)
        self.layer2 = torch.nn.Linear(m, m)
        self.layer3 = torch.nn.Linear(m, m)
        self.layer4 = torch.nn.Linear(m, 1)

        if activation_type == 'ReLU3':
            self.activation = lambda x: relu(x ** 3)
        elif activation_type == 'ReLU':
            self.activation = lambda x: relu(x)
        elif activation_type == 'sigmoid':
            self.activation = lambda x: sigmoid(x)
        elif activation_type == 'tanh':
            self.activation = lambda x: tanh(x)
        elif activation_type == 'sin':
            self.activation = lambda x: sin(x)

        self.boundary_control_type = boundary_control_type

        if boundary_control_type == 'none':
            self.boundary_enforced = False

        if not initial_constant == 'none':
            torch.nn.init.constant_(self.layer4.bias, initial_constant)  # why layer3.bias? Anyway this is not important

        # adding the structure probing initialization
        self.structure_probing_num = structure_probing_num
        self.c = torch.zeros((structure_probing_num, ))

        if structure_probing_num == 0:
            self.c.requires_grad = False
        else:
            self.c.requires_grad = True
            # self.c[structure_probing_num - 1, ] = torch.rand(1)



    def forward(self, tensor_x_batch):
        y = self.layer1(tensor_x_batch)
        y = self.layer2(self.activation(y))
        y = self.layer3(self.activation(y))
        y = self.layer4(self.activation(y))


        if self.structure_probing_num != 0:
            # adding the structure probing
            pi = torch.Tensor([math.pi])
            dim = tensor_x_batch.shape[1]
            probing = torch.ones((tensor_x_batch.shape[0], ))

            # setting probing function to be (\sum c_j cos((j+1) \pi x))(\sum c_j cos((j+1) \pi y))\sum c_j cos((j+1) \pi z)
            for d in range(dim):
                temp = torch.zeros((tensor_x_batch.shape[0], ))
                for i in range(self.structure_probing_num):
                    temp = temp + self.c[i, ] * cos((i + 1) * pi * tensor_x_batch[:, d])
                probing = probing * temp
            y = y.squeeze(1) + probing

        return y


    # to evaluate the solution with numpy array input and output
    def predict(self, x_batch):
        tensor_x_batch = torch.Tensor(x_batch)
        tensor_x_batch.requires_grad = False
        y = self.forward(tensor_x_batch)
        return y.cpu().detach().numpy()


    # evaluate the second derivative at for k-th coordinate
    def d2_exact(self, tensor_x_batch, k):
        y = self.forward(tensor_x_batch)
        tensor_weight = torch.ones(y.size())
        grad_y = autograd.grad(y, tensor_x_batch, grad_outputs=tensor_weight, retain_graph=True, create_graph=True,
                               only_inputs=True)
        d2y_k = autograd.grad(outputs=grad_y[0][:, k], inputs=tensor_x_batch, grad_outputs=tensor_weight,
                              retain_graph=True)[0][:, k]
        return d2y_k


    # evaluate the Laplace at tensor_x_batch
    def laplace(self, tensor_x_batch):
        d = tensor_x_batch.shape[1]
        y = self.forward(tensor_x_batch)
        tensor_weight = torch.ones(y.size())
        grad_y = autograd.grad(y, tensor_x_batch, grad_outputs=tensor_weight, retain_graph=True, create_graph=True,
                               only_inputs=True)
        laplace_y = torch.zeros(y.size())
        for i in range(d):
            laplace_y = laplace_y + \
                        autograd.grad(outputs=grad_y[0][:, i], inputs=tensor_x_batch, grad_outputs=tensor_weight,
                                      retain_graph=True)[0][:, i]
        return laplace_y