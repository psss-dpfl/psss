import torch
from torch.nn import Module
from torch.nn import Linear
from opacus.grad_sample import GradSampleModule
import torch.utils.data as data_utils
from torch.utils.data import RandomSampler, DataLoader
import numpy as np


def least_power_2_upper_bound(d):
        upper_bound = 1
        while upper_bound < d:
            upper_bound = upper_bound * 2
        return upper_bound

def discrete_tensor(tensor, l2_sensitivity):
    floor = torch.floor(tensor)
    p_ceil = tensor - floor
    while True:
        random_nums = torch.rand(size=tensor.size(), device=tensor.device, requires_grad=False)
        choice_floor = (random_nums > p_ceil).type(torch.float32)

        discrete_tensor = choice_floor * floor + (1. - choice_floor) * (floor + 1.)

        l2_norm_square = torch.norm(discrete_tensor, p=2) ** 2

        if l2_norm_square <= l2_sensitivity ** 2:
            return discrete_tensor

    raise ValueError("Cannot discrete tensor")

def clip_grad_by_l2norm(grad, c):
    grad_norm = torch.norm(grad, p=2)
    return grad * (min(grad_norm, c) / grad_norm)


def skellam_noise(size, lambda_, device):
    noise = torch.poisson(torch.ones(size=size) * lambda_) - torch.poisson(torch.ones(size=size) * lambda_)
    return noise.to(device)


# cross-entropy loss for logistic regression
""" pred = <w,x>
    loss = -y log (sigmoid(<w,x>)) - (1-y) (1-log (sigmoid(<w,x>)))
         = -y log (sigmoid(<w,x>)) - (1-y) log(sigmoid(-<w,x>))
    when y=1, loss = log (1+exp(-<w,x>)) 
    when y=0, loss = log (1+exp(<w,x>)) 
"""
def criterion(pred, label):
    return torch.mean(torch.log(1. + torch.pow(torch.exp(-1. * pred), 2. * label - 1.)))


def compute_per_sample_grad(model, x, label, criterion):
    x = torch.unsqueeze(x)
    label = torch.unsqueeze(label)

    pred = model(x)
    loss = criterion(pred, label)

    return torch.autograd.grad(loss, list(model.parameters))

def load_tensor_by_batch(tensor, batch_size=128, drop_last=False):
    # convert to tensor
    loader = data_utils.DataLoader(tensor, batch_size=batch_size, drop_last=drop_last, shuffle=True)
    return loader

def sample_batch_of_tensor(tensor, batch_size):
    sampler = RandomSampler(tensor, replacement=False, num_samples=batch_size)
    loader = DataLoader(tensor, sampler=sampler, batch_size=batch_size)
    return loader

def str2bool(v):
    if v.lower() in ['yes', 'true']:
        return True
    else:
        return False
    
def sigmoid_deriv(order):
    d = order

    deriv_array = np.zeros(d+1)
    ans = 1 / (1+np.exp(-0))
    # print(0, '-th order derivative', ans)
    deriv_array[0] = ans

    ak_array = np.zeros(d+2)
    ak_array[0] = 0
    ak_array[1] = 1
    ak_array[2] = -1
    ans = ak_array[1] / np.power(2,1) + ak_array[2] / np.power(2,2)
    # print(1, '-th order derivative', ans)
    deriv_array[1]=ans

    # print(ak_array)
    for K in range(2,d+1):
        # update a_k
        tmp_array = np.zeros(d+2)
        tmp_array[K+1] = -K*ak_array[K]
        for k in range(1,K+1):
            tmp_array[k] = k*ak_array[k] - (k-1)*ak_array[k-1]
        ans = 0
        for k in range(K+2):
            ans = ans + tmp_array[k] / np.power(2,k)
        # print(K, '-th order derivative', ans)
        deriv_array[K] = ans
        # print(tmp_array)
        for k in range(d+2):
            ak_array[k] = tmp_array[k]
            
    return deriv_array

