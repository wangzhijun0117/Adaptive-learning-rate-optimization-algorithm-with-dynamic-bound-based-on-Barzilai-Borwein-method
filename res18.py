from __future__ import print_function
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

from torch.optim.optimizer import Optimizer
import math
from collections import OrderedDict
import random

import torch.utils.model_zoo as model_zoo
#import datetime
import time
import torch
from torch.optim.optimizer import required
class BBmom(Optimizer):

    def __init__(self, params, 
                 lr=1e-1,
                 steps=400,
                 beta=0.01,
                 beta1=0.9,
                 weight_decay=0.,
                 gamma=0.001,
                 nesterov=True
                 ):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if beta < 0.0:
            raise ValueError("Invalid beta value: {}".format(beta))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, steps=int(steps), gamma=gamma, beta=beta,beta1=beta1,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (beta <= 0):
            raise ValueError("Nesterov momentum requires a beta")
        super(BBmom, self).__init__(params, defaults)
        
        self._params = self.param_groups[0]['params']
        
    def __setstate__(self, state):
        super(BBmom, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        assert len(self.param_groups) == 1, ValueError("BB doesn't support per-parameter options (parameter groups)")
        group = self.param_groups[0]
        # register the global state of BB as state for the first param
        state = self.state[self._params[0]]
        state.setdefault('bb_iter', -1)
        state.setdefault('n_iter', -1)

        state['n_iter'] += 1
        if state['n_iter'] % group['steps'] == 0:
            state['bb_iter'] += 1
            sum_dp_dg = 0
            sum_dp_norm = 0
#            b = 0
            for p in self._params:
                if state['n_iter'] == 0:
                    with torch.no_grad():
                        self.state[p]['grad_aver'] = torch.zeros_like(p)
                        self.state[p]['grads_prev'] = torch.zeros_like(p)
                        self.state[p]['params_prev'] = torch.zeros_like(p)
                        self.state[p]['grad_aver2'] = torch.zeros_like(p)
                        
                if state['bb_iter'] > 1:
                    params_diff = p.detach() - self.state[p]['params_prev']
                    grads_diff = self.state[p]['grad_aver2'] - self.state[p]['grads_prev']
                    sum_dp_dg += (grads_diff * params_diff).sum().item()
                    sum_dp_norm += params_diff.norm().item() ** 2
                    
#                    b += torch.norm(p.grad.data)/torch.norm(self.state[p]['grad_aver'])/len(self._params)
#                    print(b)
                    
                if state['bb_iter'] > 0:
                    self.state[p]['grads_prev'].copy_(self.state[p]['grad_aver2'])
                    self.state[p]['params_prev'].copy_(p.detach())
                    self.state[p]['grad_aver2'].zero_()
                    
            if state['bb_iter'] > 1:
                if abs(sum_dp_dg) >= 1e-10:
                    lr_hat = sum_dp_norm / (sum_dp_dg * group['steps'])
                    lr = abs(lr_hat)
                    a = group['lr']
                    group['lr'] = lr
                    
                    print('学习率',group['lr'])
                    upper_bound = a * (1 + 1 / (group['gamma'] * (state['n_iter']+1)))
                    print('上界',upper_bound)
                    if group['lr'] > upper_bound:
                        group['lr'] = upper_bound
                        print('超出上界')
                    
        for p in self._params:

            if p.grad is None:
                continue
            d_p = p.grad.data
            if group['weight_decay'] != 0:
                d_p.add_(group['weight_decay'], p.data)

            if group['nesterov']:
                d_p = d_p.add(group['beta1'], d_p)
            else:
                d_p = d_p
                
            self.state[p]['grad_aver'].mul_(group['beta1']).add_(1 - group['beta1'], d_p)    
            p.data.add_(-group['lr'], self.state[p]['grad_aver'])
            
            with torch.no_grad():
                    self.state[p]['grad_aver2'].mul_(1 - group['beta']).add_(group['beta'], self.state[p]['grad_aver'])
                    
        return loss
class DBB(Optimizer):
    def __init__(self,
                 params,
                 lr=1e-1,
                 steps=400,
                 beta=0.01,
                 beta2=0.001,
                 weight_decay=0.,gamma=0.3,
                 ):
        assert lr > 0.0, ValueError("Invalid initial learning rate: {}".format(lr))
        assert steps > 0, ValueError("Invalid steps: {}".format(steps))
        assert 0.0 < beta <= 1.0, ValueError("Invalid beta value: {}".format(beta))
        assert beta2 > 0.0, ValueError("Invalid minimal learning rate: {}".format(beta2))
        assert weight_decay >= 0.0, ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            steps=int(steps),
            beta=beta,
            beta2=beta2,
            gamma=gamma,
        )

        super(DBB, self).__init__(params, defaults)

        assert len(self.param_groups) == 1, ValueError("BB doesn't support per-parameter options (parameter groups)")

        self._params = self.param_groups[0]['params']
        self.i = 0
        # self.lrr = []
        # self.a1 =[]
        # self.a2 = []
        # self.lrrr = []
        # self.cost = None
        self.s = 0 
        self.y = 0
    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()
        assert len(self.param_groups) == 1, ValueError("BB doesn't support per-parameter options (parameter groups)")
        group = self.param_groups[0]
        # register the global state of BB as state for the first param
        state = self.state[self._params[0]]
        state.setdefault('bb_iter', -1)
        state.setdefault('n_iter', -1)
        
        state['n_iter'] += 1
        if state['n_iter'] % group['steps'] == 0:
            state['bb_iter'] += 1
            sum_dp_dg = 0
            sum_dp_norm = 0

#            sum_dg_norm = 0

            for p in self._params:
                if state['n_iter'] == 0:
                    with torch.no_grad():
                        self.state[p]['grad_aver'] = torch.zeros_like(p)
                        self.state[p]['grads_prev'] = torch.zeros_like(p)
                        self.state[p]['params_prev'] = torch.zeros_like(p)
                        
                if state['bb_iter'] > 0:
                    params_diff = p.detach() - self.state[p]['params_prev']
                    grads_diff = self.state[p]['grad_aver'] - self.state[p]['grads_prev']
                    
                    sum_dp_dg += (grads_diff * params_diff).sum().item()
                    sum_dp_norm += params_diff.norm().item() ** 2

                if state['bb_iter'] > 0:
                    self.state[p]['grads_prev'].copy_(self.state[p]['grad_aver'])
                    self.state[p]['params_prev'].copy_(p.detach())
                    self.state[p]['grad_aver'].zero_()
                    
                
                
#            print(group['gamma'])        
            if state['bb_iter'] > 1:
                if abs(sum_dp_dg) >= 1e-10:
                    self.s = group['gamma']*sum_dp_norm + (1-group['gamma'])*self.s
                    self.y = group['gamma']*sum_dp_dg + (1-group['gamma'])*self.y
                    
                    # self.a1.append(self.y)
                    # self.a2.append(self.s)
                    
                    lr = abs((self.s) / (self.y * group['steps']))
#                    self.lrrr.append(abs(lr))

                    a = group['lr']
                    group['lr'] = lr
                    
                    print('学习率',group['lr'])
                    upper_bound = a * (1 + 1 / (group['beta2'] * (state['n_iter']+1)))
                    print('上界',upper_bound)
                    if group['lr'] > upper_bound:
                        group['lr'] = upper_bound
                        print('超出上界')
                # self.lrrr.append(abs(group['lr']))
                
        for p in self._params:

            if p.grad is None:
                continue
            d_p = p.grad.data
#            self.state[p]['grad_aver'].mul_(1 - group['beta']).add_(group['beta'], d_p)
            if group['weight_decay'] != 0:
                d_p.add_(group['weight_decay'], p.data)

            # update gradients
            p.data.add_(-group['lr'],d_p)
            # average the gradients
            with torch.no_grad():
                self.state[p]['grad_aver'].mul_(1 - group['beta']).add_(group['beta'], d_p)
#                self.state[p]['grad_aver'] = self.state[p]['grad_aver'] /(1-(1-group['beta'])**state['n_iter'])
        return loss

class BB(Optimizer):
    def __init__(self,
                 params,
                 lr=1e-1,
                 steps=400,
                 beta=0.01,
                 min_lr=1e-1,
                 max_lr=10.0,
                 weight_decay=0.,
                 gamma=0.001,
                 ):
        assert lr > 0.0, ValueError("Invalid initial learning rate: {}".format(lr))
        assert steps > 0, ValueError("Invalid steps: {}".format(steps))
        assert 0.0 < beta <= 1.0, ValueError("Invalid beta value: {}".format(beta))
        assert min_lr > 0.0, ValueError("Invalid minimal learning rate: {}".format(min_lr))
        assert max_lr > min_lr, ValueError("Invalid maximal learning rate: {}".format(max_lr))
        assert weight_decay >= 0.0, ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            steps=int(steps),
            beta=beta,
            min_lr=min_lr,
            max_lr=max_lr,gamma=gamma,
        )

        super(BB, self).__init__(params, defaults)

        assert len(self.param_groups) == 1, ValueError("BB doesn't support per-parameter options (parameter groups)")

        self._params = self.param_groups[0]['params']
        # self.a1 = []
        # self.a2 = []
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        assert len(self.param_groups) == 1, ValueError("BB doesn't support per-parameter options (parameter groups)")
        group = self.param_groups[0]
        # register the global state of BB as state for the first param
        state = self.state[self._params[0]]
        state.setdefault('bb_iter', -1)
        state.setdefault('n_iter', -1)

        state['n_iter'] += 1
        # a = 0.1
        if state['n_iter'] % group['steps'] == 0:
            state['bb_iter'] += 1
            sum_dp_dg = 0
            sum_dp_norm = 0
#            sum_dg_norm = 0
            for p in self._params:
                if state['n_iter'] == 0:
                    with torch.no_grad():
                        self.state[p]['grad_aver'] = torch.zeros_like(p)
                        self.state[p]['grads_prev'] = torch.zeros_like(p)
                        self.state[p]['params_prev'] = torch.zeros_like(p)
                        
                if state['bb_iter'] > 1:
                    params_diff = p.detach() - self.state[p]['params_prev']
                    grads_diff = self.state[p]['grad_aver'] - self.state[p]['grads_prev']
                    sum_dp_dg += (grads_diff * params_diff).sum().item()
                    sum_dp_norm += params_diff.norm().item() ** 2
               
                if state['bb_iter'] > 0:
                    self.state[p]['grads_prev'].copy_(self.state[p]['grad_aver'])
                    self.state[p]['params_prev'].copy_(p.detach())
                    self.state[p]['grad_aver'].zero_()
                    
            # if state['bb_iter'] > 1: 
            #         self.a1.append(sum_dp_dg)
            #         self.a2.append(sum_dp_norm)
                    
            if state['bb_iter'] > 1:
                if abs(sum_dp_dg) >= 1e-10:
                    lr_hat = sum_dp_norm / (sum_dp_dg * group['steps'])
                    lr = abs(lr_hat)
                    a = group['lr']
                    group['lr'] = lr
                            
                    print('学习率',group['lr'])
                    upper_bound = a * (1 + 1 / (group['gamma'] * (state['n_iter']+1)))
                    print('上界',upper_bound)
    
                    if group['lr'] > upper_bound:
                        group['lr'] = upper_bound
                        print('超出上界')
        
        for p in self._params:

            if p.grad is None:
                continue
            d_p = p.grad.data
            if group['weight_decay'] != 0:
                d_p.add_(group['weight_decay'], p.data)

            # update gradients
            p.data.add_(-group['lr'], d_p)

            # average the gradients
            with torch.no_grad():
                self.state[p]['grad_aver'].mul_(1 - group['beta']).add_(group['beta'], d_p)

        return loss
class AdaBound(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3,
                 eps=1e-8, weight_decay=0, amsbound=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= final_lr:
            raise ValueError("Invalid final learning rate: {}".format(final_lr))
        if not 0.0 <= gamma < 1.0:
            raise ValueError("Invalid gamma parameter: {}".format(gamma))
        defaults = dict(lr=lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps,
                        weight_decay=weight_decay, amsbound=amsbound)
        super(AdaBound, self).__init__(params, defaults)

        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))

    def __setstate__(self, state):
        super(AdaBound, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsbound', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')
                amsbound = group['amsbound']

                state = self.state[p]

                # State initialization
#                print(state)
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsbound:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsbound:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsbound:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                # Applies bounds on actual learning rate
                # lr_scheduler cannot affect final_lr, this is a workaround to apply lr decay
                final_lr = group['final_lr'] * group['lr'] / base_lr
                lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)

                p.data.add_(-step_size)

        return loss

class AdaBbb(Optimizer):
    def __init__(self,params,
            steps=400,min_lr=1e-1,max_lr=10.0, gamma=1e-3,beta=0.01,
            weight_decay=0., lr=1e-3, beta2=0.999, final_lr=0.1,eps=1e-8,
             ):
        assert steps > 0, ValueError("Invalid steps: {}".format(steps))
        assert min_lr > 0.0, ValueError("Invalid minimal learning rate: {}".format(min_lr))
        assert max_lr > min_lr, ValueError("Invalid maximal learning rate: {}".format(max_lr))
        assert weight_decay >= 0.0, ValueError("Invalid weight_decay value: {}".format(weight_decay))
        assert lr > 0.0, ValueError("Invalid initial learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(beta2))
        if not 0.0 <= final_lr:
            raise ValueError("Invalid final learning rate: {}".format(final_lr))
     
        defaults = dict(
            steps=int(steps),gamma=gamma,beta=beta,
            min_lr=min_lr,max_lr=max_lr,
            lr=lr, beta2=beta2, final_lr=final_lr,
            eps=eps,weight_decay=weight_decay
        )
    
        super(AdaBbb, self).__init__(params, defaults)
        assert len(self.param_groups) == 1, ValueError("doesn't support per-parameter options (parameter groups)")
        
        self._params = self.param_groups[0]['params']
        self.i =0 
        self.flr = []
        self.blr = []
        self.ub = []
        self.lb = []
        self.fanwei = []
            
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        group = self.param_groups[0]
        state = self.state[self._params[0]]
        state.setdefault('bb_iter', -1)
        state.setdefault('n_iter', -1)

        state['n_iter'] += 1          
        if state['n_iter'] % group['steps'] == 0:
            state['bb_iter'] += 1
            sum_dp_dg = 0
            sum_dp_norm = 0

            for p in self._params:
                if state['n_iter'] == 0:
                    self.state[p]['grad_aver'] = torch.zeros_like(p)
                    self.state[p]['grads_prev'] = torch.zeros_like(p)
                    self.state[p]['params_prev'] = torch.zeros_like(p)
                    self.state[p]['final_lr'] = group['final_lr']
                    self.state[p]['step'] = 0
                    self.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
                        
                if state['bb_iter'] > 1:
                    params_diff = p.detach() - self.state[p]['params_prev']
                    grads_diff = self.state[p]['grad_aver'] - self.state[p]['grads_prev']
                    sum_dp_dg = (grads_diff * params_diff).sum()
                    sum_dp_norm = params_diff.norm() ** 2

                if state['bb_iter'] > 0:
                    self.state[p]['grads_prev'].copy_(self.state[p]['grad_aver'])
                    self.state[p]['params_prev'].copy_(p.detach())
                    self.state[p]['grad_aver'].zero_()
                    
                if state['bb_iter'] > 1:
#                    if abs(sum_dp_dg) >= 1e-10:
                    lr_hat = sum_dp_norm / (sum_dp_dg * group['steps'])
#                    print(abs(lr_hat))
                    lr_scaled = abs(lr_hat) * (state['bb_iter'] + 1)
                    self.blr.append(abs(lr_hat))
                    if (lr_scaled > group['max_lr']) or (lr_scaled < group['min_lr']):
                        blr = 1.0 / (state['bb_iter'] + 1)
                        self.i = self.i + 1
                    else:
                        blr = abs(lr_hat)

                    self.state[p]['final_lr'] = blr
                    print(self.state[p]['final_lr'])
                    print('--------------------------------',self.i)
                    self.flr.append(self.state[p]['final_lr'])
            self.ub.append(group['max_lr']/(state['bb_iter'] + 1))
            self.lb.append(group['min_lr']/(state['bb_iter'] + 1))
        for group in self.param_groups:
            for p in self._params:
                if p.grad is None:
                    continue
                grad = p.grad.data

                exp_avg_sq =  self.state[p]['exp_avg_sq']
                beta2 = group['beta2']

                self.state[p]['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction2 = 1 - beta2 ** self.state[p]['step']
                step_size = group['lr'] * math.sqrt(bias_correction2)
                
                lower_bound = self.state[p]['final_lr'] * (1 - 1 / (group['gamma'] * self.state[p]['step'] + 1))
                upper_bound = self.state[p]['final_lr'] * (1 + 1 / (group['gamma'] * self.state[p]['step']))
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(grad)
            
                p.data.add_(-step_size)
                
                with torch.no_grad():
                    self.state[p]['grad_aver'].mul_(1 - group['beta']).add_(group['beta'], grad)

        return loss
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
              if isinstance(m,nn.Conv2d):
                  n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                  m.weight.data.normal_(0, math.sqrt(2. / n))
              elif isinstance(m, nn.BatchNorm2d):
                  m.weight.data.fill_(1)
                  m.bias.data.zero_()

                    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ResNet18(num_classes=10):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    return model


def get_parameter_number(net):
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return  trainable_num


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    lo=0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        lo += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    l.append(lo/len(train_loader))
    print('train loss :',lo/len(train_loader))
def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    c.append(correct / len(test_loader.dataset))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    set_seed(args.seed)
#    use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                       transform=transforms.Compose([
                         transforms.RandomCrop(32,padding=4),
                            transforms.RandomHorizontalFlip(),
                           transforms.Resize((32, 32)),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010])
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010])
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model =  ResNet18().to(device)
    #model.apply(reset_parameters)
    steps = len(train_loader)
    
    # optimizer = Adam(model.parameters(),weight_decay=5e-4)
    #optimizer = AdaBound2(model.parameters(),weight_decay=5e-4, steps=steps)#, steps=steps, max_lr=3, min_lr=1. / 3
    #optimizer = AdaBbb(model.parameters(), steps=steps, beta = 4./steps, weight_decay=5e-4, beta2=0.999)

    print(get_parameter_number(model))
    print(torch.cuda.get_device_name(0))
    print(torch.__version__)
    # optimizer = BBmom(model.parameters(), steps=steps,beta = 4./steps,weight_decay=5e-4,nesterov=False)
    optimizer = DBB(model.parameters(), steps=steps,beta = 4./steps,weight_decay=5e-4)
    
    global c,l
    c=[]
    l=[]
    start =time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        
    end =time.perf_counter()
    print('Running time: %s s'%(end-start))
    #np.savetxt('adabbvggflr128zt.txt',optimizer.flr)
    #np.savetxt('adabbvggblr128zt.txt',optimizer.blr)
    #np.savetxt('adabbvggub128zt.txt',optimizer.ub)
    #np.savetxt('adabbvgglb128zt.txt',optimizer.lb)
    #if args.save_model:
    #    torch.save(model.state_dict(), "res34_adabb.pt")
if __name__ == '__main__':
    main()
    plt.figure()
    x_axis = np.linspace(0, 50, len(c), endpoint=True)
    plt.plot(x_axis,c)
    plt.show()
    
    plt.figure()
    x_axis = np.linspace(0, 50, len(l), endpoint=True)
    plt.plot(x_axis,l)
    plt.show()
    
    np.savetxt('Dbblossres18.txt',l)
    np.savetxt('Dbbaccres18.txt',c)
