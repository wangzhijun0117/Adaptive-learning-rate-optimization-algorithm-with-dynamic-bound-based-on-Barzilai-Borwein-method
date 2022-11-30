import torch
from torch.optim.optimizer import Optimizer
import math

class AdaBBbound(Optimizer):
    def __init__(self,params,
            steps=400,gamma=1e-3,beta=0.01,
            weight_decay=0., lr=1e-3, beta2=0.999, bblr=0.1,eps=1e-8,
             ):
        assert steps > 0, ValueError("Invalid steps: {}".format(steps))
        assert weight_decay >= 0.0, ValueError("Invalid weight_decay value: {}".format(weight_decay))
        assert lr > 0.0, ValueError("Invalid initial learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(beta2))
        if not 0.0 <= bblr:
            raise ValueError("Invalid final learning rate: {}".format(bblr))
     
        defaults = dict(
            steps=int(steps),gamma=gamma,beta=beta,
            lr=lr, beta2=beta2, bblr=bblr,
            eps=eps,weight_decay=weight_decay
        )
    
        super(AdaBBbound, self).__init__(params, defaults)
        assert len(self.param_groups) == 1, ValueError("doesn't support per-parameter options (parameter groups)")
        
        self._params = self.param_groups[0]['params']
        self.pre_lr = bblr

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        group = self.param_groups[0]
        state = self.state[self._params[0]]
        state.setdefault('bb_iter', -1)
        state.setdefault('n_iter', 0)

        state['n_iter'] += 1   
     
        if (state['n_iter']-1) % group['steps'] == 0:
            state['bb_iter'] += 1
            sum_dp_dg = 0
            sum_dp_norm = 0

            for p in self._params:
                if state['n_iter'] == 1:
                    with torch.no_grad():
                        self.state[p]['grad_aver'] = torch.zeros_like(p)
                        self.state[p]['grads_prev'] = torch.zeros_like(p)
                        self.state[p]['params_prev'] = torch.zeros_like(p)
                        self.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
                if state['bb_iter'] > 1:
                    params_diff = p.detach() - self.state[p]['params_prev']
                    grads_diff = self.state[p]['grad_aver'] - self.state[p]['grads_prev']
                    sum_dp_dg += (grads_diff * params_diff).sum().item()
                    sum_dp_norm += params_diff.norm().item() ** 2

                if state['bb_iter'] > 0:
                    self.state[p]['grads_prev'].copy_(self.state[p]['grad_aver'])
                    self.state[p]['params_prev'].copy_(p.detach())
                    self.state[p]['grad_aver'].zero_()
                    
            if state['bb_iter'] > 1:
                if abs(sum_dp_dg) >= 1e-10:
                    lr_hat = sum_dp_norm / (sum_dp_dg * group['steps'])
                    lr = abs(lr_hat)
                    pre_lr = group['bblr']
                    group['bblr'] = lr
                    
            upper_bound = pre_lr * (1 + 1 / (group['gamma'] * state['n_iter']))
            if group['bblr'] > upper_bound:
              group['bblr'] = upper_bound

        for group in self.param_groups:
            for p in self._params:
                if p.grad is None:
                    continue
                grad = p.grad.data

                exp_avg_sq =  self.state[p]['exp_avg_sq']
                beta2 = group['beta2']

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction2 = 1 - beta2 ** state['n_iter']
                step_size = group['lr'] * math.sqrt(bias_correction2)
                
                lower_bound = group['bblr'] * (1 - 1 / (group['gamma'] * state['n_iter'] + 1))
                upper_bound = group['bblr'] * (1 + 1 / (group['gamma'] * state['n_iter']))
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(grad)
            
                p.data.add_(-step_size)
                
                with torch.no_grad():
                    self.state[p]['grad_aver'].mul_(1 - group['beta']).add_(group['beta'], grad)

        return loss