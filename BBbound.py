import torch
from torch.optim.optimizer import Optimizer
    
class BBbound(Optimizer):
    def __init__(self,
                 params,
                 lr=1e-1,
                 steps=400,
                 beta=0.01,
                 beta2=0.999,
                 weight_decay=0.,
                 gamma=0.001,
                 ):
        assert lr > 0.0, ValueError("Invalid initial learning rate: {}".format(lr))
        assert steps > 0, ValueError("Invalid steps: {}".format(steps))
        assert 0.0 < beta <= 1.0, ValueError("Invalid beta value: {}".format(beta))
        assert 0.0 < beta2 <= 1.0, ValueError("Invalid beta value: {}".format(beta2))
        assert weight_decay >= 0.0, ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            steps=int(steps),
            beta=beta,
            beta2=beta2,
            gamma=gamma,
        )

        super(BBbound, self).__init__(params, defaults)

        assert len(self.param_groups) == 1, ValueError("BB doesn't support per-parameter options (parameter groups)")

        self._params = self.param_groups[0]['params']
        self.lrr = []
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        assert len(self.param_groups) == 1, ValueError("BB doesn't support per-parameter options (parameter groups)")
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
                    
            if state['bb_iter'] > 1:
                if abs(sum_dp_dg) >= 1e-10:
                    lr_hat = sum_dp_norm / (sum_dp_dg * group['steps'])
                    lr = abs(lr_hat)
                    pre_lr = group['lr']
                    group['lr'] = lr

                    upper_bound = pre_lr * (1 + 1 / ((1-group['beta2']) * (state['n_iter']+1)))**(group['steps']*group['beta']/state['bb_iter'])
                    if group['lr'] > upper_bound:
                        group['lr'] = upper_bound

        for p in self._params:

            if p.grad is None:
                continue
            d_p = p.grad.data
            if group['weight_decay'] != 0:
                d_p.add_(group['weight_decay'], p.data)
            p.data.add_(-group['lr'], d_p)
            with torch.no_grad():
                self.state[p]['grad_aver'].mul_(1 - group['beta']).add_(group['beta'], d_p)

        return loss