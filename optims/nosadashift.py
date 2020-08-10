import math
import torch
from torch.optim.optimizer import Optimizer
from collections import deque


'''
    nosadam + adashift
'''
class NosAdaShift(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, gamma=0, lr_decay=False,reduce_func=torch.max,keep_num=10):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= gamma:
            raise ValueError("Invalid gamma value: {}".format(gamma))
        beta1, _ = betas
        exp_weight_sum = sum(beta1 ** i for i in range(keep_num))
        first_grad_weight = beta1 ** (keep_num - 1) / exp_weight_sum
        last_grad_weight = 1. / exp_weight_sum
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, gamma=gamma, lr_decay=lr_decay,
                        reduce_func=reduce_func,keep_num=keep_num,
                        first_grad_weight=first_grad_weight,
                        last_grad_weight=last_grad_weight
                        )
        super(NosAdaShift, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(NosAdaShift, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('lr_decay', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                # amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                    state["grad_deque"] = deque([grad.clone()], maxlen=group["keep_num"])

                    state['B_old'] = 0
                    state['B_new'] = 1
                    # if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        # state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                grad_deque = state["grad_deque"]

                # if amsgrad:
                #     max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                beta2 = state['B_old']/state['B_new']
                gamma = group['gamma']
                # pnorm = group['pnorm']
                lr_decay = group['lr_decay']

                state['step'] += 1

                step = state['step']
                state['B_old'] += math.pow(step, -gamma)
                state['B_new'] += math.pow(step+1, -gamma)


                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                grad_apply = len(grad_deque) == group["keep_num"]
                offset_grad = grad_deque[0]
                grad_deque.append(grad.clone())
                if not grad_apply:
                    continue

                first_grad_weight = group["first_grad_weight"]
                last_grad_weight = group["last_grad_weight"]

                (exp_avg.sub_(first_grad_weight, offset_grad).mul_(beta1)
                 .add_(last_grad_weight, grad))

                reduce_func = group["reduce_func"] or (lambda x: x)
                reduced_grad_sq = reduce_func(offset_grad.mul_(offset_grad))
                exp_avg_sq.mul_(beta2).add_(1 - beta2, reduced_grad_sq)
                bias_correction = 1 - beta2 ** (state["step"] - group["keep_num"])
                denom = exp_avg_sq.div(bias_correction).sqrt_().add_(group["eps"])

                p.data.addcdiv_(-group["lr"], exp_avg, denom)

        return loss