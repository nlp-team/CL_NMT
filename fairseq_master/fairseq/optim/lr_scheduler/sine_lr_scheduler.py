# MIT License
#
# Copyright (c) 2019 nlp-team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import numpy as np

from . import FairseqLRScheduler, register_lr_scheduler


@register_lr_scheduler('sine')
class SineScheduler(FairseqLRScheduler):
    """---"""

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        if len(args.lr) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with sine.'
                ' Consider --lr-scheduler=fixed instead.'
            )

        self.min_lr = args.lr[0]
        self.max_lr = args.max_lr

        assert self.max_lr > self.min_lr, 'max_lr must be more than lr'

        self.period = args.lr_period_updates

        if self.period <= 0:
            assert args.max_update >= 0, 'Either --max_update or --lr-period-updates must be set'
            self.period = args.max_update - args.warmup_updates

        self.lr_step = 1
        self.lr_shrink = args.lr_shrink
        self.shrink_min = args.shrink_min
        self.stepsize = self.period // 2
        self.perturbate = args.perturbate
        # initial learning rate
        self.lr = args.min_lr
        self.optimizer.set_lr(self.lr)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        # fmt: off
        parser.add_argument('--max-lr', type=float, metavar='LR',
                            help='max learning rate, must be more than args.lr')
        parser.add_argument('--lr-period-updates', default=-1, type=float, metavar='LR',
                            help='initial number of updates per period')
        parser.add_argument('--lr-shrink', default=0.1, type=float, metavar='LS',
                            help='shrink factor for annealing')
        parser.add_argument('--shrink-min', action='store_true',
                            help='if set, also shrinks min lr')
        parser.add_argument('--perturbate', action='store_true',
                            help='if set, it distorts learning rate a small amount')
        # fmt: on

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        i = math.floor(num_updates / self.period)
        t_i = self.period
        t_curr = num_updates - (self.period * i)

        lr_shrink = self.lr_shrink ** i
        max_lr = self.max_lr * lr_shrink
        if self.shrink_min:
            min_lr = self.min_lr * lr_shrink
        else:
            min_lr = self.min_lr

        self.lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * (t_curr - self.stepsize) / self.stepsize))
        if self.perturbate:
            self.lr = np.random.normal(self.lr, (self.max_lr - min_lr) / 16)
        self.lr = max(min(max_lr, self.lr), min_lr)
        self.optimizer.set_lr(self.lr)
        return self.lr
