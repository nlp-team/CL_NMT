import torch.optim

from . import FairseqOptimizer, register_optimizer

import math
import sys
sys.path.append('..')
sys.path.append('..')
sys.path.append('..')
from optims import adabound


@register_optimizer('adabound')
class FairseqAdaBound(FairseqOptimizer):
    def __init__(self, args, params):
        super().__init__(args, params)
        self._optimizer = adabound.AdaBound(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--adabound-betas', default='(0.9, 0.999)', metavar='B',
                            help='betas for Adam optimizer')
        parser.add_argument('--adabound-eps', type=float, default=1e-8, metavar='D',
                            help='epsilon for Adam optimizer')
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
        parser.add_argument('--final_lr', default=0.1, type=float, metavar='D',
                            help='The final learning rates')

        # fmt: on

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            'lr': self.args.lr[0],
            'betas': eval(self.args.adabound_betas),
            'eps': self.args.adabound_eps,
            'weight_decay': self.args.weight_decay,
            'final_lr': self.args.final_lr,
        }

