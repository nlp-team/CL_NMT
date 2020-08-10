import torch.optim

from . import FairseqOptimizer, register_optimizer

import math

import sys

sys.path.append('..')
sys.path.append('..')
sys.path.append('..')
from optims import nospadam


@register_optimizer('nospadam')
class FairseqNosPadam(FairseqOptimizer):
    def __init__(self, args, params):
        super().__init__(args, params)
        self._optimizer = nospadam.NosPadam(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--nospadam-betas', default='(0.9, 0.999)', metavar='B',
                            help='betas for Adam optimizer')
        parser.add_argument('--nospadam-eps', type=float, default=1e-8, metavar='D',
                            help='epsilon for Adam optimizer')
        parser.add_argument('--nospadam-gamma',type=float,default=0,metavar='D',
                            help='gamma for NosAdam optimizer')
        parser.add_argument('--nospadam-partial', type=float, default=0.125, metavar='D',
                            help='partial for padam optimizer')
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')


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
            'betas': eval(self.args.nospadam_betas),
            'eps': self.args.nospadam_eps,
            'gamma':self.args.nospadam_gamma,
            'partial':self.args.nospadam_partial,
            'weight_decay': self.args.weight_decay,
        }