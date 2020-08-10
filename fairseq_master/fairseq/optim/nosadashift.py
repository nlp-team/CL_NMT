import torch.optim

from . import FairseqOptimizer, register_optimizer

import math

import sys

sys.path.append('..')
sys.path.append('..')
sys.path.append('..')
from optims import nosadashift


@register_optimizer('nosadashift')
class FairseqNosAdaShift(FairseqOptimizer):
    def __init__(self, args, params):
        super().__init__(args, params)
        self._optimizer = nosadashift.NosAdaShift(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--nosadashift-betas', default='(0.9, 0.999)', metavar='B',
                            help='betas for Adam optimizer')
        parser.add_argument('--nosadashift-eps', type=float, default=1e-8, metavar='D',
                            help='epsilon for Adam optimizer')
        parser.add_argument('--nosadashift-gamma',type=float,default=0,metavar='D',
                            help='gamma for NosAdam optimizer')
        parser.add_argument('--nosadashift-keep_num', type=float, default=10, metavar='D',
                            help=' keep_num for adaShift optimizer')
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
            'betas': eval(self.args.nosadashift_betas),
            'eps': self.args.nosadashift_eps,
            'gamma':self.args.nosadashift_gamma,
            'keep_num':self.args.nosadashift_keep_num,
            'weight_decay': self.args.weight_decay,
        }