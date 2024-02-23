import argparse
import os
import sys
from typing import Any

import toml


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


class Args:
    instance = None
    instance_arg = None

    def __init__(self):
        self.args = self._get_parser().parse_args()
        self.args.toml = self.args.toml or os.path.join('configs', '{}.toml'.format(str(sys.argv[0])[:-3]))
        if os.path.exists(self.args.toml):
            self.update()
        else:
            print('Trying to load config file {} but it doesnt exist'.format(self.args.toml))

    def update(self):
        from_file = toml.load(self.args.toml)
        for k, v in from_file.items():
            if not getattr(self.args, k):
                setattr(self.args, k, v)

    @staticmethod
    def _get_parser():
        parser = argparse.ArgumentParser(description='ViT Activation Maximisation Visualiser')
        parser.add_argument('-t', '--toml', type=str, help='Path to toml file that contains the default configs')
        parser.add_argument('-x', '--experiment', type=str, help='Experiment name')
        parser.add_argument('--seed', type=int, default=6247423, help='Random seed')

        parser.add_argument('-n', '--network', type=int, default=35, help='Max 100, get Network')
        parser.add_argument('-l', '--layer', type=int, default=0, help='Index of the layer in which to look.')
        parser.add_argument('-f', '--feature', type=int, default=0, help='Index of the vector element to maximise.')
        parser.add_argument('-y', '--target', type=int, default=0, help='# Feature')  # TODO: What is this?
        parser.add_argument('-m', '--method', type=str, default='in_feat', help='Family of vectors in which to look.',
                            choices=["ffnn", 'in_feat', 'keys', 'queries', 'values', 'out_feat'])

        parser.add_argument('-v', '--tv', type=float, default=1.0,
                            help='Regularisation constant for TotalVar. Will be scaled by 0.0005.')
        parser.add_argument('--no_preaugment', type=bool, default=False,
                            help="Whether to disable the preaugmentation used by Ghiasi & Kamezi to jitter ViT input.")
        parser.add_argument('--n_patches', type=int, default=0,
                            help="Amount of patches to send into the model to optimise. (Defaults to the standard model input.)")
        parser.add_argument('-r', '--lr', type=float, default=0.1,
                            help='Learning rate of gradient ascent.')
        parser.add_argument('-s', '--sign', type=int, default=1, choices=[1, -1],
                            help="Sign of the activation to maximise. If -1, the activation is minimised.")
        parser.add_argument('-i', '--iterations', type=int, default=400,
                            help="Amount of iterations to refine input for.")
        parser.add_argument('-g', '--grid', type=float, default=0, help='Variable "a" for development')
        parser.add_argument('-p', '--patch', type=int, default=16, help='Patch-Size for Visualization')

        parser.add_argument('-d', '--dir', type=str, default=None, help='Images dir to find top5 classes')
        parser.add_argument('--output_square', type=bool, default=True,
                            help="Whether to reshape the result into a square if it is rectangular.")
        parser.add_argument('--log_every', type=int, default=10,
                            help="Iterations between prints.")
        parser.add_argument('--save_every', type=int, default=100,
                            help="Iterations between intermediate saves.")

        return parser

    def get_args(self):
        return self.args

    def get_name(self):
        return '_'.join(['{}{}'.format(k[0], v) for k, v in vars(self.args).items() if v is not None])

    @staticmethod
    def get_instance() -> (argparse.Namespace, Any):
        Args.instance = Args.instance or Args()
        return Args.instance
