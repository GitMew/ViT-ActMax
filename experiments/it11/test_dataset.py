import pdb

from torch.utils.data import DataLoader
from vit_actmax.datasets import weird_image_net, image_net
from vit_actmax.hooks.transformer.vit import SimpleViTGeLUHook, SimpleClipGeLUHook
from vit_actmax.model import model_library
from vit_actmax.utils import exp_starter_pack
import torch
from tqdm import tqdm


@torch.no_grad()
def main():
    exp_name, args, _ = exp_starter_pack()
    # network, batch_size = 99, 24
    # dim_all = 12 * 768 * 4

    data = image_net.train()
    pdb.set_trace()


if __name__ == '__main__':
    main()
