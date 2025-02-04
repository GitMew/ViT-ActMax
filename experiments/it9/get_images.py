import os

import torchvision.utils
from torch.utils.data import DataLoader
from vit_actmax.datasets import weird_image_net
from vit_actmax.hooks.transformer.vit import SimpleViTGeLUHook
from vit_actmax.model import model_library
from vit_actmax.utils import exp_starter_pack
from vit_actmax.saver.base import BASE_OUTPUT_DIRECTORY
import torch
from tqdm import tqdm


@torch.no_grad()
def main():
    exp_name, args, _ = exp_starter_pack()
    method = 'eval'
    # method = 'train'
    feat_count = 8

    dataset = weird_image_net.eval() if method == 'eval' else weird_image_net.train()
    indices = torch.load(f'{method}.pt')
    indices = indices.view(12, -1)[:, :feat_count]

    parent = BASE_OUTPUT_DIRECTORY / method
    parent.mkdir(exist_ok=True)
    for l in tqdm(range(12)):
        for f in range(feat_count):
            img, _ = dataset[indices[l, f]]
            torchvision.utils.save_image(img, (parent / f'{l}_{f}.png').as_posix())


if __name__ == '__main__':
    main()
