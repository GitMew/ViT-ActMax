import os
import pdb

import torchvision.utils
from torch.utils.data import DataLoader
from vit_actmax.datasets import weird_image_net
from vit_actmax.hooks.transformer.vit import SimpleViTGeLUHook, SaliencyViTGeLUHook
from vit_actmax.model import model_library
from vit_actmax.utils import exp_starter_pack
from vit_actmax.saver.base import BASE_OUTPUT_DIRECTORY
import torch
from tqdm import tqdm


@torch.no_grad()
def main():
    exp_name, args, _ = exp_starter_pack()
    method = 'eval' if args.grid == 1 else 'train'
    feat_count = 256
    network = 35
    patch_size = 32

    dataset = weird_image_net.eval() if method == 'eval' else weird_image_net.train()
    indices = torch.load(f'{method}_{network}.pt')
    indices = indices.view(12, -1)[:, :feat_count]
    model, image_size, _, _ = model_library[network]()
    hook = SaliencyViTGeLUHook(model)

    parent_natural    = BASE_OUTPUT_DIRECTORY / f"{method}_{network}"
    parent_normalised = BASE_OUTPUT_DIRECTORY / f"{method}_{network}_mask"
    parent_natural.mkdir(exist_ok=True)
    parent_normalised.mkdir(exist_ok=True)
    for l in tqdm(range(12)):
        for f in range(feat_count):
            img, _ = dataset[indices[l, f]]
            img = img.cuda().unsqueeze(0)
            act = hook(img, l, f).view(1, image_size // patch_size, image_size // patch_size)
            torchvision.utils.save_image(img, (parent_natural    / f'{l}_{f}.png').as_posix())
            torchvision.utils.save_image(act, (parent_normalised / f'{l}_{f}.png').as_posix(), normalize=True)


if __name__ == '__main__':
    main()
