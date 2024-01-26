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
    method = 'eval'
    feat_count = 8
    patch_size = 16

    dataset = weird_image_net.eval() if method == 'eval' else weird_image_net.train()
    indices = torch.load(f'{method}.pt')
    indices = indices.view(12, -1)[:, :feat_count]
    model, image_size, _, _ = model_library[34]()
    hook = SaliencyViTGeLUHook(model)

    parent_mask            = BASE_OUTPUT_DIRECTORY / f"{method}_mask"
    parent_mask_normalised = BASE_OUTPUT_DIRECTORY / f"{method}_mask_normalized"
    parent_mask.mkdir(exist_ok=True)
    parent_mask_normalised.mkdir(exist_ok=True)
    for l in tqdm(range(12)):
        for f in range(feat_count):
            img, _ = dataset[indices[l, f]]
            img = img.cuda().unsqueeze(0)
            act = hook(img, l, f).view(1, image_size // patch_size, image_size // patch_size)
            torchvision.utils.save_image(act, (parent_mask            / f'{l}_{f}.png').as_posix(), normalize=False)
            torchvision.utils.save_image(act, (parent_mask_normalised / f'{l}_{f}.png').as_posix(), normalize=True)


if __name__ == '__main__':
    main()
