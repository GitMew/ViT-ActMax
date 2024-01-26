import os
import pdb

import torchvision.utils
from torch.utils.data import DataLoader
from vit_actmax.datasets import weird_image_net, image_net
from vit_actmax.hooks.transformer.vit import SimpleViTGeLUHook, SaliencyViTGeLUHook, SaliencyClipGeLUHook
from vit_actmax.model import model_library
from vit_actmax.utils import exp_starter_pack
from vit_actmax.saver.base import BASE_OUTPUT_DIRECTORY

import torch
from tqdm import tqdm


@torch.no_grad()
def main():
    exp_name, args, _ = exp_starter_pack()
    method = 'eval'
    method = 'train'
    feat_count = 60
    patch_size = 16

    dataset = image_net.eval() if method == 'eval' else image_net.train()
    indices = torch.load(f'clip_{method}.pt')
    indices = indices.view(12, -1)[:, :feat_count]
    model, image_size, _, _ = model_library[99]()
    hook = SaliencyClipGeLUHook(model)

    parent_natural         = BASE_OUTPUT_DIRECTORY / "clip" / f"{method}"
    parent_mask            = BASE_OUTPUT_DIRECTORY / "clip" / f"{method}_mask"
    parent_mask_normalised = BASE_OUTPUT_DIRECTORY / "clip" / f"{method}_mask_normalized"
    parent_natural.mkdir(exist_ok=True, parents=True)
    parent_mask.mkdir(exist_ok=True, parents=True)
    parent_mask_normalised.mkdir(exist_ok=True, parents=True)
    for l in tqdm(range(12)):
        for f in range(feat_count):
            img, _ = dataset[indices[l, f]]
            img = img.cuda().unsqueeze(0)
            act = hook(img, l, f).view(1, image_size // patch_size, image_size // patch_size)
            torchvision.utils.save_image(img, os.path.join(parent_natural,         f'{l}_{f}.png'))
            torchvision.utils.save_image(act, os.path.join(parent_mask,            f'{l}_{f}.png'), normalize=False)
            torchvision.utils.save_image(act, os.path.join(parent_mask_normalised, f'{l}_{f}.png'), normalize=True)


if __name__ == '__main__':
    main()
