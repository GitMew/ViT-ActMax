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
    feat_count = 128
    patch_size = 32
    network = 98
    layer, feature = args.layer, args.feature
    method = 'eval' if args.grid == 1.0 else 'train'
    dataset = image_net.eval() if method == 'eval' else image_net.train()
    indices = torch.load(f'{method}_{layer}_{feature}.pt')
    model, image_size, _, _ = model_library[network]()
    hook = SaliencyClipGeLUHook(model)

    parent_natural    = BASE_OUTPUT_DIRECTORY / f"{method}_{layer}_{feature}"
    parent_normalised = BASE_OUTPUT_DIRECTORY / f"{method}_{layer}_{feature}_mask"
    parent_natural.mkdir(exist_ok=True)
    parent_normalised.mkdir(exist_ok=True)
    for v, i in tqdm(indices):
        img, _ = dataset[i]
        img = img.cuda().unsqueeze(0)
        act = hook(img, layer, feature).vpiew(1, image_size // patch_size, image_size // patch_size)
        torchvision.utils.save_image(img, (parent_natural    / f'{i}.png').as_posix())
        torchvision.utils.save_image(act, (parent_normalised / f'{i}.png').as_posix(), normalize=True)


if __name__ == '__main__':
    main()
