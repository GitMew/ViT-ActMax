import torch
from vit_actmax.augmentation import Clip, Tile, Jitter, RepeatBatch, ColorJitter
from vit_actmax.augmentation.pre import GaussianNoise
from vit_actmax.hooks.transformer.vit import ViTAttHookHolder
from vit_actmax.inversion import ImageNetVisualizer
from vit_actmax.inversion.utils import new_init
from vit_actmax.loss import LossArray, TotalVariation
from vit_actmax.loss.image_net import ViTFeatHook, ViTEnsFeatHook
from vit_actmax.model import model_library
from vit_actmax.saver import ExperimentSaver
from vit_actmax.utils import exp_starter_pack


def main():
    exp_name, args, _ = exp_starter_pack()
    # layer, method, network, patch, coef = args.layer, args.method, args.network, args.patch, float(args.sign)
    layer, feature, grid = args.layer, args.feature, args.grid
    method = args.method
    network, coef, patch = 34, +1, 384
    tv = 0.0005
    model, image_size, _, _ = model_library[network]()

    saver = ExperimentSaver(f'GridTVL{layer}F{feature}_{tv}x{grid}_N{network}_M{method}_S{coef}_P{patch}',
                            save_id=True, disk_saver=True)

    loss = LossArray()
    loss += ViTEnsFeatHook(ViTAttHookHolder(model, sl=slice(layer, layer + 1), **{method: True}), key=method, feat=feature,
                           coefficient=1 * coef)
    loss += TotalVariation(2, 384, coefficient=tv * grid)

    pre, post = torch.nn.Sequential(RepeatBatch(8), ColorJitter(8, shuffle_every=True),
                                    GaussianNoise(8, True, 0.5, 400), Tile(image_size // patch), Jitter()), Clip()
    image = new_init(patch, 1)

    visualizer = ImageNetVisualizer(loss, saver, pre, post, print_every=10, lr=.1, steps=400, save_every=100)
    image.data = visualizer(image)
    saver.save(image, 'final')


if __name__ == '__main__':
    main()
