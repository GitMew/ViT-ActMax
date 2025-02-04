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
    args = exp_starter_pack()[1]
    layer, feature, method, network = args.layer, args.feature, args.method, args.network
    coef, lr = args.sign, args.lr
    tv = 0.0005
    grid = args.tv
    model, image_size, _, _ = model_library[network]()

    saver = ExperimentSaver(f'LRL{layer}F{feature}_{tv}x{grid}_LR{lr}_N{network}_M{method}_S{coef}', save_id=True,
                            disk_saver=True)

    loss = LossArray()
    loss += ViTEnsFeatHook(ViTAttHookHolder(model, sl=slice(layer, layer + 1), **{method: True}), key=method,
                           feat=feature, coefficient=1 * coef)
    loss += TotalVariation(2, image_size, coefficient=tv * grid)

    pre, post = torch.nn.Sequential(RepeatBatch(8), ColorJitter(8, shuffle_every=True),
                                    GaussianNoise(8, True, 0.5, 400), Tile(1), Jitter()), Clip()
    image = new_init(image_size, 1)

    visualizer = ImageNetVisualizer(loss, None, pre, post, print_every=10, lr=lr, steps=400, save_every=100)
    image.data = visualizer(image)
    saver.save(image, 'final')


if __name__ == '__main__':
    main()
