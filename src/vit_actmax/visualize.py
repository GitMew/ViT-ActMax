import torch
import torchvision.transforms

from .augmentation import Clip, Tile, Jitter, RepeatBatch, ColorJitter
from .augmentation.pre import GaussianNoise
from .hooks.transformer.vit import ViTAttHookHolder, ViTGeLUHook, GELUHOOK_RESULT_NAME
from .inversion import ImageNetVisualizer
from .inversion.utils import new_init
from .loss import LossArray, TotalVariation
from .loss.image_net import ViTFeatHook, ViTEnsFeatHook
from .model import model_library
from .saver import ExperimentSaver
from .utils import exp_starter_pack


def visualize_feature(layer, feature):
    layer, feature = int(layer), int(feature)
    print('asghar *******************************')
    print(layer, feature)
    args = exp_starter_pack()[1]
    layer, feature, grid = layer, feature, args.grid
    network = 38
    print(model_library)
    model, image_size, _, _ = model_library[network]()
    model.eval()
    model.requires_grad_(False)
    tv = args.tv

    saver = ExperimentSaver(f'VisL{layer}_F{feature}_N{network}_TV{tv}', save_id=True, disk_saver=True)

    loss = LossArray()
    loss += ViTEnsFeatHook(ViTGeLUHook(model, sl=slice(layer, layer + 1)), key=GELUHOOK_RESULT_NAME, feat=feature, coefficient=1)
    loss += TotalVariation(2, image_size, coefficient=0.0005 * tv)

    pre, post = torch.nn.Sequential(RepeatBatch(8), ColorJitter(8, shuffle_every=True),
                                    GaussianNoise(8, True, 0.5, 400), Tile(1), Jitter()), Clip()

    image = new_init(image_size, 1)
    visualizer = ImageNetVisualizer(loss, None, pre, post, print_every=10, lr=0.1, steps=400, save_every=1000000)
    image.data = visualizer(image)
    # saver.save(image, 'final')
    image = torchvision.transforms.ToPILImage()(image[0].detach().cpu())
    return image


if __name__ == '__main__':
    visualize_feature(3, 4)
