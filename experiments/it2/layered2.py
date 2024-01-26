from torchvision.models import resnet18
import pdb

from vit_actmax.augmentation import Clip, Jitter, Centering, RepeatBatch, Zoom, ColorJitter
from vit_actmax.augmentation.post import ClipSTD
from vit_actmax.augmentation.pre import Layered
from vit_actmax.datasets import image_net
from vit_actmax.hooks import LayerHook
from vit_actmax.inversion import ImageNetVisualizer
from vit_actmax.inversion.utils import new_init
from vit_actmax.loss import LossArray, TotalVariation, LayerActivationNorm, ActivationHook, MatchBatchNorm, NormalVariation
from vit_actmax.loss import ColorVariation
from vit_actmax.loss.hooks.activation import ActivationReluHook
from vit_actmax.loss.regularizers import FakeBatchNorm
from vit_actmax.model import model_library
from vit_actmax.saver import ExperimentSaver
from vit_actmax.utils import exp_starter_pack
import torch

from torchvision.models import resnet18


def main():
    exp_name, args, _ = exp_starter_pack()
    network, layer, feature = args.network, args.layer, args.feature
    model, image_size, batch_size, name = model_library[network]()
    layer_hook = LayerHook(model, torch.nn.BatchNorm2d, layer, ActivationReluHook)
    # layer_hook = LayerHook(model, torch.nn.Dropout, layer, ActivationReluHook)
    batch_size = min(64, batch_size)

    saver = ExperimentSaver(f'LayeredStable{layer}{name}_{feature}', save_id=True, disk_saver=True)

    clipper = ClipSTD()

    image = new_init(image_size, batch_size, last=None, zero=True)

    pre, post = Layered(0), Clip()

    for i in range(10):
        loss = LossArray()
        loss += LayerActivationNorm(layer_hook, model, coefficient=1)
        loss += TotalVariation(size=image_size, coefficient=0.005 * 10)
        visualizer = ImageNetVisualizer(loss, None, pre, post, print_every=10, lr=.01, steps=20)
        image.data = visualizer(image).data
        saver.save(image, 'noise')
        delta = clipper(image, 0.1)
        total = pre(delta).detach().clone()
        saver.save(total, 'total')
        pre.x = total.detach().clone() + pre.x
        image = new_init(image_size, batch_size, last=None, zero=True)


if __name__ == '__main__':
    main()
