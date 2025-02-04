from torchvision.models import resnet18

from vit_actmax.augmentation import Clip, Jitter, Centering, RepeatBatch, Zoom, ColorJitter
from vit_actmax.datasets import image_net
from vit_actmax.hooks import LayerHook
from vit_actmax.inversion import ImageNetVisualizer
from vit_actmax.inversion.utils import new_init
from vit_actmax.loss import LossArray, TotalVariation, LayerActivationNorm, ActivationHook, MatchBatchNorm
from vit_actmax.loss.regularizers import FakeBatchNorm
from vit_actmax.model import model_library
from vit_actmax.saver import ExperimentSaver
from vit_actmax.utils import exp_starter_pack
import torch


def main():
    exp_name, args, _ = exp_starter_pack()
    network = args.network
    layer, feature = args.layer, args.feature
    model, image_size, batch_size, name = model_library[network]()
    grid = args.grid
    batch_size = min(64, batch_size)

    saver = ExperimentSaver(f'GridSearchTV{layer}{name}_{grid}_{feature}', save_id=True, disk_saver=True)

    loss = LossArray()

    layer_hook = LayerHook(model, torch.nn.BatchNorm2d, layer, ActivationHook)
    loss += LayerActivationNorm(layer_hook, model, coefficient=1)

    resnet_bn = FakeBatchNorm(resnet18, image_net.normalizer).cuda()
    # loss += MatchBatchNorm(resnet_bn, coefficient=50.0 * c)
    loss += TotalVariation(size=image_size, coefficient=5.0 * grid)

    pre, post = None, torch.nn.Sequential(Clip())
    image = new_init(image_size, batch_size)

    visualizer = ImageNetVisualizer(loss, None, pre, post, print_every=10, lr=.01, steps=400, save_every=100)
    image.data = visualizer(image).data
    saver.save(image[:4])


if __name__ == '__main__':
    main()
