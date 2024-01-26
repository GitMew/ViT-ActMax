from torchvision.models import resnet18

from vit_actmax.augmentation import Clip, Jitter, Centering, RepeatBatch, Zoom, ColorJitter
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
    network = args.network
    layer, feature = args.layer, args.feature
    model, image_size, batch_size, name = model_library[network]()
    # model = resnet18(pretrained=True).cuda()
    grid = args.grid
    batch_size = min(64, batch_size)

    saver = ExperimentSaver(f'TV{layer}{name}_{grid}_{feature}', save_id=True, disk_saver=True)

    loss = LossArray()

    layer_hook = LayerHook(model, torch.nn.BatchNorm2d, layer, ActivationReluHook)
    loss += LayerActivationNorm(layer_hook, model, coefficient=1)

    resnet_bn = FakeBatchNorm(resnet18, image_net.normalizer).cuda()
    # loss += MatchBatchNorm(resnet_bn, coefficient=0.5 * 0.1)
    # loss += NormalVariation(size=image_size, coefficient=0.05 * 10)
    # loss += ColorVariation(size=image_size, coefficient=0.05 * 10)
    loss += TotalVariation(size=image_size, coefficient=0.05 * 10)

    pre, post = None, torch.nn.Sequential(Clip())
    image = new_init(image_size, batch_size)

    visualizer = ImageNetVisualizer(loss, saver, pre, post, print_every=10, lr=.01, steps=400, save_every=100)
    image.data = visualizer(image).data
    saver.save(image)


if __name__ == '__main__':
    main()
