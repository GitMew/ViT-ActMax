"""
Generalised visualiser based on
    experiments/it5/best.py
    experiments/it15/vis35.py
    visualize.py
"""
from typing import Callable
from dataclasses import dataclass
from enum import Enum

import torch
import torchvision

from .augmentation import Clip, Tile, Jitter, RepeatBatch, ColorJitter
from .augmentation.pre import GaussianNoise
from .hooks.transformer.vit import ViTAttHookHolder, ViTGeLUHook, ViTAbsHookHolder
from .inversion import ImageNetVisualizer
from .inversion.utils import new_init
from .loss import LossArray, TotalVariation
from .loss.image_net import ViTFeatHook, ViTEnsFeatHook
from .model import model_library as hundred_standard_models
from .model.library import ModelLibrary
from .saver import ExperimentSaver
from .utils import exp_starter_pack


#################
### Interface ###
#################
class VectorFamily(Enum):
    ATTENTION_INPUT   = 1 # Equal to FFNN output of the previous layer
    ATTENTION_KEYS    = 2
    ATTENTION_QUERIES = 3
    ATTENTION_VALUES  = 4
    ATTENTION_OUTPUT  = 5 # Equal to FFNN input
    FFNN_HIDDEN_ACTIVATION = 6

    def toString(self) -> str:
        return FAMILY_NAMES[self]

    @staticmethod
    def fromString(s: str) -> "VectorFamily":
        reverse_mapping = {v:k for k,v in FAMILY_NAMES.items()}
        return reverse_mapping[s]

FAMILY_NAMES = {
    VectorFamily.ATTENTION_INPUT: "in_feat",
    VectorFamily.ATTENTION_KEYS: "keys",
    VectorFamily.ATTENTION_QUERIES: "queries",
    VectorFamily.ATTENTION_VALUES: "values",
    VectorFamily.ATTENTION_OUTPUT: "out_feat",
    VectorFamily.FFNN_HIDDEN_ACTIVATION: "ffnn"
}


@dataclass
class OptimisationTarget:
    network_identifier: int
    layer_index: int
    choose_among: VectorFamily
    element_index: int
    regularisation_constant: float

    @staticmethod
    def fromArgs():
        args = exp_starter_pack()[1]
        return OptimisationTarget(
            network_identifier=args.network,
            layer_index=args.layer,
            choose_among=VectorFamily.fromString(args.method),
            element_index=args.feature,
            regularisation_constant=args.tv
        )


HookConstructor = Callable[[torch.nn.Module, slice, str], ViTAbsHookHolder]
vit_ffnn_hooker = lambda model, sl, method: ViTGeLUHook(model, sl=sl)
vit_att_hooker  = lambda model, sl, method: ViTAttHookHolder(model, sl=sl, **{method: True})


###################
### Interpreter ###
###################
def maximiseActivation(t: OptimisationTarget, model_library: ModelLibrary=hundred_standard_models,
                       make_ffnn_hook: HookConstructor=vit_ffnn_hooker, make_att_hook: HookConstructor=vit_att_hooker):
    """
    The most generally applicable activation maximisation function possible.
    Takes a target to maximise, hooks into the parts of the model that track this target, and saves the resulting image.

    No reference is made to the command line. Also, this function switches between attention and FFNN hooks automatically,
    rather than forcing the user to maximise an element of 1 single family of vectors. Finally, to support adding custom
    models, both the hook constructors as well as the model library have become paramters.
    """
    # Look up the model
    model, image_size, _, _ = model_library[t.network_identifier]()
    sl = slice(t.layer_index, t.layer_index+1)
    method = t.choose_among.toString()

    # Make a hook that will track all activations of a certain family in a certain layer.
    if t.choose_among == VectorFamily.FFNN_HIDDEN_ACTIVATION:
        hook = make_ffnn_hook(model, sl, method)
    else:
        hook = make_att_hook(model, sl, method)

    # As far as I understand it, a hook produces a dictionary with interesting results, which is extracted by a FeatHook based on the given key.
    loss = LossArray()
    loss += ViTEnsFeatHook(hook, key=method, feat=t.element_index, coefficient=1)
    loss += TotalVariation(2, image_size, coefficient=0.0005 * t.regularisation_constant)

    # Visualiser setup
    pre, post = torch.nn.Sequential(RepeatBatch(8), ColorJitter(8, shuffle_every=True),
                                    GaussianNoise(8, True, 0.5, 400), Tile(1), Jitter()), Clip()
    saver = ExperimentSaver(folder=f"actmax_network{t.network_identifier}", save_id=True, disk_saver=True)
    visualizer = ImageNetVisualizer(loss, saver, pre, post, print_every=10, lr=0.1, steps=400, save_every=100)

    # Make image
    image = new_init(image_size, batch_size=1)
    image.data = visualizer(image, file_prefix=f"{t.choose_among} L{t.layer_index} F{t.element_index} TV{t.regularisation_constant}")
    image = torchvision.transforms.ToPILImage()(image[0].detach().cpu())
    return image


def main():
    maximiseActivation(OptimisationTarget.fromArgs())


if __name__ == "__main__":  # Needs to be run with command-line arguments.
    main()