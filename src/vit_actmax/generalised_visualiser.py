"""
Generalised visualiser based on
    experiments/it5/best.py
    experiments/it15/vis35.py
    visualize.py
"""
from typing import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
import torchvision

from .augmentation import Clip, Tile, Jitter, RepeatBatch, ColorJitter
from .augmentation.pre import GaussianNoise
from .hooks.transformer.vit import ViTAttHookHolder, ViTGeLUHook, ViTAbsHookHolder
from .inversion import ImageNetVisualizer
from .inversion.utils import *
from .loss import LossArray, TotalVariation
from .loss.image_net import ViTFeatHook, ViTEnsFeatHook
from .model import model_library as hundred_standard_models
from .model.library import ModelLibrary
from .saver.base import ExperimentSaver, ReshapeImageSaver, ImageSaver
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

    @staticmethod
    def fromArgs(args) -> "OptimisationTarget":
        return OptimisationTarget(
            network_identifier=args.network,
            layer_index=args.layer,
            choose_among=VectorFamily.fromString(args.method),
            element_index=args.feature
        )


@dataclass
class Hyperparameters:
    regularisation_constant: float
    sign: int
    iterations: int
    learning_rate: float

    @staticmethod
    def fromArgs(args) -> "Hyperparameters":
        return Hyperparameters(
            regularisation_constant=args.tv,
            sign=args.sign,
            iterations=args.iterations,
            learning_rate=args.lr
        )


@dataclass
class OutputSettings:
    make_image_square: bool
    log_every: int
    save_every: int

    @staticmethod
    def fromArgs(args) -> "OutputSettings":
        return OutputSettings(
            make_image_square=args.output_square,
            log_every=args.log_every,
            save_every=args.save_every
        )


HookConstructor = Callable[[torch.nn.Module, slice, str], ViTAbsHookHolder]
vit_ffnn_hooker = lambda model, sl, method: ViTGeLUHook(model, sl=sl)
vit_att_hooker  = lambda model, sl, method: ViTAttHookHolder(model, sl=sl, **{method: True})


@dataclass
class Constructors:
    model_library: ModelLibrary=hundred_standard_models
    make_ffnn_hook: HookConstructor=vit_ffnn_hooker
    make_att_hook: HookConstructor=vit_att_hooker
    make_image_base: ImageBatchCreator=random_batch


###################
### Interpreter ###
###################
def maximiseActivation(t: OptimisationTarget, h: Hyperparameters, v: OutputSettings, c: Constructors):
    """
    The most generally applicable activation maximisation function possible.
    Takes a target to maximise, hooks into the parts of the model that track this target, and saves the resulting image.

    No reference is made to the command line. Also, this function switches between attention and FFNN hooks automatically,
    rather than forcing the user to maximise an element of 1 single family of vectors. Finally, to support adding custom
    models, both the hook constructors as well as the model library have become paramters.
    """
    # Look up the model
    model, image_size, _, model_name = c.model_library[t.network_identifier]()
    sl = slice(t.layer_index, t.layer_index+1)
    method = t.choose_among.toString()
    try:
        patch_size = model.config.patch_size
    except:
        patch_size = 16

    # Make a hook that will track all activations of a certain family in a certain layer.
    if t.choose_among == VectorFamily.FFNN_HIDDEN_ACTIVATION:
        hook = c.make_ffnn_hook(model, sl, method)
    else:
        hook = c.make_att_hook(model, sl, method)

    # As far as I understand it, a hook produces a dictionary with interesting results, which is extracted by a FeatHook based on the given key.
    loss = LossArray()
    loss += ViTEnsFeatHook(hook, key=method, feat=t.element_index, coefficient=1 * h.sign)
    loss += TotalVariation(2, image_size, coefficient=0.0005 * h.regularisation_constant)

    # Visualiser setup
    post = Clip()
    pre = torch.nn.Sequential(
        RepeatBatch(8),
        ColorJitter(8, shuffle_every=True),
        GaussianNoise(8, True, 0.5, 400),
        Tile(1),
        Jitter()
    )

    folder = f"{t.network_identifier}_" + model_name.replace("/", "--")
    stem   = f"{t.choose_among.toString()}-L{str(t.layer_index).zfill(2)}-F{str(t.element_index).zfill(4)}-TV{h.regularisation_constant}"
    if not isinstance(image_size, int) and v.make_image_square:  # If the image size isn't an int, it's a rectangle. Rectangular images can be reshaped into a square after generation.
        n_patches             = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        patch_grid_sidelength = int(np.ceil(np.sqrt(n_patches)))
        saver = ReshapeImageSaver(folder, patch_size=patch_size, patches_per_row=patch_grid_sidelength, save_id=False)
    else:
        saver = ImageSaver(folder, save_id=False)

    visualizer = ImageNetVisualizer(loss, saver, pre, post, lr=h.learning_rate, steps=h.iterations,
                                    save_every=v.save_every, print_every=v.log_every)

    # Make image
    image = new_init(image_size, batch_size=1, patch_size=patch_size, base=c.make_image_base)
    image.data = visualizer(image, file_prefix=stem)
    image = torchvision.transforms.ToPILImage()(image[0].detach().cpu())
    return image


def main():  # Import this from somewhere to run it.
    args = exp_starter_pack()[1]
    maximiseActivation(
        OptimisationTarget.fromArgs(args),
        Hyperparameters.fromArgs(args),
        OutputSettings.fromArgs(args),
        Constructors()
    )
