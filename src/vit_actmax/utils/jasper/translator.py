import torch

# FIXME: Code so old the dependencies don't even exist anymore
from ...datasets.audio.librispeech.nvidia.manifest import Manifest
from ...model.jasper import GreedyCTCDecoder


class Translator:
    def __init__(self, converter: Manifest):
        self.decoder = GreedyCTCDecoder()
        self.converter = converter

    def __call__(self, prediction: torch.Tensor, length: torch.Tensor) -> [str]:
        chars = self.decoder(prediction)
        return self.converter.reverse(chars, True, length)
