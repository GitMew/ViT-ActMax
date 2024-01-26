import os
from typing import Any
from pathlib import Path

import torch
import torchvision

from ..utils.experiment import _get_exp_name


BASE_OUTPUT_DIRECTORY = Path(os.getcwd()) / "_output-actmax"  # Used to be called "desktop", but... who does that?
BASE_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)


class AbstractSaver:
    def __init__(self, extension: str, folder: str = None, save_id: bool = False):
        subfolder = Path(_get_exp_name() if folder is None else folder)
        self.subfolder = subfolder / (extension + "_" + self.__class__.__name__)
        self._id = 0
        self.extension = extension
        self.save_id = save_id

    def _get_mkdir_path(self, *path):
        stem = "_".join(map(str, path))
        if self.save_id:
            stem = f"{self._id}_" + stem
        parent = BASE_OUTPUT_DIRECTORY / self.subfolder
        parent.mkdir(exist_ok=True)
        return (parent / stem).with_suffix(self.extension).as_posix()

    def save(self, result: torch.Tensor, *path):
        full_path = self._get_mkdir_path(*path)
        print("Saving to", full_path)
        self.save_function(result, full_path)
        self._id += 1

    def save_function(self, result: Any, path: str):
        raise NotImplementedError


class ImageSaver(AbstractSaver):
    def save_function(self, result: Any, path: str):
        # torchvision.utils.save_image(result, path, nrow=self.nrow[len(result) - 1])
        torchvision.utils.save_image(result, path, nrow=1)

    @staticmethod
    def get_nrow() -> torch.tensor:
        bs = torch.arange(1, 1000)
        p = bs.view(1, -1).repeat(len(bs), 1)
        q = (bs.view(-1, 1) / p).floor().int()
        feasible = ((p * q) == bs.view(-1, 1))
        sm = p + q
        sm[feasible == False] = (len(bs) + 1) * 10
        return p[torch.arange(len(bs)), sm.argmin(dim=-1)]

    def __init__(self, folder: str, save_id: bool = False):
        super().__init__('png', folder, save_id)
        self.nrow = self.get_nrow()


class TensorSaver(AbstractSaver):
    def save_function(self, result: Any, path: str):
        torch.save(result, path)

    def __init__(self, folder: str, save_id: bool = False):
        super().__init__('pth', folder, save_id)


class ExperimentSaver(AbstractSaver):
    def save_function(self, result: Any, path: str):
        pass

    def save(self, result: torch.Tensor, *path):
        self.image.save(result, *path)
        if not self.disk_saver:
            self.tensor.save(result, *path)

    def __init__(self, folder: str, save_id: bool = False, disk_saver: bool = False):
        super().__init__('none', folder, save_id)
        self.image = ImageSaver(folder=folder, save_id=save_id)
        self.disk_saver = disk_saver
        self.tensor = TensorSaver(folder=folder, save_id=save_id)


""" 


 sq = int(np.ceil(np.sqrt(batch_size)) + 1)
        p = torch.arange(1, sq)
        squares = (((batch_size / p).ceil().int() * p) == batch_size)
        xs = p[squares][-1].item()
        ys = batch_size // xs

"""
