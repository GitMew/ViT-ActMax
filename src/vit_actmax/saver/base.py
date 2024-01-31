import os
from typing import Any
from pathlib import Path

import torch
import torchvision

from ..utils.experiment import _get_exp_name


BASE_OUTPUT_DIRECTORY = Path(os.getcwd()) / "_output-actmax"  # Used to be called "desktop", but... who does that?
BASE_OUTPUT_DIRECTORY.mkdir(exist_ok=True)


class AbstractSaver:

    def __init__(self, extension: str, folder: str=None, save_id: bool=False):
        subfolder = Path(_get_exp_name() if folder is None else folder)
        self.subfolder = subfolder #/ (extension + "_" + self.__class__.__name__)
        self._id = 0
        self.extension = extension
        self.save_id = save_id

    def _get_mkdir_path(self, *path):
        stem = "_".join(map(str, path))
        if self.save_id:
            stem += f"_{self._id}"
        parent = BASE_OUTPUT_DIRECTORY / self.subfolder
        parent.mkdir(exist_ok=True, parents=True)
        return (parent / stem).with_suffix("." + self.extension).as_posix()

    def save(self, result: torch.Tensor, *path):
        full_path = self._get_mkdir_path(*path)
        print("Saving to", full_path)
        self.save_function(result, full_path)
        self._id += 1

    def save_function(self, result: Any, path: str):
        raise NotImplementedError


class ImageSaver(AbstractSaver):

    def __init__(self, folder: str, save_id: bool = False):
        super().__init__('png', folder, save_id)
        self.nrow = ImageSaver.get_nrow()

    def save_function(self, result: Any, path: str):
        stack_batch_vertically = True
        if stack_batch_vertically:
            images_per_row = 1
        else:
            images_per_row = self.nrow[len(result) - 1]  # Not sure what self.nrow is, but whatever.

        # save_image saves a batch of images by putting them on a grid.
        # Empty grid positions are filled with pad values; there can also be padding between the grid tiles.
        torchvision.utils.save_image(result, path, nrow=images_per_row, padding=0, pad_value=1.0)

    @staticmethod
    def get_nrow() -> torch.tensor:
        bs = torch.arange(1, 1000)
        p = bs.view(1, -1).repeat(len(bs), 1)
        q = (bs.view(-1, 1) / p).floor().int()
        feasible = ((p * q) == bs.view(-1, 1))
        sm = p + q
        sm[feasible == False] = (len(bs) + 1) * 10
        return p[torch.arange(len(bs)), sm.argmin(dim=-1)]


class ReshapeImageSaver(ImageSaver):

    def __init__(self, folder: str, patch_size: int, patches_per_row: int, save_id=False):
        super().__init__(folder, save_id)
        self.patch_size = patch_size
        self.patches_per_row = patches_per_row

    def save_function(self, result: Any, path: str):
        """
        Reshape to a grid of patches_per_row patches of size patch_size x patch_size per row.

        If the amount of patches in the image is not a multiple of a whole multiple of patches_per_row, white patches
        are inserted to fill up the rest.

        If the input contains multiple batches, they are stacked on top of each other, unless they would be padded with
        white patches, in which case (sadly) patches of one batch example will take their place.
        """
        # Crop image to fit exactly the patch size (you could also pad with white pixels, but I'm too stupid for that)
        _, C, H, W = result.shape
        assert H % self.patch_size == 0
        assert W % self.patch_size == 0

        result = result.unfold(2, self.patch_size, self.patch_size)\
                       .unfold(3, self.patch_size, self.patch_size)\
                       .permute(0, 2, 3, 1, 4, 5)\
                       .flatten(start_dim=0, end_dim=2)
        # What this does:
        #   - Incoming is a batch of dimensions batches x colour x height x width.
        #   - unfold(dimension, size, step) with size == step basically splits into chunks of that size along that dimension. If you do it once, you have strips. If you do it twice, you have patches.
        #   - permute swaps dimensions. In this case, we move colour from the front to the middle: batches x (y_patches x x_patches) x (colour x y_pixels x x_pixels).
        #   - flatten is used for serialising 2D, 3D, ... into 1D. What comes out is (batches * patches) x colour x pixel_height x pixel_width.

        # Now that we have a list of patches, we abuse save_image's call to make_grid, which thinks that we are passing in big, distinct images.
        torchvision.utils.save_image(result, path, padding=0, pad_value=1.0, nrow=self.patches_per_row)


class TensorSaver(AbstractSaver):
    def save_function(self, result: Any, path: str):
        torch.save(result, path)

    def __init__(self, folder: str, save_id: bool = False):
        super().__init__('pth', folder, save_id)


class ExperimentSaver(AbstractSaver):

    def __init__(self, folder: str, save_id: bool = False, disk_saver: bool = False):
        super().__init__('none', folder, save_id)
        self.imagesaver  =  ImageSaver(folder=folder, save_id=save_id)
        self.tensorsaver = TensorSaver(folder=folder, save_id=save_id)
        self.save_to_disk = disk_saver

    def save_function(self, result: Any, path: str):
        pass

    def save(self, result: torch.Tensor, *path):
        self.imagesaver.save(result, *path)
        if not self.save_to_disk:
            self.tensorsaver.save(result, *path)


""" 


 sq = int(np.ceil(np.sqrt(batch_size)) + 1)
        p = torch.arange(1, sq)
        squares = (((batch_size / p).ceil().int() * p) == batch_size)
        xs = p[squares][-1].item()
        ys = batch_size // xs

"""
