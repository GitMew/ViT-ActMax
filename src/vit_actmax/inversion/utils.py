from typing import Union, Tuple, Callable
import torch


ImageBatchCreator = Callable[[int,int,int,int], torch.Tensor]  # batch size, height, width, patch size -> Tensor[B x C x H x W]

random_batch: ImageBatchCreator = lambda batch_size,height,width,_: torch.rand(size=(batch_size, 3, height, width))
zero_batch: ImageBatchCreator   = lambda batch_size,height,width,_: torch.zeros(size=(batch_size, 3, height, width))


def new_init(size: Union[int,Tuple[int,int]], batch_size: int=1, patch_size: int=16, last: torch.nn=None, padding: int=-1,
             base: ImageBatchCreator=random_batch) -> torch.Tensor:
    if isinstance(size, int):
        size = (size,size)

    height, width = size
    output = base(batch_size, height, width, patch_size)
    # output += 0.5
    output = output.cuda()
    if last is not None:
        smaller_height = height if padding == -1 else height - padding
        smaller_width  = width if padding == -1 else width - padding

        up = torch.nn.Upsample(size=(smaller_height, smaller_width), mode='bilinear', align_corners=False).cuda()
        scaled_last = up(last)

        offset_height = (height - smaller_height) // 2
        offset_width  = (width - smaller_width) // 2
        output[:, :, offset_height:offset_height+smaller_height, offset_width:offset_width+smaller_width] = scaled_last

    output = output.detach().clone()
    output.requires_grad_()
    return output
