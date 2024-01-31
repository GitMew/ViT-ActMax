from typing import Union, Tuple
import torch


def new_init(size: Union[int,Tuple[int,int]], batch_size: int=1, last: torch.nn=None, padding: int=-1, zero: bool=False) -> torch.nn:
    if isinstance(size, int):
        size = (size,size)

    height, width = size
    if zero:
        output = torch.zeros(size=(batch_size, 3, height, width))
    else:
        output = torch.rand(size=(batch_size, 3, height, width))
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
