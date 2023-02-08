import torch
from torch import nn
import torch.nn.functional as F
from torch.types import _int, _float, _bool, Number, _dtype, _device, _qscheme, _size
from torch import Tensor

from typing import *

class Conv1dSamePadding(nn.Conv1d):
    stride:int
    # dilation
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """
    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride, self.dilation, self.groups)

def twrap(x: Union[_int, _size])->_size:
    if isinstance(x, int):
        return (x,)
    return x

def conv1d_same_padding(input: Tensor, weight: Tensor, bias: Optional[Tensor]=None, _stride:Tuple[int]=(1,), _dilation:Tuple[int]=(1,), groups: _int=1) -> Tensor:
    
    # stride and dilation are expected to be tuples.
    kernel, dilation, stride = weight.size(2), _dilation[0], _stride[0]
    l_out = l_in = input.size(2)
    padding = ((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                    padding=padding // 2,
                    dilation=dilation, groups=groups)


class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            Conv1dSamePadding(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        return self.layers(x)
