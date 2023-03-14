import torch
from torch import nn

from .utils import ConvBlock, Conv1dSamePadding


class ResNetBaseline(nn.Module):
    """A PyTorch implementation of the ResNet Baseline
    From https://arxiv.org/abs/1909.04939

    Attributes
    ----------
    sequence_length:
        The size of the input sequence
    mid_channels:
        The 3 residual blocks will have as output channels:
        [mid_channels, mid_channels * 2, mid_channels * 2]
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, in_channels: int, mid_channels: int = 3,
                 num_pred_classes: int = 1) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'in_channels': in_channels,
            'num_pred_classes': num_pred_classes
        }

        block_channels = [
            (in_channels, mid_channels*2),
        ]
        n_hidden_blocks = 2
        block_channels += [(mid_channels*2, mid_channels*2)] * n_hidden_blocks
        block_channels += [(mid_channels*2, mid_channels)]
        
        self.network_blocks = [ResNetBlock(in_channels=in_channels, out_channels=out_channels) for in_channels, out_channels in block_channels]
        
        self.head = nn.Sequential(*self.network_blocks)
        self.tail = nn.Linear(block_channels[-1][1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.head(x)
        x = x.mean(dim=-1)
        y = self.tail(x)
        
        return y

def no_op(x: torch.Tensor) -> torch.Tensor:
    return x

class ResNetBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        channels = [in_channels, out_channels, out_channels, out_channels]
        kernel_sizes = [8, 5, 3]

        self.layers = nn.Sequential(*[
            ConvBlock(in_channels=channels[i], out_channels=channels[i + 1],
                      kernel_size=kernel_sizes[i], stride=1) for i in range(len(kernel_sizes))
        ])

        self.match_channels = False
        
        if in_channels != out_channels:
            self.match_channels = True
            self.residual = nn.Sequential(*[
                Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=1, stride=1),
                nn.BatchNorm1d(num_features=out_channels)
            ])
        else:
            self.residual = no_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        inputs:torch.Tensor = x#torch.clone(x)
        
        if self.match_channels:
            _x = self.layers(inputs) + self.residual(inputs)
            # _x = (_x + self.residual(inputs)
        else:
            _x = self.layers(inputs)
        
        return _x
