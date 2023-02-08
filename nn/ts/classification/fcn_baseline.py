import torch
from torch import nn
from torch.autograd import Variable
from nn.common import VecMap

from nn.nalu import NeuralArithmeticLogicUnit, NeuralArithmeticLogicUnitCell

from .utils import ConvBlock


class FCNBaseline(nn.Module):
    """A PyTorch implementation of the FCN Baseline
    From https://arxiv.org/abs/1909.04939

    Attributes
    ----------
    sequence_length:
        The size of the input sequence
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, in_channels: int, num_pred_classes: int = 1) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'in_channels': in_channels,
            'num_pred_classes': num_pred_classes
        }

        self.layers = nn.Sequential(*[
            # ConvBlock(in_channels, 128, 8, 1),
            ConvBlock(in_channels, 1024, 7, 1),
            ConvBlock(1024, 512, 3, 1),
        ])
        
        self.final = nn.Linear(512, num_pred_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.layers(x)
        return self.final(x.mean(dim=-1))

from nn.nac import NeuralAccumulatorCell
from nn.nalu import NeuralArithmeticLogicUnit

class FCNNaccBaseline(nn.Module):
    def __init__(self, in_channels:int, num_pred_classes:int=1) -> None:
        super().__init__()
        
        # for easier saving and loading
        self.input_args = {
            'in_channels': in_channels,
            'num_pred_classes': num_pred_classes
        }

        self.fcn_encode = VecMap(nn.Sequential(*[
            ConvBlock(in_channels, 128, 8, 1),
            ConvBlock(128, 256, 5, 1),
            ConvBlock(256, 128, 3, 1),
        ]))
        
        self.nacc_decode = VecMap(nn.Sequential(
            NeuralArithmeticLogicUnitCell(128, 128),
            NeuralArithmeticLogicUnitCell(128, 16),
            # NeuralArithmeticLogicUnitCell(5, num_pred_classes),
        ), output_shape=(None, 16))
        
        self.final = nn.Linear(16, num_pred_classes)
        # self.final = nn.LazyLinear(num_pred_classes)
        
    def parameters(self, recurse: bool = True):
        return (
            list(super().parameters(recurse))+
            list(self.fcn_encode.parameters())+
            list(self.nacc_decode.parameters())
        )
        
    def forward(self, inputs:torch.Tensor):
        x = self.fcn_encode(inputs.unsqueeze(1)).squeeze()
        outputs = Variable(torch.zeros((x.size(0), self.input_args['num_pred_classes'])))
        
        for i in range(x.size(0)):
            _x = self.nacc_decode(x.mean(dim=-1))
            
            y = self.final(_x)
            
            print(i, y.shape)
            outputs[i, :] = y
        
        return outputs