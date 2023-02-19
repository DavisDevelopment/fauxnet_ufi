import torch
from torch import nn, zeros
from torch.autograd import Variable
from nn.common import VecMap

from nn.nalu import NeuralArithmeticLogicUnit, NeuralArithmeticLogicUnitCell

from .utils import ConvBlock, ConvBlock2D


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
        
        self.input_shape = (None, in_channels, None)
        self.output_shape = (None, num_pred_classes)

        self.layers = nn.Sequential(*[
            ConvBlock(in_channels, 256, 5, 1),
            ConvBlock(256, 256, 4, 1),
            ConvBlock(256, 256, 4, 1),
            ConvBlock(256, 128, 3, 1)
        ])
        
        self.final = nn.Linear(128, num_pred_classes)
        
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        input_shape = x.shape
        batch_len, n_channels, seq_len = input_shape
        assert x.ndim == len(self.input_shape)
        
        # y = Variable(zeros((batch_len, self.output_shape[1]), requires_grad=True))
        
        # for i in range(batch_len):
        _x = x
        _x = self.layers(_x)
        _x = self.activation(_x)
        _x = _x.mean(dim=-1)
        _x = self.final(_x)
        
        # y[i] = _x
    
        return _x
    
class FCNBaseline2D(nn.Module):
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
            ConvBlock2D(in_channels, 256, 5, 1),
            ConvBlock2D(256, 256, 4, 1),
            ConvBlock2D(256, 256, 4, 1),
            ConvBlock2D(256, 128, 3, 1)
        ])
        
        self.final = None#nn.Linear(800, num_pred_classes)
        
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.layers(x)
        x = self.activation(x)
        print(x.shape)
        
        x = x.flatten(end_dim=1)
        print(x.shape)
        x = x.mean(dim=-1)
        print(x.shape)
        x = x.mean(dim=-1)
        print(x.shape)
        if self.final is None:
            self.final = nn.Linear(len(x), self.input_args['num_pred_classes'])
        
        x = self.final(x)
        
        return x

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
            NeuralArithmeticLogicUnitCell(128, 32),
            NeuralArithmeticLogicUnitCell(32, 16),
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
            
            outputs[i, :] = y
        
        return outputs