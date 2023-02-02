import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from .nac import NeuralAccumulatorCell
from torch.nn.parameter import Parameter
from operator import mul

class NeuralArithmeticLogicUnitCell(nn.Module):
    """A Neural Arithmetic Logic Unit (NALU) cell [1].

    Attributes:
        in_dim: size of the input sample.
        out_dim: size of the output sample.

    Sources:
        [1]: https://arxiv.org/abs/1808.00508
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.eps = 1e-10

        self.G = Parameter(torch.Tensor(out_dim, in_dim))
        self.nac = NeuralAccumulatorCell(in_dim, out_dim)
        self.register_parameter('bias', None)

        init.kaiming_uniform_(self.G, a=math.sqrt(5))

    def forward(self, input):
        a = self.nac(input)
        g = torch.sigmoid(F.linear(input, self.G, self.bias))
        add_sub = g * a
        log_input = torch.log(torch.abs(input) + self.eps)
        m = torch.exp(self.nac(log_input))
        mul_div = (1 - g) * m
        y = add_sub + mul_div
        return y

    def extra_repr(self):
        return 'in_dim={}, out_dim={}'.format(
            self.in_dim, self.out_dim
        )

from typing import *

Size = Union[int, Tuple[int], Iterable[int]]

class NeuralArithmeticLogicUnit(nn.Module):
    """A stack of NAC layers.

    Attributes:
        num_layers: the number of NAC layers.
        in_dim: the size of the input sample.
        hidden_dim: the size of the hidden layers.
        out_dim: the size of the output.
    """
    def __init__(self, input_size:Size, hidden_size:Size, output_size:Size, num_layers:int=1,  **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self._reshape = (
            not isinstance(input_size, int),
            not isinstance(hidden_size, int),
            tuple(output_size) if not isinstance(output_size, int) else None
        )
        
        hidden_dim = hidden_size if isinstance(hidden_size, int) else np.prod(list(hidden_size))
        in_dim = input_size if isinstance(input_size, int) else np.prod(list(input_size))
        out_dim = output_size if isinstance(output_size, int) else np.prod(list(output_size))

        layers = []
        for i in range(num_layers):
            layers.append(
                NeuralArithmeticLogicUnitCell(
                    in_dim=(hidden_dim if i > 0 else in_dim),
                    out_dim=(hidden_dim if i < num_layers - 1 else out_dim),
                )
            )
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        (reshape_X, _, reshape_y) = self._reshape
        if reshape_X:
            x = x.flatten()
        y = self.model(x)
        if reshape_y is not None:
            y = y.reshape(reshape_y)
        return y
