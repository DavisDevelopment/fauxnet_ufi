import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch import Tensor, tensor, typename
from torch.nn.parameter import Parameter
from torch.jit import script
from typing import *

class NacCell(nn.Module):
   """A Neural Accumulator (NAC) cell [1].

   Attributes:
      in_dim: size of the input sample.
      out_dim: size of the output sample.

   Sources:
      [1]: https://arxiv.org/abs/1808.00508
   """
   
   __constants__ = ['in_dim', 'out_dim', 'W_hat', 'M_hat']

   def __init__(self, input_shape:Tuple[int], output_shape:Tuple[int]):
      super().__init__()
      
      self._in_shape = input_shape
      self._out_shape = output_shape
      self.in_dim = torch.prod(tensor(input_shape)).item()
      self.out_dim = torch.prod(tensor(output_shape)).item()
      
      self.W_hat = Parameter(Tensor(self.out_dim, self.in_dim))
      self.M_hat = Parameter(Tensor(self.out_dim, self.in_dim))

      self.register_parameter('W_hat', self.W_hat)
      self.register_parameter('M_hat', self.M_hat)
      self.register_parameter('bias', None)

      self._reset_params()

   def _reset_params(self):
      init.kaiming_uniform_(self.W_hat)
      init.kaiming_uniform_(self.M_hat)

   def forward(self, input:Tensor)->Tensor:
      shape = input.shape
      input = input.flatten()
      print(input.dtype)

      W: Tensor = torch.tanh(self.W_hat) * torch.sigmoid(self.M_hat)
      y: Tensor = F.linear(input, W, self.bias)
      y = y.reshape(self._out_shape)

      return y

   def extra_repr(self)->str:
      return f'in_shape={self._in_shape}, out_shape={self._out_shape}'
   
# @script
class NeuralAccumulatorCell(nn.Module):
   """A Neural Accumulator (NAC) cell [1].

   Attributes:
      in_dim: size of the input sample.
      out_dim: size of the output sample.

   Sources:
      [1]: https://arxiv.org/abs/1808.00508
   """
   
   W_hat:Tensor
   M_hat:Tensor
   bias:Optional[Tensor]

   def __init__(self, in_dim, out_dim):
      super().__init__()
      
      self.in_dim = in_dim
      self.out_dim = out_dim

      self.W_hat = Parameter(torch.Tensor(out_dim, in_dim))
      self.M_hat = Parameter(torch.Tensor(out_dim, in_dim))

      self.register_parameter('W_hat', self.W_hat)
      self.register_parameter('M_hat', self.M_hat)
      self.register_parameter('bias', None)

      self._reset_params()

   def _reset_params(self):
      init.kaiming_uniform_(self.W_hat)
      init.kaiming_uniform_(self.M_hat)

   def forward(self, input:Tensor):
      input = input.float()
      
      assert input.shape[0] == self.in_dim, f'{input.shape[0]} != {self.in_dim}'
      
      W = torch.tanh(self.W_hat) * torch.sigmoid(self.M_hat)
      y = F.linear(input, W.float(), self.bias)
      return y

   def extra_repr(self):
      return 'in_dim={}, out_dim={}'.format(
         self.in_dim, self.out_dim
      )

class NAC(nn.Module):
   """A stack of NAC layers.

   Attributes:
      num_layers: the number of NAC layers.
      in_dim: the size of the input sample.
      hidden_dim: the size of the hidden layers.
      out_dim: the size of the output.
   """

   def __init__(self, num_layers, in_dim, hidden_dim, out_dim):
      super().__init__()
      
      self.num_layers = num_layers
      self.in_dim = in_dim
      self.hidden_dim = hidden_dim
      self.out_dim = out_dim

      layers = []
      for i in range(num_layers):
         layers.append(
            NeuralAccumulatorCell(
               hidden_dim if i > 0 else in_dim,
               hidden_dim if i < num_layers - 1 else out_dim,
            )
         )
      
      self.model = nn.Sequential(*layers)

   def forward(self, x):
      out = self.model(x)
      return out
