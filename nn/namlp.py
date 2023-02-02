import torch
import numpy as np

from torch import tensor, Tensor, randn, empty, zeros
from torch import nn
from torch.nn import *
from torch.nn.functional import relu
from torch import optim
from torch.jit import script, ignore, optimize_for_inference, freeze, export

from torch.autograd import Variable
import torch.nn.functional as F
from typing import *

from nn.nac import NeuralAccumulatorCell

class NacMlp(Module):
   def __init__(self, input_size:int, output_size:int, hidden_size:Optional[List[int]]=None, L:Callable[..., Module]=NeuralAccumulatorCell, show_work=False):
      super().__init__()
      
      self.input_size = input_size
      self.hidden_size = hidden_size
      self.output_size = output_size
      self.show_work = show_work
      # hsi = iter(hidden_size)
      
      if hidden_size is None or len(hidden_size) == 0:
         self._m = L(input_size, output_size)
      
      else:
         prev_size = self.input_size
         layers = []
         for next_size in hidden_size:
            layers.append(L(prev_size, next_size))
            prev_size = next_size
         layers.append(L(prev_size, self.output_size))
         # self.logic_layers = layers
         self._m = Sequential(*layers)
      
   def forward(self, X:Tensor):
      return self._m(X)
   
   def extra_repr(self) -> str:
      # return super().extra_repr()
      return f'show_work={self.show_work}'
   
   def parameters(self, recurse: bool = True):
      return self._m.parameters()
   
# class NacMlp2d(Module):
#    def __init__(self, input_shape, output_shape, hidden=None):
#       super().__init__()
#       assert input_shape is not None and output_shape is not None
      
#       (in_rows, in_cols) = input_shape
#       (out_rows, out_cols) = output_shape
      
#       self._fx = NacMlp(in_cols, out_cols, hidden_size=[in_cols, out_cols])
#       self._fy = NacMlp(in_rows, out_rows, hidden_size=[in_rows, out_rows])
      
#       self.input_shape = input_shape
#       self.output_shape = output_shape
#       n_rows, n_cols = cast(tuple, input_shape)
#       self.n_rows = n_rows
#       self.n_cols = n_cols
      
      
#    def forward(self, X:Tensor):
#       y = Variable(zeros(self.output_shape))
#       for i in range(self.n_rows):
#          y[i] = self._fx()
      