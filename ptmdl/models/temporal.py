import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch import Tensor, tensor, typename, Size
from torch.nn.parameter import Parameter

from typing import *

from torch.nn import *
from torch.nn.functional import *

from econn.market import Market

class NextNet(Module):
   _vb : TensorBuffer
   _eb : Optional[TensorBuffer] = None
   _v_next : Optional[Tensor]
   
   def __init__(self, buffer_size:int):
      super().__init__()
      self._buffer_size = buffer_size
      
      self._vb = TensorBuffer(buffer_size)
      self._tb = TensorBuffer(buffer_size)
      self._eb = TensorBuffer(buffer_size)
      
   def buffer(self)->Tensor:
      return self._vb.get()
   
   def push(self, input, elapsed=1.0):
      self._vb.push(input)
      self._tb.push(elapsed)
      
   def next(self, input):
      """
      -
       compute forecast for next value

       Parameters
       ----------
       input : Tensor
      """
      raise NotImplementedError()
   
   def _fc_criterion(self, x, y):
      return mse_loss(x, y)
      
   def forward(self, input):
      lfc = self._v_next
      self.push(input)
      if lfc is not None:
         self._eb.push(self._fc_criterion(input, lfc))
      
      fc = self.next(input)
      self._v_next = fc
      
      return fc

class MemNextNet(NextNet):
   def __init__(self, mem_size:int, input_shape):
      super().__init__(mem_size)
      
      self.mem_size = mem_size
      # self._buffer_size = mem_size
      # self._vb = torch.empty((mem_size, *input_shape), dtype=torch.float32)
      # self._vbi = 0
      self._v_next = None
      
      # self._mem_cell = LSTMCell(torch.prod(tensor(input_shape)).item(), mem_size) #? f(input) -> (hidden_state, cell_state)
      self._mem_cell = LSTM(1, hidden_size=4, num_layers=3, batch_first=True)
      #? nxt_cell(input, hidden_state, cell_state) -> next `input`
      #self.nxt_cell = None
      
      self.input_shape = input_shape
      # self._build(input_shape)
      
   def nxt_cell(self, input, hidden_state, cell_state):
      raise NotImplementedError()
      
   def next(self, input):
      h, c = self.mem_cell(input)
      next_input = self.nxt_cell(input, h, c)
      return next_input.squeeze()
   

class NacMemNextNet(MemNextNet):
   def __init__(self, mem_size: int, input_shape):
      super().__init__(mem_size, input_shape)
      from ptmdl.models.nac import NacCell
      
      self._nxt_cell = Sequential(
         NacCell((1, mem_size), (32, 32)),
         NacCell((32, 32), (32, 32)),
         NacCell((32, 32), input_shape),
         PReLU()
      )
      
   # def 
      
   def nxt_cell(self, input, hidden_state, cell_state):
      state:Tensor = cell_state
      nxt = self._nxt_cell(state).squeeze()
      return nxt


class DeltaNet(Module):
   def __init__(self, memsize:int):
      pass

class TemporalDecompositionUnit(Module):
   pass