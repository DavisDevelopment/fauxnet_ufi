import torch
import torch.nn as nn
from torch.nn import *
from torch import Tensor, tensor

from torch.nn.utils import weight_norm
import torch.nn.functional as F
from typing import *

class EnsembleBaseline(nn.Module):
   def __init__(self, components:List[Module]):
      super().__init__()
      
      self.components = components

   def reduce(self, results:List[Tensor]):
      raise NotImplementedError()
   
   def forward(self, inputs:Tensor):
      return self.reduce([m(inputs) for m in self.components])
   
class TsClassifyingEnsemble(nn.Module):
   def __init__(self, components:List[Module]):
      super().__init__()
      
      # self.num_pred_classes = num_pred_classes
      self.components = components
      
   def forward(self, inputs:Tensor):
      #TODO: when using output-method #2, terminate computation eagerly to improve performance
      outputs = [m(inputs).argmax() for m in self.components]
      h = {}
      for o in outputs:
         o = int(o)
         h[o] = (h.setdefault(o, 0) + 1)
      # print(h)
      th = tensor([h[i] for i in range(len(h))])
      if th[1] > 1:
         return tensor([0, 1])
      else:
         return tensor([1, 0])
      #? two ways of doing this: 
      #? 1. regular argmax() operation
      #? 2. output 1 if `1` is present in `outputs` more than once, else output 0
      
      th[1]
      print(th)
      return th.to(torch.float32)