import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch import Tensor, tensor, typename
from torch.nn.parameter import Parameter

from typing import *

from torch.nn import *
from torch.nn.functional import *

class AutoClassify(Module):
   def __init__(self, nclasses:int, class_labels:Optional[List[str]]=None):
      super().__init__()
      
      self.nclasses = nclasses
      self.class_labels = class_labels
      
   def forward(self, input:Tensor)->int:
      raise NotImplementedError()
   
class NacAutoClassify(AutoClassify):
   def __init__(self, input_shape, hidden_shape=(64, 64), nclasses:int=10, class_labels=None):
      super().__init__(nclasses, class_labels=class_labels)
      
      from ptmdl.models.nac import NacCell, NeuralAccumulatorCell
      
      output_shape = (1, nclasses)
      self.lbl_net = Sequential(
         NacCell(input_shape, hidden_shape),
         NacCell(hidden_shape, hidden_shape),
         Sigmoid(),
         NacCell(hidden_shape, output_shape),
         Sigmoid()
      )
      
   def forward(self, input:Tensor):
      out = self.lbl_net(input).squeeze()
      lbl = torch.argmax(out)
      return lbl