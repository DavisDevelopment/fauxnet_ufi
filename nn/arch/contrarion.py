
import torch
from torch import Tensor, nn, jit, zeros, full
from torch.autograd import Variable
from torch.nn import *
# from torch.optim import *

from typing import *
from dataclasses import dataclass
from tools import flat

@dataclass
class Args:
   input_shape:Optional[Iterable[int]] = None
   output_shape:Optional[Iterable[int]] = None
   in_seq_len:Optional[int] = None
   out_seq_len:Optional[int] = 1
   num_input_channels:Optional[int] = 4
   input_channels:Optional[tuple] = ('open', 'high', 'low', 'close')
   num_predicted_classes:Optional[int] = 2
   
   symbol:Optional[str] = None
   df:Optional[Any] = None
   
   shuffle:bool = True
   val_split:float = 0.12
   epochs:int = 30

class ApexEnsemble(nn.Module):
   components:List[Module]
   component_names:List[str]
   
   def __init__(self, components:Dict[str, Module], config:Args, **kw):
      super().__init__()
      #TODO wrap each component with an "encode" and "decode" layer, before the component's input layer, and after the component's output layer respectively
      
      self.config = config
      self.components = []
      self.component_names = []
      for name, m in components.items():
         self.component_names.append(name)
         self.components.append(m)
      
      L = self.n_layers = len(self.components)
      
      #* NOTE: the cW parameter is meant to be the weights used for the weighted averaging of the votes for the components, but is not used currently
      self.batch_first = kw.pop('batch_first', True)
      
   def _fix(self):
      self.components = list(map(lambda x: x[1] if isinstance(x, tuple) else x, self.components))
      
   def parameters(self):
      return (
         list(super().parameters()) +
         flat([list(a.parameters()) for a in self.components])
      )
      
   def call_components(self, inputs:Tensor):
      up = 0
      down = 0
      
      for sigid, signal in enumerate(self.components):
         label = signal(inputs.unsqueeze(0))[0].argmax()
         # print(label.shape)
         if label == 1:
            up += 1
         elif label == 0:
            down += 1
         
         else:
            print(f'ERROR: invalid signal-label "{label}"')
      
      # print(f'up={up}, down={down}')   
      if up > down:
         # outs[i, 1] = 1.0
         return 1
      else:
         # outs[i, 0] = 1.0
         return 0
      
   def forward(self, inputs:Tensor)->Tensor:
      self._fix()
      
      if self.batch_first:
         
         batch_dim = inputs.size(0)
         outs = Variable(zeros((batch_dim, 2), requires_grad=True))
         
         for i in range(batch_dim):
            y = self.call_components(inputs[i])
            if y == 0:
               outs[i, 0] = 1.0
            else:
               outs[i, 1] = 1.0
         
         return outs
      
      else:
         print(inputs.shape)
         outs = Variable(zeros((2,), requires_grad=True))
         y = self.call_components(inputs)
         if y == 1:
            outs[1] = 1.0
         elif y == 0:
            outs[0] = 1.0
         return outs

class Contrarion(Module):
   def __init__(self, pro:Module, con:Module):
      super().__init__()
      
      self.pro = pro
      self.con = con
      
      
   def forward(self, inputs:Tensor):
      print('Urine', inputs.size())
      
      pro_y = self.pro(inputs)
      con_y = self.con(inputs)