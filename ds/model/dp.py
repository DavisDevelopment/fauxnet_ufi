
import numpy as np
import functools
import toolz
import pandas as pd
#import modin.pandas as pd

from numpy import *
import numba

from functools import *
from cytoolz import *

from sklearn.preprocessing import MinMaxScaler
from builtins import eval
import builtins

class SymIndexer:
   def __init__(self):
      pass

   def __getitem__(self, idx):
      return idx

_TrueNoValue = object()
class LambdaIndexer:
   def __getitem__(self, idx):
      def f(self_, value=_TrueNoValue):
         if value is _TrueNoValue:
            return self_[idx]
         else:
            self_[idx] = value
      return f

idx = SymIndexer()
fidx = LambdaIndexer()
print(idx[:, 0, :, :])

def passthru_(*args):
   return args

"""
d = ShapeDescr('a, b, c, d') * 14
d.array(0)
"""

class ShapeDescription:
   """
   i.e. ShapeDescription('open', 'high', 'low', 'close')
   """
   def __init__(self, *shape):
      if isinstance(shape[0], str):
         self.names = shape[:]
         self.shape = (len(self.names),)
      else:
         self.names = None
         self.shape = tuple()
      
      self._scalers = {}
      self.current_context_name = ''
      
   def by(self, n:int):
      if len(self.shape) == 1 and n is not None:
         self.shape = (self.shape[0] * n,)
      else:
         self.shape = (n, *self.shape)
      return self
   
   def rby(self, n:int):
      self.shape = (*self.shape, n)
   
   def __getitem__(self, i):
      return self.shape[i]
   
   @cached_property
   def column_idx_builder(self):
      global idx
      expr = 'lambda j: idx[' + (':,' * (len(self.shape) - 1)) + 'j]'
      return builtins.eval(expr)
   
   def itercolidxs(self, n:int):
      ff = self.column_idx_builder
      
      for i in range(n):
         yield ff(i)
         
from typing import *
         
class Signature:
   __slots__ = ['input_shape', 'output_shape']
   
   input_shape:  Union[ShapeDescription, Tuple[int]]
   output_shape: Union[ShapeDescription, Tuple[int]]
   
   def __init__(self, i, o):
      # super().__init__()
      self.input_shape = i
      self.output_shape = o

class DataProcessor:
   def __init__(self):
      self.scaling_strategy = ScalingStrategy.Default
   
   def scale(self, x:ndarray):
      pass
   
   
from enum import Enum
class ScalingStrategy(Enum):
   Default = 0
   PerColumn = 1
