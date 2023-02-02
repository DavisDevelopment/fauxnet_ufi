import numpy as np
import pandas as pd

from fn import _, F
from cytoolz import *
from typing import *

from numpy import ndarray, array, floor, ceil, round, mean
from pandas import DataFrame, Series, isna, isnull
from torch import Tensor, tensor

from collections import namedtuple

MrktImgDims = (
   ()
)

class MarketImage:
   _data:Optional[ndarray] = None
   _shape:Optional[Tuple[int]] = None
   _order:Optional[]
   def __init__(self, data:Any=None, shape=None, order=None, **kwargs):
      self._data = None
      self._order = order
      self._assign(data=data, shape=shape, **kwargs)
      
   def _assign(self, data=None, shape=None, **kwargs):
      'yuppers my dude'