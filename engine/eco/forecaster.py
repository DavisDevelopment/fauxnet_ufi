
from ds.forecasters import NNForecaster

from numpy import ndarray
from pandas import DataFrame, Series
import numpy as np
import pandas as pd
#import modin.pandas as pd

from engine.mixin import EngineImplBase as Engine
from engine.data.argman import ArgumentManager
from typing import *
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler

def resolve_index(df:DataFrame, i=None):
   if i is None:
      return None
   if isinstance(i, tuple):
      return tuple([resolve_index(df, j) for j in i])
   elif isinstance(i, slice):
      i:slice = i
      if i.start is None and i.step is None:
         return slice(resolve_index(df, i.stop))
      else:
         return slice(resolve_index(df, i.start), resolve_index(df, i.stop), i.step)
   else:
      last_i = (len(df) - 1)
      if i <= 0:
         return last_i - i
      else:
         return i

def convert_index(o, i=None):
   if isinstance(i, int):
      if i < 0:
         raise IndexInFuture()
      else:
         dti = o.ds.time_range
         return dti.iloc[len(dti)]
   else:
      return i

class TradeEngineForecaster:
   nnf: NNForecaster = None
   engine: Engine = None
   argm: ArgumentManager = None
   ds:DataSource = None
   
   scalers:Optional[Dict[str, TransformerMixin]] = None
   _ready = False
   
   def __init__(self, owner:Engine, fc:NNForecaster):
      self.engine = owner
      self.nnf = fc
      self.argm = self.engine.argm
      self.ds = self.engine.book
      
   @property
   def date_range(self):
      dti = self.ds.time_range
      dti = dti[:self.engine.current_date]
      return dti
      
   def call(self, index=0):
      tr = self.ds.time_range
      tr = tr[:self.engine.current_date]
      date = tr.iloc[len(tr) - 1 - index]
      X = self.engine.get_args(fc, date=date)
      ypred = self.engine.forecast_inner(self.nnf, X)
      ypred = list(ypred.values())
      return ypred
   
   def __call__(self, index=0):
      return self.call(index=0)
   
class IndexInFuture(Exception): pass