import pandas as pd
#import modin.pandas as pd
import numpy as np
from numpy import ndarray
from pandas import DateOffset, Timestamp, Series, DataFrame, DatetimeIndex
# from datetime import datetime, date
# dt.
from frozendict import frozendict
from tools import *
from engine.utils import *
from cytoolz import *
from fn import F, _
from typing import *

from engine.trading import TradingEngineBase
from engine.data.argman import ArgumentManager
from pprint import pprint
import datetime as dt

class ForecasterManager:
   pass

class ForecastManager:
   owner:TradingEngineBase
   argm:Optional[ArgumentManager] = None
   
   def __init__(self, owner=None):
      self.owner = owner
      self.argm = None
      self._fc_prims = None
      
   def attach(self, owner):
      self.owner = owner
      
   def init(self):
      self.argm = self.owner.argm
      
   @cached_property
   def forecasters(self):
      return frozendict(self.owner.forecasters)
   
   @property
   def fc_prims(self):
      if self._fc_prims is not None:
         return self._fc_prims
      
      fcm = self.forecasters
      out = {sym:{} for sym in self.argm.symbols}
      for fcid, nn in fcm.items():
         all_outputs = self.argm.apply_all_df(nn, n_steps=nn.params.n_steps)
         for sym, outputs in all_outputs.items():
            outputs = reducenum(outputs)
            out[sym][fcid] = convert_df(outputs)
      self._fc_prims = out
      return out
   
   def get_forecast_primitives(self, date:pd.Timestamp)->Dict[str, Dict[str, ndarray]]:
      r = {}
      for sym, outs in self.fc_prims.items():
         r[sym] = valmap(partial(thePeeAndPoop, date), outs)
      return r
   
   def agg_forecast_primitives(self, stoopid:Dict[str, List[ndarray]], method='avg')->Dict[str, ndarray]:
      # o = valmap(lambda l: np.asanyarray(l), valfilter(notnone, stoopid))
      o = stoopid.copy()
      for sym, l in stoopid.items():
         if l is None or len(l) == 0:
            del o[sym]
            continue
         l = [x for x in l if x is not None]
         nd = np.asanyarray(l)
         o[sym] = ndreduce(nd, method=method).T
      return o
   
def convert_df(df: pd.DataFrame)->Dict[pd.Timestamp, ndarray]:
   out = {}
   for ts, y in df.iterrows():
      out[ts] = y
   return frozendict(out)

def thePeeAndPoop(date, d):
   if isinstance(d, (dict, frozendict)):
      return d.get(date, None)
   
   if date in d.index:
      return d.loc[date].values
   else:
      return None

from numba import jit, prange, vectorize
         
@jit(cache=True)
def ndreduce(piss_and_shit:ndarray, method='avg'):
   a = piss_and_shit.T
   r = np.zeros((a.shape[0], 1))
   for i in prange(a.shape[0]):
      c = a[i]
      if method == 'avg' or method == 'mean':
         r[i] = np.mean(c)
      elif method == 'min':
         r[i] = np.min(c)
      elif method == 'max':
         r[i] = np.max(c)
      else:
         r[i] = np.nan
   return r.T

@jit(cache=True)
def reducenum(n, factor=0.02):
   r = (n - (n * factor))
   return r