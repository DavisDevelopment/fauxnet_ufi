from engine.trading import *
from engine.utils import *
from engine.utils import _reducer
from engine.utypes import MultiAdvisorPolicy, FrozenSeries
from fn import _, F
from time import time
from numba import jit, vectorize, generated_jit, prange
from numba.experimental import jitclass
import numba as nb
from typing import *
from tools import closure, nor, unzip
from ds.gtools import ojit
from engine.trading import TradingEngineBase
from frozendict import frozendict

from cytoolz import *
import cytoolz as tlz
from tools import getcols, getcolsmatching


from dataclasses import dataclass, asdict, astuple
import numpy as np
from numpy import ndarray
from pandas import Series, Timestamp
from engine.utils import notnone

def calc_moe(ytrue, ypred):
   offset = np.abs(ypred - ytrue)
   # print('no. of outliers:', len(b))
   if not np.isscalar(ytrue) and not np.isscalar(ypred):
      offset[offset == 0] = 1e-8
      ytrue[ytrue == 0] = ytrue.mean()
   
   offset_pct = offset/ytrue
   if not np.isscalar(offset_pct):
      margin_of_error = offset_pct.mean()
   
   else:
      margin_of_error = offset_pct
   return margin_of_error

@dataclass(order=True, repr=True)
class ForecastBase:
   generated_by:Optional[str] = None
   sym:Optional[str] = None
   
   roi:Optional[float] = None
   
   current_data:Optional[Series] = None
   predicted_data:Optional[Series] = None
   true_data:Optional[Series] = None
   
   
   def asdict(self):
      return dict(sym=self.sym, roi=self.roi, current_data=self.current_data, predicted_data=self.predicted_data, true_data=self.true_data)

class Forecast(ForecastBase):
   _ytyp = None
   _moe:Optional[float] = None
   
   def __data_repr__(self, s:Optional[Series])->Any:
      if s is None: 
         return None
      return (s.open, s.high, s.low, s.close)
   
   def __repr__(self):
      r = type(self).__name__ + '('
      enc = lambda k,v: f'{k}={repr(v)}'
      def put(k, v=pd):
         if v is None:
            return 
         elif v is pd:
            return put(k, getattr(self, k))
         nonlocal r
         r += f'{enc(k,v)},'
      
      put('sym')
      put('roi')
      put('current_data', self.__data_repr__(self.current_data))
      put('predicted_data', self.__data_repr__(self.predicted_data))
      put('true_data', self.__data_repr__(self.true_data))
            
      r += ')'
      return r
   
   @property
   def forecast(self): return self.predicted_data
   
   def isnotempty(self):
      return notnone(self.predicted_data)
   
   @property
   def ytyp(self):
      if notnone(self._ytyp): return self._ytyp
      
      t = None
      p = None
      if self.predicted_data is not None:
         p = self.predicted_data.to_numpy()
      if self.true_data is not None:
         t = self.true_data.to_numpy()
      r = (t, p)
      if None not in r:
         self._ytyp = r
      return r
   
   @property
   def moe(self):
      if notnone(self._moe): 
         return self._moe
      
      ytyp = self.ytyp
      if None not in ytyp:
         t,p = ytyp
         self._moe = calc_moe(t, p)
         return self._moe
      return None
   
   def fulfill(self, real:Series):
      if self.true_data is None:
         self.true_data = real
         self.moe + 1.0
         
   def asdict(self):
      return tlz.merge(super().asdict(), {})
     
def as_fc_aggregator(ufunc):
   def wrapped(inputs):
      n = len(inputs)
      if n == 0:
         return None
      elif n == 1:
         return inputs[0]
      else:
         return ufunc(inputs)
   return wrapped

def _most_accurate(forecasts:List[Forecast])->Forecast:
   assert None not in [fc.true_data for fc in forecasts]
   cols = 'open,high,low,close'.split(',')
   def s_ae(p, t):
      p:np.ndarray = p[cols].to_numpy()
      t:np.ndarray = t[cols].to_numpy()
      error:ndarray = (p - t)
      
      error = abs(error)
      return error.sum()
   
   def ae(node:Forecast):
      if node.predicted_data is None or node.true_data is None:
         return np.inf
      else:
         return s_ae(node.predicted_data, node.true_data)
   
   return minby(ae, forecasts)

class MultiForecast:
   sym:str
   _nodes:List[Forecast]
   # _default_aggr = as_fc_aggregator(_fcmean)
   _selector:Callable[[List[Forecast]], Forecast]
   
   def __init__(self, sym, nodes):
      self.sym = sym
      self._nodes = nodes
      self._selector = _most_accurate
      
   def _aggregate(self, func, acc=None):
      if acc is None:
         acc = self._default_aggr
      return acc(list(map(self._nodes, func)))
   
   def _tail(self, name, inputs):
      from fnmatch import translate
      import re
      pat = lambda s: re.compile(translate(s))
      series = pat('*_data')
      
   def __repr__(self):
      roi = f'{self.roi*100:.2f}%'
      return f'MultiForecast(sym={self.sym}, roi={roi})'
   
   @cached_property
   def roi(self):
      return np.array(list(map(_.roi, self._nodes))).mean()
   
   def __getattr__(self, name):
      f = attrgetter(name)
      r = list(map(f, self._nodes))
      if len(r) == 1:
         return r[0]
      else:
         r = self._tail(name, r)
         return r
         
def merge_forecasts(a:Forecast, b:Forecast):
   assert isinstance(a, Forecast) and isinstance(b, Forecast)
   da, db = a.asdict(), b.asdict()
   dab = dzip(da, db)
   dab = dissoc(valmap(lambda p: nor(*p) if len(p) > 1 else p[0], dab), 'sym', 'roi')
   
   for k, v in dab.items():
      if not k.endswith('_data'):
         continue
      if v is None:
         continue
      elif isinstance(v, tuple):
         n = len(v)
         if n == 0: continue
         elif n == 1:
            dab[k] = v[0]
         elif n >= 2:
            dab[k] = reduce(lambda l, r: (l + r)/2.0, v)

   return Forecast(
       sym=nor(a.sym, b.sym),
       roi=max(a.roi, b.roi),
       **dab
   )
   
def merge_advice_entries(a, b):
   return merge_forecasts(a, b)

toadvmaps = F() >> (lambda dm: {d.sym:d for d in dm})
toadvmaps = F() >> (map, toadvmaps)
toadvmaps >>= list

gets = lambda d, *keys: tuple(d.get(k, None) for k in keys)
findlap = lambda m: reduce(lambda x, y: x & y, map(lambda x: set(x.keys()), m))

def _fcagg(forecasts:List[Forecast]):
   syms = set([fc.sym for fc in forecasts])
   assert len(syms) == 1
   sym = list(syms)[0]
   
   return MultiForecast(sym, forecasts)

def compute_vinn_center(advice_lists:List[List[Forecast]], agg:Callable[[List[Forecast]], Forecast]=None):
   if agg is None:
      agg = _fcagg
   
   advice_lists = [l for l in advice_lists if len(l) > 0]
   def advl(l:List[List[Forecast]], f):
      return [[f(fc) for fc in row] for row in l]
   
   sig = [sorted([fc.sym for fc in row]) for row in advice_lists]
   
   sig = [set(row) for row in sig]
   
   flatadv = flat(advice_lists)
   def sel(predicate):
      return list(filter(predicate, flatadv))
   def fromall(func):
      return list(map(func, flatadv))
   
   generator_names = list(set(fromall(_.generated_by)))
   res = []
   if len(sig) >= 2:
      area = reduce(lambda l, r: l | r, sig)
      for sym in area:
         q = sel(_.sym == sym)
         fc = agg(q)
         pprint(fc)
         res.append(fc)
   return res