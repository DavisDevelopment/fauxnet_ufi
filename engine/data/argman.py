import pandas as pd
#import modin.pandas as pd 
import numpy as np
from numpy import ndarray
from pandas import DateOffset, Timestamp, Series, DataFrame, DatetimeIndex
# from datetime import datetime, date
# dt.

from tools import *
from engine.utils import *
from cytoolz import *
from fn import F, _
from typing import *

from engine.trading import TradingEngineBase
from pprint import pprint
import datetime as dt

G_cache_ctors = OrderedDict()
def cacheCtor(method):
   f = method
   if id(method) not in G_cache_ctors:
      G_cache_ctors[id(method)] = f
      # G_cache_ctors[id(f)] = f
   # return once(method)
   return f

class ArgumentManager:
   
   pricemaps:Mapping[Timestamp, Mapping[str, float]] = None
   freq:DateOffset = None
   
   date_ranges:OrderedDict[str, pd.DatetimeIndex] = None
   owner:TradingEngineBase = None
   
   def __init__(self, owner=None):
      self.owner = owner
      if notnone(owner):
         self.attach(owner)
      
      self.columns = 'open,high,low,close'.split(',')
      self.freq = None
      self.loc_ranges = None
      self.iloc_ranges = None
      self.ndas = None
      self.scaled_ndas = None
      self.pricemaps = None
      
         
   def attach(self, owner:"TradingEngineBase"):
      self.owner = owner
      self.init()
      
   def init(self):
      from engine.trading import TradingEngineBase
      
      assert notnone(self.owner)
      engine:TradingEngineBase = self.owner
      
      self.freq = engine.freq
      self.loc_ranges = OrderedDict()
      self.date_ranges = OrderedDict()
      self.iloc_ranges = OrderedDict()
      self.pricemaps = OrderedDict()
      self.ndas = OrderedDict()
      self.scaled_ndas = OrderedDict()
      
   @memoize
   def allDates(self):
      st, et = self.owner.book.time_range
      return (ts for ts in pd.date_range(start=st, end=et, freq=self.freq, normalize=True))
      
   def rangeOf(self, sym:str)->DatetimeIndex:
      if sym not in self.date_ranges.keys():
         i:DatetimeIndex = self.owner.book[sym].index
         r = pd.date_range(start=i.min(), end=i.max(), freq=self.freq, normalize=True)
         self.date_ranges[sym] = r
         return r
      else:
         r:pd.DatetimeIndex = self.date_ranges[sym]
         return r
      
   def irangeOf(self, sym:str):
      dti:DatetimeIndex = self.rangeOf(sym)
      # dti.map(lambda x: dti.get_loc)
      return np.arange(len(dti))

   def idxOf(self, date:Timestamp, sym:str):
      dti = self.rangeOf(sym)
      # print(dti)
      try:
         i = dti.get_loc(self.freq.rollback(date))
         return i
      except KeyError: #? 'date' is not found in the DataFrame for the given symbol
         return -1
      
   def indexOf(self, date:Timestamp, sym:str, unsafe=False):
      dti:DatetimeIndex = self.rangeOf(sym)
      if unsafe:
         try:
            return dti.get_loc(date)
         except KeyError:
            return -1
      try:
         date = self.freq.rollback(date)
         return dti.get_loc(date)
      except KeyError:
         return -1
   
   def get_nda(self, sym:str)->ndarray:
      if sym in self.ndas.keys():
         return self.ndas[sym]
      
      doc:DataFrame = self.owner.book[sym]
      nda:ndarray = doc[self.columns].to_numpy()
      self.ndas[sym] = nda
      return nda
   
   def scale_nda(self, a:ndarray, sm:Dict[str, Any], inverse=False):
      assert a.ndim == 2 and a.shape[1] == len(self.columns)
      
      a = a.T
      r = np.empty_like(a)
      
      for i in range(len(self.columns)):
         s = sm[self.columns[i]]
         x = (s.transform if not inverse else s.inverse_transform)(a[i].reshape(-1, 1))[:, 0]
         r[i] = x
      return r.T
   
   def get_sm(self, sym:str):
      if self.owner.scalers is None:
         self.owner.build_scalers()
      return self.owner.scalers.get(sym, None)
   
   def get_scaled_nda(self, sym:str):
      if sym in self.scaled_ndas.keys():
         return self.scaled_ndas[sym]
      nda = self.get_nda(sym)
      sm = self.get_sm(sym)
      scaled = self.scale_nda(nda, sm)
      self.scaled_ndas[sym] = scaled
      return scaled
   
   @memoize
   def get_features(self, sym:str, idx:Union[int, Timestamp], n_steps:int, scaled=True):
      tol = lambda a: a.tolist() if a is not None else None
      if idx is None and sym is None:
         r = OrderedDict()
         for sym in self.symbols:
            idxs = self.irangeOf(sym)
            
            row = [tol(self.get_features(sym, idx, n_steps, scaled=scaled)) for idx in idxs]
            row = [x for x in row if x is not None and len(x) > 0]
            row = r[sym] = np.asarray(row)
         return r
      
      elif sym is None:
         symbols = self.symbols
         return {sym: self.get_features(sym, idx, n_steps, scaled=scaled) for sym in symbols}
      
      data:ndarray = (self.get_nda if not scaled else self.get_scaled_nda)(sym)
      date = None
      # print(dt)
      if isinstance(idx, (dt.date, dt.datetime, Timestamp)):
         date = idx
         idx = self.idxOf(idx, sym)
      
      if idx == -1 or idx < (n_steps + 1):
         return None
      
      i:int = idx
      X:ndarray = data[i-n_steps+1:i+1, :]
      
      #TODO support skipping of sanity checks
      assert len(X) == n_steps
      assert np.isclose(data[i, :], X[-1]).all()
      
      return X
   
   @memoize
   def get_target(self, sym:str, idx:Union[int, Timestamp], scaled=True):
      if idx is None and sym is None:
         r = OrderedDict()
         for sym in self.symbols:
            idxs = self.irangeOf(sym)
            row = [self.get_target(sym, idx, scaled=scaled) for idx in idxs]
            # print(row)
            row = [x for x in row if x is not None]
            row = r[sym] = np.asarray(row)
         return r
      
      elif sym is None:
         symbols = self.symbols
         return {sym: self.get_target(sym, idx, scaled=scaled) for sym in symbols}
      
      if isinstance(idx, Timestamp):
         idx = self.idxOf(idx, sym)
      
      data:ndarray = self.get_nda(sym)
      
      i:int = idx
      
      if i < (len(data) - 1):
         return data[i+1, :]
      return np.full((len(self.columns),), np.nan)
   
   def all_features(self, sym=None, n_steps=14, scaled=True):
      return self.get_features(sym, None, n_steps, scaled=scaled)
   
   def all_targets(self, sym=None, scaled=True):
      return self.get_target(sym, None, scaled=scaled)
   
   @memoize
   def xya(self, n_steps=14, scaled=True):
      X = self.all_features(sym=None, n_steps=n_steps, scaled=scaled)
      y = self.all_targets(sym=None, scaled=scaled)
      y = valmap(lambda a: a[n_steps:], y)
      return dzip(X, y)
   
   @memoize
   def xy(self, sym=None, n_steps=14, scaled=True):
      if sym is None:
         return self.xya(n_steps=n_steps, scaled=scaled)
      
      return self.xya(n_steps=n_steps, scaled=scaled)[sym]
   
   def apply_all(self, nn, n_steps=14, tailscale=True):
      X = self.all_features(sym=None, n_steps=n_steps)
      syms = list(self.symbols)
      sizes = valmap(len, X)
      # sizes = [sizes[s] for s in syms]
      catted = np.concatenate([X[s] for s in syms])
      out = nn(catted)
      r = {}
      i = 0
      for sym in syms:
         v = out[i:i+sizes[sym]]
         i += sizes[sym]
         if tailscale:
            v = self.scale_nda(v, self.get_sm(sym), inverse=True)
         r[sym] = v
      return r
   
   def apply_all_df(self, nn, n_steps=14, tailscale=True):
      X = self.all_features(sym=None, n_steps=n_steps)
      syms = list(self.symbols)
      sizes = valmap(len, X)
      # sizes = [sizes[s] for s in syms]
      catted = np.concatenate([X[s] for s in syms])
      out = nn(catted)
      r = {}
      i = 0
      for sym in syms:
         v = out[i:i+sizes[sym]]
         i += sizes[sym]
         if tailscale:
            v = self.scale_nda(v, self.get_sm(sym), inverse=True)
         r[sym] = v
      
      def todf(sym, nd):
         if len(nd) == 1:
            return None
         idx = self.rangeOf(sym)[n_steps+1:]
         r = pd.DataFrame(data=nd, columns=self.columns, index=idx)
         
         return r
      
      o = {}
      for sym, nd in r.items():
         e = todf(sym, nd)
         if e is not None:
            o[sym] = e
      
      return o
      
   @property
   def symbols(self):
      return self.owner.book.keys()
   
   @memoize
   def pdpm(self, date=None):
      assert date is not None
      today = self.owner.book._agg.loc[self.freq.rollback(date)]
      c = getcolsmatching('*_close', today)

      today:Series = today[c].rename(lambda s: s[:s.index('_')])
      pm = today.to_dict()
      return pm
   
   @memoize
   def pricemap(self, idx:Union[Timestamp, int]):
      if not isinstance(idx, int):
         return self.pdpm(date=idx)
      
      return dict([(sym, self.owner.book._agg[f'{sym}_close'].iloc[idx]) for sym in self.symbols])
   
   def build_scalers(self):
      me = self.owner
      if me.scalers is not None:
         return me.scalers

      # * intelligently build a list of the columns that will need scaling
      cols = set()
      for f in me.forecasters.values():
            hp = f.params
            cols |= set(hp.target_columns)
      cols = list(cols)

      def scalermap(name: str):
         sm = {k: me._Scaler(copy=True, clip=True) for k in cols}
         for k, scaler in sm.items():
            setattr(scaler, 'symbol', name)
            setattr(scaler, 'column_name', k)
         return sm

      bk = me.book
      scalers = {}
      me.sc_statemap = bidict()

      # ? for each document (historical DataFrame) we have
      for name in bk.keys():
            doc = bk[name]
            doc = doc[cols]
            token_scalers = scalermap(name)
            scalers[name] = token_scalers
            # ? and for each column that will need to have scaling handled
            for c in cols:
               # ? using the available historical data for that column of that document
               data = doc[c].to_numpy()
               sc = token_scalers[c]
               # ? fit the Scaler to said data
               sc.fit(data.reshape(-1, 1))

      me.scalers = scalers
   
   def precompute_cache(self):
      print('...cache')
      # calls = list(unique(G_cache_ctors.values(), key=lambda x: id(x)))
      calls = list(G_cache_ctors.values())
      print(calls)
      for f in calls:
         f(self)
      return self
   
   @cacheCtor
   def _mkpmcache(self):
      print('Cache-Ctor')
      for ts in (self.allDates()):
         prices = self.pricemap(ts)
   
   @cacheCtor
   def _mkXcache(self):
      print('Cache-Ctor')
      stepCounts = [nnf.params.n_steps for nnf in self.owner.forecasters.values()]
      for ts in (self.allDates()):
         for n_steps in stepCounts:
            X = self.get_features(None, ts, n_steps)
            
