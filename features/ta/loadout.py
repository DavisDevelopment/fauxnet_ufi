import numpy as np
import pandas as pd
import pandas_ta as ta

from pandas import DataFrame, Series
from cytoolz.dicttoolz import merge
from typing import *
from faux.pgrid import PGrid
from inspect import isfunction, signature
import random
from random import sample
from itertools import product
from tools import isiterable, dotget
from cachetools import cached
from functools import lru_cache

import features.ta.volatility as vta

typename = lambda x: type(x).__qualname__

def dfget(df:DataFrame, proto:Union[str, Callable[[DataFrame], Series], Callable[[DataFrame], DataFrame]])->Union[Series, DataFrame]:
   if isinstance(proto, str):
      return df[proto]
   else:
      return proto(df)

def mkNode(ref:str, fn):
   if '.' in ref:
      fn_categ, fn_name = ref.split('.', maxsplit=2)
   else:
      fn_categ, fn_name = '', ref
   
   if isfunction(fn):
      reserved_names = 'open,high,low,close,volume'.split(',')
      special_cases = {'open_':'open'}
      
      sig = signature(fn)
      inputs = []
      config = []
      
      for arg_name, arg in sig.parameters.items():
         if arg_name in reserved_names:
            inputs.append(arg)
         elif arg_name in special_cases:
            # setattr(arg, '_name', special_cases[arg_name])
            # print(arg.name)
            inputs.append(arg)
         else:
            config.append(arg)
         # print(ref, arg_name, arg.kind)
         
      fn_ref = fn.__qualname__
      
      node = (fn_ref, fn, inputs, config)
      return node
   
   raise TypeError(fn)

def ind_merge(df:DataFrame, result:Any):
   if isinstance(result, DataFrame):
      rd:DataFrame = result
   elif isinstance(result, Series):
      rd:DataFrame = DataFrame({result.name:result})
   else:
      raise TypeError(f'Expected either a DataFrame or a Series, got {np.typename(result)}')
   if len(rd.dropna()) == 0:
      raise ValueError('Invalid computation: empty result')
   for c in rd.columns:
      df[c] = rd[c]
   return df

IRO = [
   ta,
   vta
]

@lru_cache
def resolve_ifn(fnref):
   if isinstance(fnref, str):
      for i in range(len(IRO)-1, 0, -1):
         ifn = dotget(IRO[i], fnref, None)
         if ifn is not None and isfunction(ifn):
            return ifn
   elif isfunction(fnref):
      return fnref
   
   raise NameError(f'Unknown function reference {fnref} of type {typename(fnref)}')

@lru_cache
def bind_indicator(name:str):
   ifn = getattr(ta, name, None)
   if ifn is None:
      raise ValueError(f'No indicator named {name} could be found')
   assert callable(ifn), f'{name} is not a function'
   fn_ref, fn, fn_inputs, fn_opts = mkNode(name, ifn)
   
   special_cases = {'open_':'open'}
   
   def wrapped_indicator(df:pd.DataFrame, options:Dict[str, Any]={}, inplace=False, append=False):
      df = df if inplace else df.copy()
      callkw = {}
      for p in fn_inputs:
         if p.name in options:
            v = dfget(df, options[p.name])
         elif p.name in special_cases:
            v = dfget(df, special_cases[p.name])
         else:
            v = dfget(df, p.name)
         
         callkw[p.name] = v
         
      for p in fn_opts:
         if p.name in options:
            v = options[p.name]
         elif p.name in df.columns:
            v = dfget(df, p.name)
         else:
            continue
         callkw[p.name] = v
      result = fn(**callkw)
      return ind_merge(df, result)
   
   return wrapped_indicator
   
class Indicators:
   def __init__(self, *items):
      self.l = []
      for x in items:
         k, o = x
         self.add(k, o)
      
   def add(self, ind_name:str, doptions=None, **kwoptions):
      if doptions is None:
         options = kwoptions.copy()
      else:
         options = merge(doptions, kwoptions)
      wfn = bind_indicator(ind_name)
      self.l.append((ind_name, options, wfn))
      return self
   
   def apply(self, df:pd.DataFrame, inplace=False, append=False):
      df = df if inplace else df.copy()
      for (name, options, fn) in self.l:
         df = fn(df, options, inplace=inplace, append=append)
      return df
   
   def items(self):
      return [(n, o) for n,o,f in self.l]
   
   def __getstate__(self):
      return dict(
         l=self.items()
      )
      
   def __setstate__(self, state):
      self.l = []
      for k, o in state.get('l', []):
         self.add(k, o)
   
  
class IndicatorBag:
   def __init__(self, *items):
      self.l = []
      # for x in items
      
   def add(self, name:str, **items):
      options = PGrid(params=items)
      self.l.append((name, options))
      return self
   
   def sampling(self, N:int=3, batched=False):
      item_indices = list(range(len(self.l)))
      idx_samples = product(*([item_indices] * N))
      #TODO
      
      for betty in idx_samples:
         if len(set(betty)) != N:
            continue
         sampled_items = tuple(self.l[i] for i in betty)
         
         if batched:
            yield self.combinations(items=sampled_items)
         else:
            yield from self.expand(items=sampled_items)
   
   def expand(self, items=None):
      L = (items if (items is not None and isiterable(items)) else self.l)
      assert all((x in self.l for x in L))
      
      self.expanded_params = {}
      kl = []
      pgl = []
      
      for k, pg in L:
         kl.append(k)
         pgl.append(pg.expand())
      
      # allp = []
      for perm in product(*pgl, repeat=1):
         options = dict(zip(kl, perm))
         # print(options)
         # allp.append(options)
         yield options
      # return allp
   
   def combinations(self, items=None):
      return list(self.expand(items=items))