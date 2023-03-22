import numpy as np
import pandas as pd
import pandas_ta as ta

from pandas import DataFrame, Series
from cytoolz.dicttoolz import merge
from typing import *
from faux.pgrid import *
from inspect import isfunction, signature

def dfget(df:DataFrame, proto:Union[str, Callable[[DataFrame], Series], Callable[[DataFrame], DataFrame]])->Union[Series, DataFrame]:
   if isinstance(proto, str):
      print(df)
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
      sig = signature(fn)
      inputs = []
      config = []
      
      for arg_name, arg in sig.parameters.items():
         if arg_name in reserved_names:
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

def bind_indicator(name:str):
   ifn = getattr(ta, name, None)
   if ifn is None:
      raise ValueError(f'No indicator named {name} could be found')
   assert callable(ifn), f'{name} is not a function'
   fn_ref, fn, fn_inputs, fn_opts = mkNode(name, ifn)
   
   def wrapped_indicator(df:pd.DataFrame, options:Dict[str, Any]={}, inplace=False, append=False):
      df = df if inplace else df.copy()
      callkw = {}
      print(options)
      print(fn_inputs)
      for p in fn_inputs:
         if p.name in options:
            v = dfget(df, options[p.name])
         else:
            v = dfget(df, p.name)
         
         callkw[p.name] = v
         
      print(fn_opts)
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
   
  
class IndicatorBag:
   def __init__(self, *items):
      self.l = []
      # for x in items
      
   def add(self, name:str, **items):
      options = PGrid(params=items)
      self.l.append((name, options))
      return self
   
   def expand(self):
      self.expanded_params = {}
      kl = []
      pgl = []
      
      for k, pg in self.l:
         kl.append(k)
         pgl.append(pg.expand())
      
      allp = []
      for perm in product(*pgl, repeat=1):
         options = dict(zip(kl, perm))
         # print(options)
         allp.append(options)
      return allp