import numpy as np
import pandas as pd
import pandas_ta as ta

from pandas import DataFrame, Series
from typing import *

from cytoolz import *
from fn import F, _

from inspect import signature, isfunction, Signature
from datatools import renormalize, rescale
from tools import closure, dotget, dotgets, before, after, gets, once

def dfget(df:DataFrame, proto:Union[str, Callable[[DataFrame], Series], Callable[[DataFrame], DataFrame]])->Union[Series, DataFrame]:
   if isinstance(proto, str):
      return df[proto]
   else:
      return proto(df)
   
def renorm(n, min1, max1, min2, max2):
   d1 = (max1 - min1)
   d2 = (max2 - min2)
   return (d2 * (n - min1) / d1) + min2

def norm(n, lo, hi):
   return (n - lo) / (hi - lo)
   
def channel_ratios(chdf:DataFrame, columns=None):
   assert len(chdf) == 4 or columns is not None and len(columns) == 4
   if columns is None:
      columns = list(chdf.columns)
   close, lo, mid, hi = tuple(chdf[c] for c in columns)
   
   c = norm(close, lo, hi)
   m = norm(mid, lo, hi)
   
   return (c, m)

class Node:
   def __init__(self, name:str, fn, inputs, config, **proto) -> None:
      self.name = name
      self._fn = fn
      
      self.inputs = inputs
      self.config = config
      
      self.proto = proto.copy()
      
   def __str__(self):
      r = self.name + '('
      from inspect import Signature
      print(Signature(self.config))
      r += ')'
      return r
      
   def __call__(self, df:DataFrame, **kwargs):
      callsig = {}
      
      for p in self.inputs:
         if p.name in kwargs:
            v = dfget(df, kwargs.pop(p.name))
         
         elif p.name in self.proto:
            v = dfget(df, self.proto)
         
         else:
            v = dfget(df, p.name)
         
         callsig[p.name] = v
      
      for p in self.config:
         if p.name in kwargs:
            v = kwargs.pop(p.name)
            
         elif p.name in self.proto:
            v = dfget(df, self.proto)
         
         else:
            v = p.default
         
         callsig[p.name] = v

      ret = self._fn(**callsig)
      
      return ret
   
class ApplyNode:
   def __init__(self, node:Node, append=True, inplace=False, **proto):
      self.node = node
      self.append = append
      self.inplace = inplace
      self.proto = proto
   
   def __str__(self): return str(self.node)
      
   def __call__(self, df:DataFrame, **kwargs):
      kwargs = merge(self.proto, kwargs)
      if not self.inplace:
         df = df.copy()
      r = self.node(df, **kwargs)
      if not self.append:
         return r
      
      else:
         if isinstance(r, Series):
            rd:DataFrame = DataFrame({r.name:r})
         elif isinstance(r, DataFrame):
            rd:DataFrame = r
         else:
            raise TypeError(f'Expected either a DataFrame or a Series, got {np.typename(r)}')
         
         if len(rd.dropna()) == 0:
            raise ValueError(f'Invalid computation: empty dataframe returned')
         
         for c in rd.columns:
            df[c] = rd[c]
         
         return df
      
class ApplyNodes:
   def __init__(self, nodes:List[ApplyNode]):
      self.nodes = nodes[:]
      self.silent = False
      
   def __call__(self, df:DataFrame, **kwargs):
      r = df.copy()
      for i, fn in enumerate(self.nodes):
         try:
            r = fn(r, **kwargs)
         except Exception as fatalError:
            if self.silent:
               return None
            else:
               raise fatalError
      return r

ta_method_names = [
   'momentum.ao',
   'momentum.apo',
   'momentum.bias',
   'momentum.bop',
   'momentum.brar',
   'momentum.cci',
   'momentum.cfo',
   'momentum.cg',
   'momentum.cmo',
   'momentum.coppock',
   'momentum.cti',
   'momentum.dm',
   'momentum.er',
   'momentum.eri',
   'momentum.fisher',
   'momentum.inertia',
   'momentum.kdj',
   'momentum.kst',
   'momentum.macd',
   'momentum.mom',
   'momentum.pgo',
   'momentum.ppo',
   'momentum.psl',
   'momentum.pvo',
   'momentum.qqe',
   'momentum.roc',
   'momentum.rsi',
   'momentum.rsx',
   'momentum.rvgi',
   'momentum.slope',
   'momentum.smi',
   'momentum.squeeze',
   'momentum.squeeze_pro',
   'momentum.stc',
   'momentum.stoch',
   'momentum.stochrsi',
   'momentum.td_seq',
   'momentum.trix',
   'momentum.tsi',
   'momentum.uo',
   'momentum.willr',
   'volatility.aberration',
   'volatility.accbands',
   'volatility.atr',
   'volatility.bbands',
   'volatility.donchian',
   'volatility.hwc',
   'volatility.kc',
   'volatility.massi',
   'volatility.natr',
   'volatility.pdist',
   'volatility.rvi',
   'volatility.thermo',
   'volatility.true_range',
   'volatility.ui',
   'volume.ad',
   'volume.adosc',
   'volume.aobv',
   'volume.cmf',
   'volume.efi',
   'volume.eom',
   'volume.kvo',
   'volume.mfi',
   'volume.nvi',
   'volume.obv',
   'volume.pvi',
   'volume.pvol',
   'volume.pvr',
   'volume.pvt',
   'volume.vp',
   'overlap.alma',
   'overlap.dema',
   'overlap.ema',
   'overlap.fwma',
   'overlap.hilo',
   'overlap.hl2',
   'overlap.hlc3',
   'overlap.hma',
   'overlap.hwma',
   'overlap.ichimoku',
   'overlap.jma',
   'overlap.kama',
   'overlap.linreg',
   'overlap.ma',
   'overlap.mcgd',
   'overlap.midpoint',
   'overlap.midprice',
   'overlap.ohlc4',
   'overlap.pwma',
   'overlap.rma',
   'overlap.sinwma',
   'overlap.sma',
   'overlap.ssf',
   'overlap.supertrend',
   'overlap.swma',
   'overlap.t3',
   'overlap.tema',
   'overlap.trima',
   'overlap.vidya',
   'overlap.vwap',
   'overlap.vwma',
   'overlap.wcp',
   'overlap.wma',
   'overlap.zlma',
   'statistics.entropy',
   'statistics.kurtosis',
   'statistics.mad',
   'statistics.median',
   'statistics.quantile',
   'statistics.skew',
   'statistics.stdev',
   'statistics.tos_stdevall',
   'statistics.variance',
   'statistics.zscore',
]

ta_methods = {name:dotget(ta, after(name, '.'), dotget(ta, name)) for name in ta_method_names}

from pprint import pprint
pprint(ta_methods)

special_cases = {}#TODO

def mkNode(ref:str, fn):
   fn_categ, fn_name = ref.split('.', maxsplit=2)
      
   if isfunction(fn):
      sig = signature(fn)
      inputs = []
      config = []
      
      reserved_names = 'open,high,low,close,volume'.split(',')
      for arg_name, arg in sig.parameters.items():
         if arg_name in reserved_names:
            inputs.append(arg)
         else:
            config.append(arg)
         # print(ref, arg_name, arg.kind)
         
      fn_ref = fn.__qualname__
      
      node = Node(fn_ref, fn, inputs, config)
      return node
   
   raise TypeError(fn)

@once
def make_indicator_nodes()->Dict[str, Union[Node, Dict[str, Node]]]:
   reserved_names = 'open,high,low,close,volume'.split(',')
   inodes = {}
   inode_categs = {}
   
   for ref, fn in ta_methods.items():
      fn_categ, fn_name = ref.split('.', maxsplit=2)
      
      if isfunction(fn):
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
         
         node = Node(fn_ref, fn, inputs, config)
         inodes[ref] = node
         inodes[fn_name] = node
         
         inode_categs.setdefault(fn_categ, {})[fn_name] = node
         
      else:
         raise TypeError(f'{ref}:{type(fn).__name__}')
         
   inodes.update(inode_categs)
   
   return inodes
         
indicator_nodes = make_indicator_nodes()

import features.ta.volatility as fta
indicator_nodes['volatility.bbands'] = mkNode('volatility.bbands', fta.bbands)
# indicator_nodes['volatility.donchian'] = mkNode('volatility.donchian', fta.donchian)

from copy import deepcopy

class P:
   name:Optional[str]
   ptype:Optional[Type]
   default:Optional[Any]
   
   def __init__(self, ptype:Optional[Type]=None, pvalues=None, default=None):
      self.name = None
      self.ptype = ptype
      self.pvalues = pvalues
      self.default = default
      
   def named(self, n:str):
      c = deepcopy(self)
      c.name = n
      return c
      
class Ps:
   d:Dict[Union[int, str], P]
   
   def __init__(self, **decls):
      self.d = {}
      
      self._extend(decls)
         
   def _extend(self, o:dict):
      for k, p in o.items():
         assert isinstance(k, str)
         
         if isinstance(p, P):
            self.d[k] = p.named(k)
         
         else:
            t = type(p)
            p = P(t, p)
            p.name = k
            self.d[k] = p
      # return self      

parameter_space_proto = Ps(
   mamode=[
      "dema", "ema", "fwma", "hma", "linreg", "midpoint", "pwma", "rma",
      "sinwma", "sma", "swma", "t3", "tema", "trima", "vidya", "wma", "zlma"
   ],
   length=[3, 7, 14, 20]
)

def mkAnalyzer(*indicators):
   ind_list = [i for i in indicators if isinstance(i, str)]
   for k in ind_list:
      assert k in indicator_nodes, f'{k} is not in indicator_nodes'
   
   features = [ApplyNode(fn) for fn in gets(indicator_nodes, *ind_list) if isinstance(fn, Node)]
   
   for feat in features:
      print(feat)
      
   # for x in indicators:
   #    if isinstance(x, str):
   #       features.append(ApplyNode(indicator_nodes[x]))
   #    else:
   #       raise ValueError(f'Unhandled {x}')
   
   momo = ApplyNodes(features)
   
   return momo
         
if __name__ == '__main__':
   # nodes = indicators()
   from main import load_frame
   
   df = load_frame('AMZN').drop(columns=['datetime']).fillna(method='ffill').fillna(method='bfill').dropna()
   init_cols = set(df.columns)
   
   ind_list = ['volatility.bbands', 'volatility.donchian']
   for k in ind_list:
      assert k in indicator_nodes, f'{k} is not in indicator_nodes'
   # ta.bbands
   features = [ApplyNode(fn) for fn in gets(indicator_nodes, *ind_list) if isinstance(fn, Node)]
   for feat in features:
      print(feat(df).dropna())
      print(feat)
   momo = ApplyNodes()
      
   
   
else:
   # raise Exception(__name__)
   pass