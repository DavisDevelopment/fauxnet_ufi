import numpy as np
import pandas as pd
#import modin.pandas as pd
import os
import sys
import math
import random
from enum import Enum
import re
from functools import lru_cache
from cytoolz import *
from fn import F, _

_DSBase = '/home/ryan/Documents/data'
_CacheDir = '/home/ryan/Documents/data/ds_cache'
if not os.path.exists(_CacheDir):
   os.makedirs(_CacheDir)

class DatasetKind(Enum):
   Crypto = 0
   Stock = 1
   Ext = 2
   
   # @classmethod
   def dir(self):
      return [
         f'{_DSBase}/crypto-dataset',
         f'{_DSBase}/stock-data/1 Day/Stocks'
      ][self.value]
      
   @property
   def ext(self):
      return ['csv', 'us.txt'][self.value]


Crypto = DatasetKind.Crypto
Stock = DatasetKind.Stock
Ext = DatasetKind.Ext

def get_cache_path(name, kind):
   fname = f'{name}.{kind.name.lower()}.csv'
   cache_path = os.path.join(_CacheDir, fname)
   return cache_path

def get_csv_path(name, kind:DatasetKind):
   if kind == Ext:
      return None
   return os.path.join(kind.dir(), f'{name}.{kind.ext}')

def load_cached_dataset(name, kind=Crypto, **kwargs):
   cache_path = get_cache_path(name, kind)
   if os.path.exists(cache_path):
      df = pd.read_feather(cache_path)
      df.index = df['time']

      return df
   else:
      raise Exception('cached dataset file not found')

class Symbol:
   __slots__ = ['name', 'kind', 'path']
   
   def __init__(self, name:str, kind:DatasetKind):
      self.name = name
      self.kind = kind
      self.path = get_csv_path(self.name, self.kind)
      
   def __str__(self):
      return self.name
   
   def __repr__(self):
      return f'Symbol("{self.name}", {repr(self.kind)})'

def load_dataset_(name='btcusd', kind=DatasetKind.Crypto, maxrows=None):

   base_columns = {
       Stock: 'Date,Open,High,Low,Close,Volume,OpenInt'.split(','),
       Crypto: ''
   }
   
   path = Symbol(name, kind).path
   if not os.path.exists(path):
      raise Exception('dataset file not found for {}'.format(dict(name=name, kind=kind)))
   
   df = pd.read_csv(path)
   if kind is Stock:
      col_remap = {k: k.lower() for k in base_columns[Stock]}
      col_remap.update({'Date': 'time'})
      df = df.rename(columns=col_remap)
      df.time = pd.to_datetime(df.time)
   else:
      df.time = pd.to_datetime(df.time, unit='ms')
   df = df.reset_index()
   df.index = df.time

   CACHE = True
   if CACHE:
      try:
         df2 = df.reset_index(inplace=False)
      except:
         df2 = df.reset_index(inplace=False, drop=True)
      df2.to_feather(get_cache_path(name, kind))

   if maxrows is not None:
      df = df.tail(maxrows)

   return df

from typing import *

ldExts = {}

class LDExt:
   def get(self, name:str, **kwargs):
      raise NotImplemented()
   
   def exists(self, name:str, **kwargs)->bool:
      raise NotImplemented()

   def symbols(self)->Iterable[str]:
      raise NotImplemented()
   
   def cache_name(self, name:str, **kwargs)->str:
      raise NotImplemented()

def ld_ext(ext:str, name:str, **kwargs):
   e:LDExt = ldExts[ext]
   return e.get(name, **kwargs)

def register(prefix:str, ext:LDExt):
   if isinstance(ext, type):
      return register(prefix, ext())
   assert isinstance(ext, LDExt), TypeError('ext argument must be an extension of ds.datasets.LDExt')
   ldExts[prefix] = ext

import re
ext_symbol = re.compile('([\w\d_\-.]+)/([\w\d_\-.]+)')
def ext_load_dataset(name=None, kind=None, resampling='D', **kwargs):
   if name is None:
      return None
   m = ext_symbol.match(name)
   if m is None:
      return None
   else:
      extname, name = m.group(1), m.group(2)
      if extname in ldExts.keys():
         return ld_ext(extname, name, resampling=resampling, **kwargs)
      else:
         raise NameError(f'No datasets extension named "{extname}"')

def load_dataset(name=None, kind=None, resampling='D', **kwargs):
   res = ext_load_dataset(name=name, kind=kind, resampling=resampling, **kwargs)
   if res is not None:
      return res
   
   kind = kind or Crypto

   def ret(name=None, kind=None, **kwargs):
      try:
         df = load_cached_dataset(name=name, kind=kind, **kwargs)
         return df
      except:
         return load_dataset_(name=name, kind=kind, **kwargs)

   retries = [
      lambda: ret(name=name, kind={Stock:Crypto,Crypto:Stock}.get(kind), **kwargs),
      lambda: ret(name='{}usd'.format(name), kind=kind, **kwargs)
   ]

   df = None

   try:
      df = load_cached_dataset(name, kind=kind, **kwargs)
   except:
      pass

   try:
      df = load_dataset_(name=name, kind=kind, **kwargs)
   except Exception as e:
      for f in retries:
         try:
            df = f()
         except:
            continue
      if df is None:
         raise e
   
   if resampling is not None:
      g = df.resample(resampling)
      df = g.mean(numeric_only=True)
   
   return df

def _all_available_ext_symbols(prefix=None):
   if prefix is not None:
      if prefix.endswith('/'):
         prefix = prefix[:-1]
      e = ldExts[prefix]
      
      wslash = lambda s: f'{prefix}/{s}'
      return list(map(F(wslash) >> (lambda s: Symbol(s, Ext)), e.symbols()))
   
   from itertools import chain
   
   allexts:Dict[str, LDExt] = ldExts
   symiters = []
   
   for prefix, e in allexts.items():
      wslash = lambda s: f'{prefix}/{s}'
      symiters.append(map(F(wslash) >> (lambda s: Symbol(s, Ext)), e.symbols()))
   syms = chain(*symiters)
   return list(syms)

# @lru_cache
def _all_available_symbols(kind=None):
   if kind is not None:
      if isinstance(kind, str):
         return _all_available_ext_symbols(prefix=kind)
      
      elif kind is DatasetKind.Ext:
         return _all_available_ext_symbols()
      
      fnames = os.listdir(kind.dir())
      pattern = re.compile(r"(.+)\.(?:us\.txt|csv)")
      matches = [pattern.match(f) for f in fnames]
      symbols = []
      for m in matches:
         if m is not None:
            symbols.append(m.group(1))
      return [Symbol(n, kind) for n in symbols]
   
   else:
      all_ = (_all_available_symbols(kind=Stock) + _all_available_symbols(kind=Crypto) + _all_available_symbols(kind=Ext))
      setattr(all_available_symbols, 'result', all_)
      return all_
   
def all_available_symbols(kind=None):
   r = getattr(all_available_symbols, 'result', None)
   if r is not None:
      return list(map(lambda s:s.name, r))
   else:
      return [s.name for s in _all_available_symbols(kind=kind)]

if __name__ == '__main__':
   load_dataset('aapl', kind=Stock)
   load_dataset('btc')

   print(all_available_symbols(kind=Stock))
   print(all_available_symbols(kind=Crypto))

   import random
   from multiprocessing import Pool

   pool = Pool(12)
   symbols = all_available_symbols()
   random.shuffle(symbols)

   def dothestuff(sym):
      print(sym)
      df = load_dataset(sym)
      assert (c in df.columns for c in 'open,close,high,low,time,volume'.split(','))
      return None
   res = pool.map(dothestuff, symbols)
   # pool.join()
