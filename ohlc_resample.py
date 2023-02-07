import pandas as pd
from pandas import DataFrame
import numpy as np
from torch import saddmm

from typing import *
from numba import jit, njit

from cytoolz import *
from cytoolz.functoolz import compose, compose_left

def idx_seq_reduce(idxseq):
   idxs = idxseq.to_list()
   assert ensureallequal(idxs)
   return idxs[0]

def group_name(G:Callable[[DataFrame], pd.Series], df:DataFrame)->Any:
   return idx_seq_reduce(G(df))

def resample_ohlc(df:pd.DataFrame, grouping=None, freq=None, dtcol='datetime'):
   df = df.copy()
   
   def sampler(sub_df: DataFrame):
      # idx = sub_df['_G'].iloc[0]
      O = sub_df['open'].iloc[0]
      H = sub_df['high'].max()
      L = sub_df['low'].min()
      C = sub_df['close'].iloc[-1]
      
      return pd.Series(data=[O, H, L, C], index=['open', 'high', 'low', 'close'])
   
   if grouping is not None:
      G = grouping
      if G is None:
         G = lambda dtc: dtc.dt.date
      
      df['_G'] = G(df[dtcol])
      
      grouped = df.groupby(df['_G'])
      
      return grouped.apply(sampler)
   elif freq is not None:
      resampler = df.resample(freq)
      
      return resampler.agg(sampler)

def ensureallequal(a:Iterable[Any]):
   a = list(a)
   
   l = a[0]
   for i in range(len(a)-1):
      if l != a[1+i]:
         return False
      l = a[1+i]

   return True