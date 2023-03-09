import math
import numpy as np
from numpy import ndarray
from datatools import rescale
import pandas as pd
from pandas import Series, DataFrame
import torch
from torch import Tensor, tensor, from_numpy
from .core import TensorBuffer, TwoPaneWindow
# from main import prep_data_for, ExperimentConfig, load_frame
from sklearn.preprocessing import MinMaxScaler

import pickle
from typing import *
from tools import unzip, gets

from nn.data.agg import *
from cytoolz import *
from cytoolz.dicttoolz import merge
from cytoolz.functoolz import compose_left

def renormalize(n:Series, min1=None, max1=None, min2=None, max2=None):
   if min1 is not None and max1 is not None and min2 is None and max2 is None:
      min2, max2 = min1, max1
      min1, max1 = n.min(), n.max()
   
   assert min1 is not None and max1 is not None, f'first range must be provided via min1 and max1 arguments'
   assert min2 is not None and max2 is not None, f'second range must be provided via min2 and max2 arguments'
   
   delta1 = max1 - min1
   delta2 = max2 - min2
   
   return (delta2 * (n - min1) / delta1) + min2

def donchian_uni(df:DataFrame)->Series:
   dc:DataFrame = df.ta.donchian()
   dcd:ndarray = dc.to_numpy()
   dcd = dcd.T
   channels = list(dc.columns)
   
   lower, upper = dc[channels[0]], dc[channels[2]]
   close = df['close']
   
   value = renormalize(close, lower, upper)
   
   return value

def add_indicators(df:DataFrame):
   import pandas_ta as ta
   # from pandas_ta import 
   # indicators = df.ta.indicators()
   
   #TODO encode Donchian Channels as a single channel (i.e. the position of the close relative to the upper and lower donchian values)
   # dc = df.ta.donchian(inplace=True)
   dc = donchian_uni(df)
   rsi = df.ta.rsi(inplace=True)
   
   #TODO ...and the same goes for the Bollinger Bands
   # bbands = df.ta.bbands(inplace=True)
   
   indicators = [
      ('rsi', rsi),
      ('donchian', dc)
   ]
   
   for ti in indicators:
      if isinstance(ti, DataFrame):
         for c in ti.columns:
            df[c] = ti[c]
      elif isinstance(ti, tuple):
         c, values = ti
         df[c] = np.nan
         df.loc[values.index, c] = values
   
   try:
      df = df.drop(columns=['datetime'])
   except:
      pass
   
   #* drop rows with NA values
   df = df.dropna()
   
   if False:
      print(df)
   
   return df

class DataFrameSampler:
   df: DataFrame
   ignore:List[str]
   preprocessing_funcs:List[Callable[[DataFrame], DataFrame]]
   
   def __init__(self, df:DataFrame, ignore=[], config={}, technical_analysis=True):
      
      self.df = df
      self._df = None
      self.ignore = ignore
      self.preprocessing_funcs = []
      if technical_analysis:
         self.preprocessing_funcs.append(add_indicators)
      
      self.config = merge(dict(
         track_column=3
      ), config)
      
      self._preprocessing_func = (lambda df: df)
      
   def configure(self, **kwargs):
      self.config = merge(self.config, kwargs)
      return self
   
   # @property
   def date_range(self, df:DataFrame=None):
      g = self.config
      df = df if df is not None else (self._df if self._df is not None else self.df)
      
      if df is not None:
         idx = df.index
         min_ts, max_ts = g.get('min_ts', idx.min()), g.get('max_ts', idx.max())
      else:
         min_ts, max_ts = g.get('min_ts', None), g.get('max_ts', None)
         
      return min_ts, max_ts
   
   def extract_raw_array(self, df:DataFrame, scale=True, scaler=None)->ndarray:
      data:ndarray = df.to_numpy()

      if scaler is None:
         scaler = MinMaxScaler()
         scaler.fit(data)
      
      data = scaler.transform(data)
      return data
      
   def samples(self):
      preprocess = self._preprocessing_func = compose_left(*self.preprocessing_funcs)
      df = self.df.copy()
      assert 'volume' in df.columns
      df:DataFrame = preprocess(df)
      self._df = df
      
      data:ndarray = df.to_numpy()
      scaler = MinMaxScaler()
      scaler.fit(data)

      #*TODO define ModelConfig class to pass around instead of duplicating these variables endless all around town
      in_seq_len = self.config.get('in_seq_len', 14)
      track_column = self.config.get('track_column', 3)
      out_seq_len = 1
      min_ts, max_ts = self.date_range(df)
      
      for i in range(1+in_seq_len, len(data)-out_seq_len-1):
         tstamp = df.index[i]
         
         #* enforce date-range constraints
         if min_ts is not None and tstamp < min_ts:
            continue
         
         elif max_ts is not None and tstamp > max_ts:
            continue
         
         X = data[i-in_seq_len:i]
         X = scaler.transform(X)
         X = from_numpy(X)
         X = X.unsqueeze(0).float().swapaxes(1, 2)
         
         assert X.shape[2] == in_seq_len
         
         y_cur = data[i, track_column]
         y_next = data[i+1, track_column]
         y_delta = y_next - y_cur
         y_pct = y_delta / y_cur * 100.0
         
         #TODO return the classification for `y_pct` instead of returning `y_pct`
         
         yield tstamp, X, y_pct
         
      def pack(self):
         times = []
         X = []
         Y = []
         
         for ts, x, y in self.samples():
            times.append(ts)
            X.append(x)
            Y.append(y)
         
         times = np.asarray(times)
         X = np.asanyarray(X)
         Y = np.asarray(Y)
         
         return tuple(torch.from_numpy(v) for v in (times, X, Y))