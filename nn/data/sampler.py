import math
import numpy as np
from numpy import ndarray
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

def add_indicators(df:DataFrame):
   import pandas_ta as ta
   # from pandas_ta import 
   # indicators = df.ta.indicators()
   dc = df.ta.donchian(inplace=True)
   rsi = df.ta.rsi(inplace=True)
   bbands = df.ta.bbands(inplace=True)
   
   indicators = [dc, ('rsi', rsi), bbands]
   
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
   df = df.dropna()
   print(df)
   return df

class DataFrameSampler:
   df: DataFrame
   ignore:List[str]
   preprocessing_funcs:List[Callable[[DataFrame], DataFrame]]
   
   def __init__(self, df:DataFrame, ignore=[], config={}):
      self.df = df
      self.ignore = ignore
      self.preprocessing_funcs = [add_indicators]
      self.config = config
      
   def configure(self, **kwargs):
      self.config = merge(self.config, kwargs)
      return self
      
   def samples(self):
      df = self.df.copy()
      df = compose_left(*self.preprocessing_funcs)(df)
      
      data:ndarray = df.to_numpy()

      scaler = MinMaxScaler()
      scaler.fit(data)
      
      # p, l = count_classes(data)
      # print(f'Loaded dataset contains {p} P-samples, and {l} L-samples')

      #*TODO define ModelConfig class to pass around instead of duplicating these variables endless all around town
      in_seq_len = self.config.get('in_seq_len', 14)
      out_seq_len = 1
      
      for i in range(1+in_seq_len, len(data)-out_seq_len-1):
         tstamp = df.index[i]
         
         X = data[i-in_seq_len:i]
         X = scaler.transform(X)
         X = from_numpy(X)
         X = X.unsqueeze(0).float().swapaxes(1, 2)
         
         yield tstamp, X