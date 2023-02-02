from collections import UserString
from time import sleep
from numba import jit, float32
from numba import njit as _njit

import numpy as np
from typing import *
import os
import sys
import math
import random
import re
P = os.path

from numpy import ndarray

base = P.dirname(os.getcwd())
print(base)
sys.path.append(base)

from tqdm import tqdm
import pandas as pd


# from sklearn.preprocessing import MinMaxScaler, minmax_scale
from tools import unzip
from cytoolz import *
# from kraken import krakenClient
from pathlib import Path
import torch

def renormalize(n, r_a, r_b):
   delta1 = r_a[1] - r_a[0]
   delta2 = r_b[1] - r_b[0]
   return (delta2 * (n - r_a[0]) / delta1) + r_b[0]
 
def krakenClient():
   from keychain import getkeys
   import krakenex
   from pykrakenapi import KrakenAPI
   
   keys = getkeys()
   api = krakenex.API(key=keys['kraken'][0], secret=keys['kraken'][1])
   k = KrakenAPI(api)
   
   return k

client = krakenClient()

kpairs = pd.read_csv('./kraken_pairs.csv')
names = kpairs[kpairs.columns[0]]
kpairs = kpairs.drop(columns=[kpairs.columns[0]])
kpairs['names'] = names
kpairs.set_index('names', inplace=True)

def get_pair_row(l:str, r:str):
   for pair_id, pair in kpairs.iterrows():
      (pl, pr) = pair.wsname.split('/')
      if (l, r) == (pl, pr):
         return pair_id, pair
   return None

def exchange_vectors(sym:str):
   for pair_id, pair in kpairs.iterrows():
      (pl, pr) = pair.wsname.split('/')
      if (sym in (pl, pr)):
         yield (pl, pr)
      
class Symbol:
   value_from:Optional[str] = None
   name:str
   
   def __init__(self, name:str, value_from:Optional[str]=None):
      self.name = name
      self.value_from = value_from
      
   def __str__(self):
      return self.name
   
primary_currency = 'USD'
secondary_currency = 'ETH'

def symbols(names:Iterable[str]):
   names = set(names)
   
   for pair_id, pair in kpairs.iterrows():
      (pl, pr) = pair.wsname.split('/')
      if pl in names:
         name = pl
      elif pr in names:
         pl, pr = pr, pl
         name = pl
      else:
         continue
      
      if pr in (primary_currency, secondary_currency):
         yield pair_id, Symbol(name, value_from=pr)

# l, r = 
l, r = map(set, unzip(kpairs.wsname.astype('string').str.split('/').to_numpy()))
kraken_symbols = (l | r)

stablecoins = set([
   'USDT', 'USDC', 'UST',
   'WBTC', 'TBTC'
])

fiat_currencies = set(('JPY', 'AUD', 'CAD', 'EUR', 'GBP', 'CHF', 'AED', 'USD'))

from pprint import pprint
selected = (kraken_symbols - stablecoins - fiat_currencies)
# selected_symbols = list(map(symbol, selected))
selected_symbols = []
sym2id = {}

notnone = lambda x: (x is not None)

for id, sym in symbols(selected):
   sym2id[sym.name] = id

def build_cache(saveto=None):
   prog = tqdm(selected_symbols)
   df_cache = {}
   
   for sym in prog:
      pair_id = sym2id[str(sym)]
      if pair_id is None:
         print(f'No pair found for "{sym}"')
         continue
      ohlc, _ = client.get_ohlc_data(pair_id, interval=1440)
      df_cache[str(sym)] = ohlc
      
   if saveto is not None:
      import pickle
      with open(saveto, 'wb+') as f:
         pickle.dump(df_cache, f)
      
   return df_cache

def get_cache(p='./.kcache/all.pickle'):
   import pickle
   
   if not os.path.exists(p):
      cache = build_cache(saveto=p)
   else:
      cache = pickle.load(open(p, 'rb'))
      
   return cache

def get_cache_buffer(symbols:Iterable[str], columns:Iterable[str], cache:Optional[Dict[str, pd.DataFrame]]=None, select=None, n_samples=365, tolerant=False):
   if cache is None:
      cache = get_cache()
   symbols = set(symbols)
   for sym in symbols:
      if sym not in cache:
         # cache[sym] = client.get_ohlc_data(sym2id[str(sym)], interval=1440)
         if tolerant:
            symbols.remove(sym)
         else:
            raise ValueError(f'Symbol("{sym}") not present in the cache')
   
   loaded = []
   scalers = []
   buffer = []
   assert columns is not None and len(columns) != 0, 'must provide column names via columns argument'
   if isinstance(columns, str):
      columns = list(map(lambda s: s.strip(), columns.split(','))) if ',' in columns else columns.split(' ')
   
   for symbol in symbols:
      df = cache[symbol]
      if not all((c in df.columns) for c in columns):
         raise ValueError(f'DataFrame for "{symbol}" is missing column "{column}"')
      
      if len(df) < n_samples:
         if tolerant:
            # df = df.copy(deep=True)
            # ts = df.index.to_numpy()
            new_index = pd.date_range(df.index.max() - pd.DateOffset(days=n_samples), df.index.max())
            dummy_ohlc = np.full((n_samples - len(df), len(df.columns)), pd.NA)
            df = pd.DataFrame(data=np.vstack((dummy_ohlc, df.to_numpy())), index=new_index, columns=columns)
         else:
            raise ValueError(f'.. on "{symbol}", expected at least {n_samples} rows, got {len(df)}')
      else:
         df = df[columns].tail(n_samples)
      if select is not None:
         df = select(df)
      loaded.append(symbol)
      
      dfnp = df.to_numpy()
      
      scaler = MinMaxScaler()
      scalers.append(scaler)
      avg = np.mean(dfnp[:, :-1], axis=1).reshape(-1, 1)
      scaler.fit(avg)
      # dfnp = minmax_scale(dfnp)
      dfnp = renormalize(dfnp, (dfnp.min(), dfnp.max()), (0.0, 1.0))
      buffer.append(dfnp)
   
   np_buffer = np.asanyarray(buffer, dtype='float32')
     
   return loaded, scalers, np_buffer

def calc_Xy(cache: Dict[str, pd.DataFrame], columns=None, nT_in=365, nT_out=60):
   X = []
   y = []
   
   all_symbols = list(set(cache.keys()))
   symbols = []
   assert columns is not None
   nC = len(columns)
   
   buffers = []
   
   for k, sym in enumerate(all_symbols):
      df = cache[sym][columns].tail(nT_in + nT_out)
      df = df.fillna(method='ffill').fillna(method='bfill').dropna()
      df_np:ndarray = df.to_numpy()
      
      if len(df_np) >= nT_in+nT_out:
         # if len(df_np) > nT_in:
            # df_np = df_np[:-nT_in]
         buffers.append(df_np)
         symbols.append(sym)
      else:
         continue
   
   # assert all(len(a)==nT_in for a in buffers)
   
   nK = len(symbols)
   ground = np.zeros((nK, nC))
   for k in range(nK):
      ground[k] = buffers[k][-nT_out-1, :]
   
   b_ranges = []
   move_ranges = []
   X = np.zeros((nK, nT_in, nC))
   y = np.zeros((nK, nT_out, nC))
   
   #* scale the buffers
   for k in range(len(symbols)):
      buffer = buffers[k]
      _min, _max = buffer.min(), buffer.max()
      b_ranges.append((_min, _max))
      buffer = buffers[k] = renormalize(buffer, (_min, _max), (0.0, 1.0))
      
      moves = np.diff(buffer, axis=0)
      move_range = (moves.min(), moves.max())
      moves = renormalize(moves, move_range, (0.0, 1.0))
      move_ranges.append(move_range)
      
      assert len(buffer) == nT_in+nT_out
      _x = buffer[:nT_in]
      _y = buffer[-nT_out:]
      X[k] = _x
      y[k] = _y
      
      print(f'x[k].shape={_x.shape}', f'y[k].shape={_y.shape}')
      
   return (
      symbols,
      b_ranges,
      buffers,
      move_ranges,
      ground,
      X,
      y
   )

import pickle

def load_ts_dump(from_folder='./nasdaq_100'):
   man = pickle.load(open(P.join(from_folder, 'manifest.pickle'), 'rb'))
   syms = list(man.keys())
   result = {}
   
   for sym in syms:
      df:pd.DataFrame = pd.read_feather(P.join(from_folder, f'{sym}.feather'))
      df.set_index('datetime', inplace=True, drop=True)
      df = df.resample('D').mean()
      df = df.interpolate(method='linear').fillna(method='bfill').dropna()
      result[sym] = df
   
   return result