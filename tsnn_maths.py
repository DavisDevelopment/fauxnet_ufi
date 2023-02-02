from time import sleep
from numba import jit, float32
from numba import njit as _njit

import numpy as np
from cytoolz import *
from pathlib import Path
from torch.nn import *
from nalu.layers import NaluLayer
from nalu.core import NaluCell, NacCell
from torch.jit import script, freeze, optimize_for_inference
from torch.nn import Module
import torch.nn.functional as tfn
from torch.autograd import Variable
from torch import Tensor, tensor, asarray, floor, ceil, round, mean, abs
import torch.nn as nn
import torch
from typing import *
import os
import sys
import math
import random
import re
P = os.path

base = P.dirname(os.getcwd())
print(base)
sys.path.append(base)

from tqdm import tqdm
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, minmax_scale

from econn.node import *
from econn.market import Market
from operator import methodcaller, attrgetter
from fn import F, _
from ptmdl.components import BinarizedArithmeticModule
from ptmdl.models.nac import NacCell, NeuralAccumulatorCell
from ptmdl.models.autoclassify import NacAutoClassify
from ptmdl.ops import *

from ds.data import load_dataset, all_available_symbols
from tools import unzip, Struct
from econn.prim import TensorBuffer
import pickle, logging
from pprint import pprint, pformat
from numpy import typename
      
DEBUG = False
logger = logging.getLogger(f'fauxnet:{__name__}')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
logger.addHandler(stdout_handler)

if DEBUG:
   file_handler = logging.FileHandler(f'./fauxnet_logs.log')
   file_handler.setLevel(logging.DEBUG)
   logger.addHandler(file_handler)

NOJIT=True
REJECT_STALE_SAMPLES=False

def njit(f):
   if not NOJIT:
      return _njit(f)
   else:
      return f

from ds.datatools import persistent_memoize
from builtins import int, float
from pprint import pformat

from numba import vectorize

@vectorize
def nzdiv(l, r):
   if l == 0:
      return l
   elif r == 0:
      return r
   return (l / r)


# @persistent_memoize
def load_np_buffer(all_symbols:Iterable[str]=None, symbols:Iterable[str]=None, other_than:Iterable[str]=[], columns:List[str]=None, n_symbols:int=30, n_samples:int=250, no_scale=False):
   other_than = other_than if (other_than is not None) else []
   if all_symbols is not None and symbols is None:
      symbols = set(set(all_symbols) - set(other_than))
   
   loaded = []
   scalers = []
   buffer = []
   
   assert columns is not None and len(columns) != 0, 'must provide column names via columns argument'
   if isinstance(columns, str):
      columns = list(map(lambda s: s.strip(), columns.split(','))) if ',' in columns else columns.split(' ')
   
   for symbol in symbols:
      df:pd.DataFrame = load_dataset(name=symbol, resampling='D')
      
      df = df.fillna(method='ffill').fillna(method='bfill').dropna()
            
      if not all((c in df.columns) for c in columns):
         cols = tuple(df.columns.tolist())
         spec = pformat(df.dtypes.to_dict())
         logger.debug(f'Skipping {symbol} because of missing column {c}')
         continue
      
      if len(df) < n_samples:
         logger.debug(f'Skipping {symbol} because of insufficient available samples')
         continue
      else:
         df = df[columns].tail(n_samples)
      
      today = pd.Timestamp.now().date()
      df['time'] = pd.date_range(start=(today - pd.DateOffset(n_samples - 1)), end=today)
      df.set_index('time', inplace=True)
      
      loaded.append(symbol)
      
      dfnp = df.to_numpy()
      
      scaler = MinMaxScaler()
      scalers.append(scaler)
      avg = np.mean(dfnp[:, :-1], axis=1).reshape(-1, 1)
      scaler.fit(avg)
      dfnp = minmax_scale(dfnp)
      buffer.append(dfnp)
      
      logger.debug(f'Loaded {symbol} ({len(buffer)}/{n_symbols}')
      
      if len(buffer) >= n_symbols:
         break
         
   np_buffer = np.asanyarray(buffer, dtype='float32')
     
   return loaded, scalers, np_buffer

def load_kbuffer(other_than:Iterable[str]=[], columns:List[str]=None, no_scale=False):
   cache:Dict[str, pd.DataFrame] = pickle.load(open('./.kcache/all.pickle', 'rb'))
   for symbol in cache.keys():
      if symbol in other_than:
         continue
      
      df = cache[symbol]
      # df = 
      df = df.fillna(method='ffill').fillna(method='bfill').dropna()
      print(len(df))
   
   return None   

def callseq(layers, X:Tensor)->Tensor:
   v = X
   for l in layers:
      v = l(v)
   return v
   
@njit
def x_from(data:float32[:, :], out:Optional[float32[:, :, :]]=None, outshape:Tuple[int]=(1,), window_size=30):
   nsymbols, nsamples, nchannels = data.shape
   n_window_samples = (nsamples - window_size)
   # outshape = (nsymbols, n_window_samples, window_size, nchannels-1)
   
   if out is None:
      out = np.zeros(outshape, dtype='float32')
   assert out.shape == outshape, f'Invalid output vector, expected {outshape} got {out.shape}'
   
   for i in range(1+window_size, nsamples-1):
      x = data[:, i - 1 - window_size:i-1, :]

      out[:, i-window_size-1] = x
   return out
      
@njit
def y_from(data:float32[:, :], out:Optional[float32[:]]=None, outshape:Tuple[int]=None, window_size=30):
   nsymbols, nsamples, nchannels = data.shape
   n_window_samples = (nsamples - window_size)
   
   if outshape is None:
      outshape = (nsymbols, n_window_samples)
   
   if out is None:
      out = np.zeros(outshape, dtype='float32')
   assert out.shape == outshape, f'Invalid output vector, expected {outshape} got {out.shape}'

   for i in range(1+window_size, nsamples-1):
      y = np.mean(data[:, i, :])

      out[:, i-window_size-1] = y
   return out

@njit
def y2_from(data:float32[:, :], out:Optional[float32[:]]=None, outshape:Tuple[int]=None, window_size=30):
   nsymbols, nsamples, nchannels = data.shape
   n_window_samples = (nsamples - window_size)
   
   if outshape is None:
      outshape = (nsymbols, n_window_samples, nchannels-1)
   
   if out is None:
      out = np.zeros(outshape, dtype='float32')
   assert out.shape == outshape, f'Invalid output vector, expected {outshape} got {out.shape}'

   for i in range(1+window_size, nsamples-1):
      y = data[:, i, :-1]

      out[:, i-window_size-1] = y
   return out

@njit
def samples_from_buffer_a(alldata:float32[:, :, :], window_size=30, n_symbols=30):
   nsets, nsamples, nchannels = alldata.shape
   
   total_window_samples = (nsamples-window_size)*nsets
   nbatches = (nsets // n_symbols)
   
   rem = (nsets % n_symbols)
   if rem != 0: 
      alldata = alldata[:-rem]
   
   batched_alldata = np.split(alldata, (nsets//n_symbols), axis=0)
   xshape = (len(batched_alldata), n_symbols, nsamples-window_size, window_size, nchannels)
   yshape = (len(batched_alldata), n_symbols, nsamples-window_size)
   
   ox = np.zeros(xshape, dtype='float32')
   oy = np.zeros(yshape, dtype='float32')

   for batchIdx, batch in enumerate(batched_alldata):
      y_from(batch, out=oy[batchIdx], outshape=yshape[1:], window_size=window_size)
      x_from(batch, out=ox[batchIdx], outshape=xshape[1:], window_size=window_size)
   
   return ox, oy

@njit
def samples_from_buffer_b(alldata:float32[:, :, :], window_size=30, n_symbols=30):
   nsets, nsamples, nchannels = alldata.shape
   
   total_window_samples = (nsamples-window_size)*nsets
   nbatches = (nsets // n_symbols)
   
   rem = (nsets % n_symbols)
   if rem != 0: 
      alldata = alldata[:-rem]
   
   batched_alldata = np.split(alldata, (nsets//n_symbols), axis=0)
   xshape = (len(batched_alldata), n_symbols, nsamples-window_size, window_size, nchannels-1)
   yshape = (len(batched_alldata), n_symbols, nsamples-window_size, nchannels-1)
   
   ox = np.zeros(xshape, dtype='float32')
   oy = np.zeros(yshape, dtype='float32')

   for batchIdx, batch in enumerate(batched_alldata):
      y2_from(batch, out=oy[batchIdx], outshape=yshape[1:], window_size=window_size)
      x_from(batch, out=ox[batchIdx], outshape=xshape[1:], window_size=window_size)
   
   return ox, oy

def f(a: np.ndarray):
   a = a[:400]
   # print(np.count_nonzero(a[np.isnan(a)]))
   return torch.from_numpy(a)

def samplespp(X, y):
   """
   post processing function for function approximation samples (e.g. X=input, y=output pairs)
   """
   ixshape, iyshape = X.shape, y.shape
   X = X.swapaxes(1, 2).swapaxes(4, 2)
   y = y.swapaxes(1, 2)
   # if y.shape[0] == 1:
   #    y = np.squeeze(y)
   oxshape, oyshape = X.shape, y.shape
   
   print(f'Transformed f({ixshape})->{iyshape}                        =>        f({oxshape})->{oyshape}')
   X, y = f(X), f(y)
   
   return X, y

def load_samples(n_symbols:int=30, window_size:int=14, all_symbols=None, exclude=None, symbols=None, columns=None, set='a', **kwargs):
   # set = set.lower()
   assert set in ('a', 'b')
   selected, scalers, np_buffer = load_np_buffer(all_symbols=all_symbols, other_than=exclude, symbols=symbols, columns=columns, n_symbols=n_symbols, **kwargs)

   get_samples = (samples_from_buffer_a if set == 'a' else samples_from_buffer_b)
   X, y = get_samples(np_buffer, window_size=window_size, n_symbols=n_symbols)
   tX, ty = samplespp(X, y)
   
   return selected, scalers, tX, ty

def load_ksamples(n_samples=365, select=None, columns=['open', 'high', 'low', 'close'], tolerant=False, window_size=40, n_test_samples=10):
   from datatools import get_cache, get_cache_buffer
   cache = get_cache()
   # cache_buffer = get_cache_buffer()
   selected, scalers, np_buffer = get_cache_buffer(cache.keys(), columns, cache=cache, n_samples=n_samples, tolerant=tolerant)
   X, y = samples_from_buffer_a(np_buffer, window_size=window_size, n_symbols=np_buffer.shape[0])
   tX, ty = samplespp(X, y)
   
   X_train = tX[:, :-n_test_samples]
   X_test = tX[:, -n_test_samples:]
   y_train = ty[:, :-n_test_samples]
   y_test = ty[:, -n_test_samples:]
   
   return selected, scalers, X_train, y_train, X_test, y_test

all_symbols = all_available_symbols()
random.shuffle(all_symbols)
sampling_params = Struct(
   all_symbols=all_symbols,
   exclude=None,
   n_symbols=30,
   window_size=40,
   n_samples=1000,
   columns=['open', 'high', 'low', 'close'] #TODO remove currently unused 'volume' column
)

arch_params = Struct(
   n_symbols=sampling_params.n_symbols,
   window_size=sampling_params.window_size,
   columns=sampling_params.columns[:],
   epochs=100
)

arch_params.n_symbols = 30
today = pd.Timestamp.now().date()

selected, scalers, bX, by, eval_X, eval_y = load_ksamples(n_samples=100, window_size=14)
print(bX.shape, by.shape)
arch_params = Struct(
   n_symbols=by.shape[-1],
   window_size=sampling_params.window_size,
   columns=sampling_params.columns[:],
   epochs=25
)
getbatch = lambda i: (bX[i], by[i])

encoder_layers = [
   # Dropout3d(p=0.07),
   Conv2d(4, 4, (4, 4)),
   ReLU(),
   MaxPool2d((2, 2)),
   Conv2d(4, 4, (2, 2)),
   ReLU(),
   MaxPool2d((2, 2)),
   Flatten()
]
encoder = Sequential(*encoder_layers)

n_encoder_terms:int = encoder(bX[0]).shape[1]
arch_params.encoder_output_size = (n_encoder_terms,)

print(f'n_encoder_terms: {n_encoder_terms}')

decoder_a = Sequential(
   Linear(arch_params.encoder_output_size[0], 900), 
   PReLU(), 
   Linear(900, arch_params.n_symbols),
   PReLU()
)

t1 = Sequential(encoder, decoder_a)

def set_lr(o, new_rate:float):
   for g in o.param_groups:
      g['lr'] = new_rate
      
def get_lr(o)->float:
   return tensor([g['lr'] for g in o.param_groups]).mean()
      
#  torch.optim import 

def fit(model, parameters, epochs:int, lr:float=0.0001, scheduler=None):
   optimizer = torch.optim.AdamW(parameters, lr=lr)
   criterion = MSELoss()
   scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
   scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
   pbar = tqdm(range(epochs))
   
   for i in pbar:
      optimizer.zero_grad()
      
      l = Variable(torch.zeros(bX.shape[0], dtype=torch.float32))
      
      for bi in range(bX.shape[0]):
         X, y = getbatch(bi)
         ypred = model(X)
         
         loss = criterion(ypred, y)
         loss.backward()
         
         l[bi] = loss.detach()
      
      optimizer.step()
      # scheduler2.step(l)
      pbar.set_description(f'{i+1}/{epochs})  loss={torch.mean(l).item()}')
      
   return model

tparams = list(encoder.parameters())+list(decoder_a.parameters())

# lrschedule = partial(calc_learning_rate, start=0.00005, end=0.001)
fit(t1, tparams, epochs=50)

y_pred = t1(eval_X.squeeze()).detach()
y_true = eval_y.squeeze()
print(y_pred, y_true)
print(y_pred.shape, y_true.shape)

moe_mean = np.zeros(y_pred.shape[1])
moe_bounds = np.zeros((y_pred.shape[1], 2))
moe_std = np.zeros((y_pred.shape[1]))

for sym in range(y_pred.shape[1]):
   scaler = scalers[sym]
   forecast = y_pred[:, sym].numpy()
   ground_truth = y_true[:, sym].numpy()
   forecast = scaler.inverse_transform(forecast.reshape(1, -1))
   ground_truth = scaler.inverse_transform(ground_truth.reshape(1, -1))
   
   # print(forecast.shape, ground_truth.shape)
   ae = np.abs(forecast - ground_truth)
   moe:ndarray = (nzdiv(ae, ground_truth) * 100.0)
   mmoe = moe_mean[sym] = np.mean(moe)
   (moe_min, moe_max) = moe_bounds[sym] = (moe.min(), moe.max())
   std = moe_std[sym] = np.std(moe)
   
   print(f'{selected[sym]}: {mmoe:.2f}% ({moe_min:.2f}% - {moe_max:.2f}%)')
   
print(f'total moe: {moe_mean.mean():.3f}%')

encoder, decoder_a = encoder.eval(), decoder_a.eval()

torch.save(encoder.state_dict(), './trained_cache/encoder')
torch.save(decoder_a.state_dict(), './trained_cache/decoder_a')


class MultivariateDecoder(Module):
   def __init__(self, encoder:Module, unidecoder:Sequential, nvars:int=4):
      super().__init__()
      
      self.nvars = nvars
      self.enc = encoder
      self.unidec = unidecoder
      
      self.pre_dec = Sequential(
         NacCell((n_encoder_terms, ), (arch_params.n_symbols, 100)),
         PReLU(),
         NacCell((arch_params.n_symbols, 100), (arch_params.n_symbols, 4))
      )
      
   def forward(self, inputs:Tensor):
      print(inputs.shape)
      enc_out = self.enc(inputs)
      print(enc_out.shape)
      ya = self.unidec(enc_out)
      print(ya.shape)
      nbatches = inputs.shape[0]
      
      y = Variable(torch.zeros((nbatches, arch_params.n_symbols, 4)))
      
      for i in range(nbatches):
         y[i] = self.pre_dec(enc_out[i, :])
      
      return y





# allypred = t1(bX.squeeze())
# allytrue = by.squeeze()

# print(allypred.shape, allytrue.shape)

# #TODO also measure the total volatility of the "market" during the evaluated period
# for bi in range(len(bX)):
#    tX = bX[bi]
#    moe = np.zeros((len(selected), len(tX)), dtype=np.float32)
   
#    for ti in range(len(tX)):
#       py = allypred[ti]
      
#       for si, sym in enumerate(selected):
#          scaler = scalers[si]

#          uy_v   = allytrue[ti, si].detach().numpy()
#          puy_v  = py[si].detach().numpy()
         
#          uy_v = scaler.inverse_transform(uy_v.reshape(1, -1))
#          assert len(uy_v[uy_v == 0]) == 0
#          puy_v = scaler.inverse_transform(uy_v.reshape(1, -1))
         
#          absoffset = np.abs(puy_v - uy_v)
#          e = nzdiv(absoffset, uy_v)
#          mmoe = np.mean(e[np.nonzero(e)])
         
#          # print(f'{sym}.moe={mmoe:.8f}')
#          moe[si, ti] = mmoe
         
#    moe /= 100
#    print(f'moe: {np.mean(moe)}, {np.std(moe)}')
# arch_params