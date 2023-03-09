import math
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import Series, DataFrame
import torch
from torch import Tensor, tensor, from_numpy
from .core import TensorBuffer, TwoPaneWindow
from main import prep_data_for, ExperimentConfig, load_frame

import pickle
from typing import *
from tools import unzip

def percent_change(arr):
   result = np.diff(arr)/arr[1:]*100
   return result

def pl_binary_labeling(y):
   labels = np.zeros((len(y), 2))
   
   ydelta = percent_change(y)
   thresh = 0.8
   
   E = np.argwhere((ydelta <= thresh)&(ydelta >= -thresh))
   P = np.argwhere(ydelta > thresh)
   L = np.argwhere(ydelta < -thresh)
   
   labels[L, 0] = 1.0 # loss (-1)
   labels[P, 1] = 1.0 # profit (1)
   
   return labels

def prep_frame(df:DataFrame, config:ExperimentConfig):
   from sklearn.preprocessing import MinMaxScaler
   
   df = df.copy()
   # df['delta_pct'] = df['close'].pct_change()
   
   data:ndarray = df.to_numpy()[1:]
   
   num_input_channels, in_seq_len, out_seq_len = config.num_input_channels, config.in_seq_len, config.out_seq_len
   assert None not in (in_seq_len, out_seq_len)
   win = TwoPaneWindow(in_seq_len, out_seq_len) #type: ignore
   win.load(data)


   scaler = MinMaxScaler()
   scaler.fit(data[:, :])

   seq_pairs = list(win.iter(lfn=lambda x: scaler.transform(x[:, :])))
   Xl, yl = [list(_) for _ in unzip(seq_pairs)]
   Xnp:ndarray
   ynp:ndarray 
   Xnp, ynp = np.asanyarray(Xl), np.asanyarray(yl).squeeze(1)

   ynp = ynp
   ynp:ndarray = pl_binary_labeling(ynp[:, 3])
   Xnp, ynp = Xnp, ynp

   X, y = torch.from_numpy(Xnp), torch.from_numpy(ynp)
   X, y = X.float(), y.float()

   X = X.swapaxes(1, 2)
   randsampling = torch.randint(0, len(X), (len(X),))
   X, y = X[randsampling], y[randsampling]

   test_split = config.val_split
   spliti = round(len(X) * test_split)
   X_test, y_test = X[-spliti:], y[-spliti:]
   X, y = X[:-spliti], y[:-spliti]

   X_test, X_train, y_test, y_train = X, X_test, y, y_test
   
   return X_train, y_train, X_test, y_test

def polysymbolic_dataset(name:str, **kwargs):
   symbols, idk, data = pickle.load(open('sp100_daily.pickle', 'rb'))
   # print(type(data[0]), type(data[1]), type(data[2]))
   symbols:List[str]
   data:Dict[str, DataFrame]
   proto:ExperimentConfig = ExperimentConfig(
      in_seq_len = kwargs.pop('in_seq_len', 14),
      num_input_channels = kwargs.pop('num_input_channels', 4),
      num_predicted_classes=2,
      epochs=12,
      val_split=0.4
   ) if 'proto' not in kwargs else kwargs['proto']
   
   train_x, train_y, test_x, test_y = [], [], [], []
   sizes = (math.inf, math.inf, math.inf, math.inf)
   
   for sym, df in data.items():
      xa, ya, xb, yb = prep_frame(df, proto)
      
      s1, s2, s3, s4 = sizes
      sizes = (
         min(len(xa), s1),
         min(len(ya), s2),
         min(len(xb), s3),
         min(len(yb), s4)
      )
      
      train_x.append(xa)
      train_y.append(ya)
      test_x .append(xb)
      test_y .append(yb)
   
   def trim(arr:List[Tensor], size:int):
      return [a[0:size] for a in arr]
   
   def pack(arr:List[Tensor], i:int)->Tensor:
      ndarrays = [a.numpy().astype('float32') for a in arr]
      ndarrays = trim(ndarrays, int(sizes[i]))
      nda = np.stack(ndarrays)
      return from_numpy(nda.astype('float32'))
   
   vars = (train_x, train_y, test_x, test_y)
   train_x, train_y, test_x, test_y = tuple(pack(v, i) for i, v in enumerate(vars))
   
   train_x = train_x.swapaxes(1, 2)
   test_x = test_x.swapaxes(1, 2)
   
   return train_x, train_y, test_x, test_y

def aggds(config:ExperimentConfig, symbols):
   train_x, train_y, test_x, test_y = [], [], [], []
   num_input_channels = config.num_input_channels
   
   from nn.data.sampler import add_indicators
   
   for sym in symbols:
      df = load_frame(sym)
      df = add_indicators(df)
      
      cfg = config.extend(num_input_channels=len(df.columns))
      xa, ya, xb, yb = prep_frame(df, cfg)
      
      train_x.append(xa.numpy())
      train_y.append(ya.numpy())
      test_x .append(xb.numpy())
      test_y .append(yb.numpy())
      
      num_input_channels:int = len(df.columns)
   
   vars = (train_x, train_y, test_x, test_y)   
   buffers = (a, b, c, d) = (
      TensorBuffer(10000, (num_input_channels, config.in_seq_len), dtype='float32'),
      TensorBuffer(10000, (config.num_predicted_classes,), dtype='float32'),
      TensorBuffer(10000, (num_input_channels, config.in_seq_len), dtype='float32'),
      TensorBuffer(10000, (config.num_predicted_classes,), dtype='float32'),
   )
   
   for i, arr in enumerate(vars):
      for item in arr:
         buffers[i].push(item)
   
   train_x, train_y, test_x, test_y = buffers
   rs_x = torch.randint(0, len(train_x), (len(train_x),))
   rs_y = torch.randint(0, len(train_y), (len(train_x),))
   
   vars = (train_x.T[rs_x], train_y.T[rs_y], test_x.T[rs_x], test_y.T[rs_y])
   
   train_x, train_y, test_x, test_y = tuple(map(lambda x: x.float(), vars))
   
   return config.extend(num_input_channels=num_input_channels), train_x, train_y, test_x, test_y