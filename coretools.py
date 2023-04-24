from olbi import printing

import itertools
from functools import reduce, wraps
import pickle
from time import sleep
from pprint import pprint
from numpy import asanyarray, ndarray
import termcolor
from torch import BoolTensor, random, renorm
from torch.jit._script import ScriptModule, ScriptFunction
from coinflip_backtest import backtest, count_classes
from nn.arch.transformer.TimeSeriesTransformer import TimeSeriesTransformer
from nn.common import *
from datatools import P, ohlc_resample, pl_binary_labeling, quantize, renormalize, unpack
   
from nn.namlp import NacMlp
from cytoolz import *
from cytoolz import itertoolz as iters
import pandas as pd

from datatools import get_cache, calc_Xy, load_ts_dump, rescale, norm_batches
from nn.data.core import TwoPaneWindow, TensorBuffer
from pandas import DataFrame
from typing import *
from nn.ts.classification.fcn_baseline import FCNBaseline2D, FCNNaccBaseline
# from nn.arch.transformer.TransformerDataset import TransformerDataset, generate_square_subsequent_mask
from tools import Struct, dotget, gets, maxby, minby, unzip, safe_div, argminby, argmaxby
from nn.arch.lstm_vae import *
import torch.nn.functional as F
from nn.ts.classification import LinearBaseline, FCNBaseline, InceptionModel, ResNetBaseline
from nn.arch.transformer.TransformerDataset import generate_square_subsequent_mask

from sklearn.preprocessing import MinMaxScaler
from ttools.thunk import Thunk, thunkv, thunk

def list_stonks(stonk_dir='./stonks', shuffle=True)->List[str]:
   stonk_dir = './stonks'
   from pathlib import Path
   from random import shuffle
   
   tickers = [str(P.basename(n))[:-8] for n in Path(stonk_dir).rglob('*.feather')]
   shuffle(tickers)
   
   return tickers

from cachetools import cached, Cache, keys

@cached(cache=Cache(200))
def load_frame(sym:str, dir='./stonks')->DataFrame:
   dir = './stonks'
   #* read the DataFrame from the filesystem
   df:DataFrame = pd.read_feather(P.join(dir, '%s.feather' % sym))
   
   #* iterating forward through the frame, replace all NA values with the last non-NA value
   df = df.fillna(method='ffill')
   #* iterating backward through the frame, replace all NA values with the last non-NA value
   df = df.fillna(method='bfill')
   #* drop any remaining rows containing NA values
   df = df.dropna()
   
   #* reindex the frame by datetime
   df = df.set_index('datetime', drop=False)
   df.name = sym
   
   return df

def shuffle_tensors_in_unison(*all, axis:int=0):
   state = torch.random.get_rng_state()
   # if axis == 0:
   #    return tuple(a[a.size()[axis]]for a in all:
         
   inputs = list(all)
   if len(inputs) == 0:
      return []
   input_ndim = inputs[0].ndim
   input_shape = inputs[0].size()
   
   if abs(axis) > input_ndim-1:
      raise IndexError('invalid axis')
   
   accessor = []
   for i in range(0, axis):
      accessor.append(slice(None))
   results = []
   for i, a in enumerate(inputs):
      torch.random.set_rng_state(state)
      results.append(a[(*accessor, torch.randperm(a.size()[axis]))])
   return tuple(results)

# def prep_data_for(config:ExperimentConfig):
#    df:DataFrame = config.df
#    df['delta_pct'] = df['close'].pct_change()
#    data = df.to_numpy()[1:]

#    num_input_channels, in_seq_len, out_seq_len = config.num_input_channels, config.in_seq_len, config.out_seq_len
#    assert None not in (in_seq_len, out_seq_len)
#    win = TwoPaneWindow(in_seq_len, out_seq_len) #type: ignore
#    win.load(data)


#    scaler = MinMaxScaler()
#    scaler.fit(data[:, :])

#    seq_pairs = list(win.iter(lfn=lambda x: scaler.transform(x[:, :-1])))
#    Xl, yl = [list(_) for _ in unzip(seq_pairs)]
#    Xnp:ndarray
#    ynp:ndarray 
#    Xnp, ynp = np.asanyarray(Xl), np.asanyarray(yl).squeeze(1)

#    ynp = ynp
#    ynp:ndarray = pl_binary_labeling(ynp[:, 3])
#    Xnp, ynp = Xnp, ynp

#    X, y = torch.from_numpy(Xnp), torch.from_numpy(ynp)
#    X, y = X.float(), y.float()
#    X, y = shuffle_tensors_in_unison(X, y)

#    X = X.swapaxes(1, 2)
#    randsampling = torch.randint(0, len(X), (len(X),))
#    X, y = X[randsampling], y[randsampling]

#    test_split = config.val_split
#    spliti = round(len(X) * test_split)
#    X_test, y_test = X[-spliti:], y[-spliti:]
#    X, y = X[:-spliti], y[:-spliti]

#    X_test, X_train, y_test, y_train = X, X_test, y, y_test
   
#    return X_train, y_train, X_test, y_test
   
   
