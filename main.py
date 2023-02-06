
import pickle
from time import sleep

from numpy import asanyarray, ndarray
from nn.arch.transformer.TimeSeriesTransformer import TimeSeriesTransformer
from nn.common import *
from datatools import ohlc_resample, renormalize, unpack
   
from nn.namlp import NacMlp
from cytoolz import *
import pandas as pd

from datatools import get_cache, calc_Xy, load_ts_dump
from nn.data.core import TwoPaneWindow, TensorBuffer
from pandas import DataFrame
from typing import *
# from nn.arch.transformer.TransformerDataset import TransformerDataset, generate_square_subsequent_mask
from tools import unzip
from nn.arch.lstm_vae import *
import torch.nn.functional as F

def list_stonks(stonk_dir='./stonks'):
   from pathlib import Path
   return [n[:-8] for n in Path(stonk_dir).rglob('*.feather')]

def load_frame(sym:str):
   df:DataFrame = pd.read_feather('./stonks/%s.feather' % sym)
   print(df.columns)
   return df.set_index('datetime', drop=False)

df = load_frame('AAPL')[['open', 'high', 'low', 'close', 'volume']]
data = df.to_numpy()

print(data)

enc_seq_len = 150
dec_seq_len = 30

out_seq_len = 180

win = TwoPaneWindow(180, 30)
win.load(data)
print(win)

seq_pairs = list(win.iter())
print(len(seq_pairs), 'sequences loaded')
X, y = [list(_) for _ in unzip(seq_pairs)]
X, y = np.asanyarray(X), np.asanyarray(y)
Xb, yb = torch.from_numpy(X), torch.from_numpy(y)
Xb, yb = Xb.float(), yb.float()

from nn.ts.classification import LinearBaseline, FCNBaseline, InceptionModel, ResNetBaseline

# LinearBaseline()