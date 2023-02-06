
import itertools
import pickle
from time import sleep

from numpy import asanyarray, ndarray
from torch import renorm
from nn.arch.transformer.TimeSeriesTransformer import TimeSeriesTransformer
from nn.common import *
from datatools import ohlc_resample, pl_binary_labeling, renormalize, unpack
   
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
from nn.ts.classification import LinearBaseline, FCNBaseline, InceptionModel, ResNetBaseline
from nn.arch.transformer.TransformerDataset import generate_square_subsequent_mask


def list_stonks(stonk_dir='./stonks'):
   from pathlib import Path
   return [n[:-8] for n in Path(stonk_dir).rglob('*.feather')]

def load_frame(sym:str):
   df:DataFrame = pd.read_feather('./stonks/%s.feather' % sym)
   print(df.columns)
   return df.set_index('datetime', drop=False)

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

def rescale(x:Tensor, new_min=0.0, new_max=1.0):
   old_range = (x.min(), x.max())
   return old_range, renormalize(x, old_range, (new_min, new_max))

def norm_batches(x: Tensor):
   with torch.no_grad():
      res = Variable(torch.zeros_like(x))
      for i in range(x.size(0)):
         batch = x[i]
         _, scaled_batch = rescale(batch, 0.0, 1.0)
         res[i] = scaled_batch
      return res

df = load_frame('AAPL')[['open', 'high', 'low', 'close']]
df['delta'] = df['close'].diff()

print(df)
data = df.to_numpy()[1:]

num_input_channels = 4
num_predicted_classes = 2 #* (P prob, L prob)
in_seq_len = 7
out_seq_len = 1

hidden_sizes = [
   (in_seq_len * num_input_channels * 2),
   (in_seq_len * num_input_channels),
   (in_seq_len * num_input_channels // 2)
]

win = TwoPaneWindow(in_seq_len, out_seq_len)
win.load(data)

seq_pairs = list(win.iter(lfn=lambda x: x[:, :-1]))
# print(len(seq_pairs), 'sequences loaded')
Xl, yl = [list(_) for _ in unzip(seq_pairs)]
Xnp:ndarray
ynp:ndarray 
Xnp, ynp = np.asanyarray(Xl), np.asanyarray(yl).squeeze(1)
ynp = ynp
ynp = pl_binary_labeling(ynp[:, 4])
Xnp, ynp = Xnp, ynp

X, y = torch.from_numpy(Xnp), torch.from_numpy(ynp)
X, y = X.float(), y.float()
X, y = shuffle_tensors_in_unison(X, y)
# X = norm_batches(X)

randsampling = torch.randint(0, len(X), (2000,))
X = X.swapaxes(1, 2)
X, y = X[randsampling], y[randsampling]

test_split = 0.4
spliti = round(len(X) * test_split)
X_test, y_test = X[-spliti:], y[-spliti:]
X, y = X[:-spliti], y[:-spliti]

X_test, X, y_test, y = X, X_test, y, y_test
print(X.shape, y.shape)
input()

max_epochs = 25

from tqdm import tqdm

def fit(model:Module, X:Tensor, y:Tensor, criterion=None, eval_X=None, eval_y=None, lr=0.001):
   # model = LinearBaseline(in_seq_len, num_pred_classes=num_predicted_classes)
   assert criterion is not None
   opt = torch.optim.Adam(model.parameters(), lr=lr)
   crit = criterion

   fit_iter = tqdm(range(max_epochs))
   metric_logs = OrderedDict()
   
   for e in fit_iter:
      opt.zero_grad()
      
      y_pred = model(X)
      # y_labels = torch.argmax(y, 1)
      # y_pred_labels = torch.argmax(y_pred, 1)
      
      loss = crit(y, y_pred)
      loss.backward()
      opt.step()
      
      fit_iter.set_description(f'epoch {e}')
      
      le = metric_logs[e] = dict(epoch=e, loss=loss.detach().item())
      
      if not (eval_X is None or eval_y is None):
         accuracy = evaluate(model, eval_X, eval_y)
         le['accuracy'] = accuracy
         fit_iter.set_postfix(le)
      
      
   return metric_logs, model

def evaluate(m: Module, X, y):
   y_eval = m(X)
   
   y_labels = torch.argmax(y, 1)
   y_pred_labels = torch.argmax(y_eval, 1)
   accuracy = ((y_labels == y_pred_labels).int().count_nonzero() / len(y_labels))*100.0
   
   # print('accuracy=%f' % accuracy.item())
   return accuracy.item()

core_kw = dict(in_channels=num_input_channels, num_pred_classes=2)
model_cores = [
   ResNetBaseline,
   FCNBaseline,
]
rates = [
   0.001,
   0.0001,
   0.00001
]

model_variants = (
   (
      base_factory,
      rate,
      Sequential(
         Dropout(p=0.15),
         base_factory(**core_kw),
         ReLU()
      )
   )
   
   for (base_factory, rate) in itertools.product(model_cores, rates)
)

experiments = []
for base_ctor, learn_rate, model in model_variants:
   hist, _ = fit(model, X, y, criterion=MSELoss(), eval_X=X_test, eval_y=y_test, lr=learn_rate)
   score = evaluate(model, X_test, y_test)
   hist = pd.DataFrame.from_records(list(hist.values()))
   print('final accuracy score:', score)
   
   experiments.append((score, base_ctor))
   print(hist.sort_values(by='accuracy', ascending=False))
   
summary = pd.DataFrame.from_records(experiments)