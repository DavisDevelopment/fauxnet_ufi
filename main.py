
import itertools
from functools import reduce
import pickle
from time import sleep
from pprint import pprint
from numpy import asanyarray, ndarray
from torch import BoolTensor, random, renorm
from torch.jit._script import ScriptModule, ScriptFunction
from coinflip_backtest import count_classes
from nn.arch.transformer.TimeSeriesTransformer import TimeSeriesTransformer
from nn.common import *
from datatools import P, ohlc_resample, pl_binary_labeling, renormalize, unpack
   
from nn.namlp import NacMlp
from cytoolz import *
from cytoolz import itertoolz as iters
import pandas as pd

from datatools import get_cache, calc_Xy, load_ts_dump, rescale, norm_batches
from nn.data.core import TwoPaneWindow, TensorBuffer
from pandas import DataFrame
from typing import *
from nn.ts.classification.fcn_baseline import FCNNaccBaseline
# from nn.arch.transformer.TransformerDataset import TransformerDataset, generate_square_subsequent_mask
from tools import Struct, gets, maxby, unzip, safe_div, argminby, argmaxby
from nn.arch.lstm_vae import *
import torch.nn.functional as F
from nn.ts.classification import LinearBaseline, FCNBaseline, InceptionModel, ResNetBaseline
from nn.arch.transformer.TransformerDataset import generate_square_subsequent_mask

def list_stonks(stonk_dir='./stonks'):
   from pathlib import Path
   return [str(n)[:-8] for n in Path(stonk_dir).rglob('*.feather')]

def load_frame(sym:str):
   df:DataFrame = pd.read_feather('./stonks/%s.feather' % sym)
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


   
symbols = list_stonks()

from tqdm import tqdm
from nn.optim import ScheduledOptim, Checkpoints

max_epochs = 100

def fit(model:Module, X:Tensor, y:Tensor, criterion=None, eval_X=None, eval_y=None, lr=0.001, epochs=None):
   # model = LinearBaseline(in_seq_len, num_pred_classes=num_predicted_classes)
   assert criterion is not None
   epochs = epochs if epochs is not None else max_epochs
   
   inner_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
   optimizer = Checkpoints(
      model,
      inner_optimizer,

      # ScheduledOptim(
      #    inner_optimizer, 
      #    lr_init=lr, 
      #    n_warmup_steps=20
      # )
   )
   optimizer.n_warmup_steps = 2
   crit = criterion

   fit_iter = tqdm(range(epochs))
   metric_logs = OrderedDict()
   
   for e in fit_iter:
      optimizer.zero_grad()
      
      y_pred = model(X)
      
      loss = crit(y, y_pred.squeeze())
      loss.backward()
      
      fit_iter.set_description(f'epoch {e}')
      
      le = metric_logs[e] = Struct(epoch=e, loss=loss.detach().item(), accuracy=None)
      
      if not (eval_X is None or eval_y is None):
         eval_metrics = evaluate(model, eval_X, eval_y)
         le.metrics = tuple(f'{n:.2f}' for n in (eval_metrics.p, eval_metrics.l, eval_metrics.false_positives)) #type: ignore
         le.n_pos_ids_P = eval_metrics.p #type: ignore
         le.n_pos_ids_L = eval_metrics.l #type: ignore
         le.total_neg_ids = eval_metrics.false_positives #type: ignore
         le.accuracy = eval_metrics.score #type: ignore
         le = le.asdict()
         fit_iter.set_postfix(le)

      optimizer.step(le) #type: ignore
      
   history, model = optimizer.close() #type: ignore
      
   return history, model

def snapto(x: Tensor):
   x = x.clone()
   # print(1, x[:10], x[:-10])
   
   x[x < 0.45] = -1
   # print(2, x[:10], x[:-10])
   x[x > 0.6] = 1
   # print(3, x[:10], x[:-10])
   x[x > 0.45] = 0
   # print(4, x[:10], x[:-10])
   return x

def evaluate(m: Module, X:Tensor, y:Tensor):
   y_eval = m(X)
   assert (y.shape == y_eval.squeeze().shape)
   
   count = lambda *bool_exprs: reduce(torch.logical_and, bool_exprs).int().count_nonzero().item()
   select = lambda array, *bool_exprs: array[reduce(torch.logical_and, bool_exprs)]
   
   labels = y.argmax(dim=1)
   pred_labels = y_eval.argmax(dim=1)
   q_p = (labels == 1)
   q_l = (labels == 0)
   matched:BoolTensor = (pred_labels == labels)
   
   n_p, n_l = count(q_p), count(q_l)
   n_matched, n_correct_p, n_correct_l = (
      count(matched),
      count(matched, q_p),
      count(matched, q_l)
   )
   
   n_false_p, n_false_l = (
      count(torch.logical_not(matched), pred_labels == 1),
      count(torch.logical_not(matched), pred_labels == 0)
   )
   
   # select(labels, matched.logical_not(), labels == )
   misidentified = count(pred_labels == 1, labels == 0) + count(labels == 1, pred_labels == 0)
   
   misids = safe_div(misidentified, n_p+n_l)

   accuracy = safe_div(n_correct_p + n_correct_l, n_p + n_l) * 100
   
   pct_pos_p = safe_div(n_correct_p, n_p)
   pct_pos_l = safe_div(n_correct_l, n_l)
   
   accuracy = (pct_pos_p + pct_pos_l) - misids
   
   return Struct(
      score=accuracy,
      p=safe_div(n_correct_p, n_p),
      l=safe_div(n_correct_l, n_l),
      false_positives=misids
   )
   
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
   in_seq_len:Optional[int] = None
   out_seq_len:Optional[int] = 1
   num_input_channels:Optional[int] = 4
   input_channels:Optional[tuple] = ('open', 'high', 'low', 'close')
   num_predicted_classes:Optional[int] = 2
   
   symbol:Optional[str] = None
   df:Optional[DataFrame] = None
   
   shuffle:bool = True
   val_split:float = 0.12
   epochs:int = 30
   
   def __post_init__(self):
      assert not (self.df is None and self.symbol is None), "either symbol name or a DataFrame must be provided"
      assert self.in_seq_len is not None, "size of input sequence must be provided"
      
      if self.symbol is not None and self.df is None:
         self.df = load_frame(self.symbol)
         if self.input_channels is not None:
            self.df = self.df[list(self.input_channels)]

def shotgun_strategy(config:ExperimentConfig):
   # df = load_frame('MSFT')[['open', 'high', 'low', 'close']]
   df:DataFrame = config.df
   df['delta_pct'] = df['close'].pct_change()
   data = df.to_numpy()[1:]

   num_input_channels, in_seq_len, out_seq_len = config.num_input_channels, config.in_seq_len, config.out_seq_len
   assert None not in (in_seq_len, out_seq_len)
   win = TwoPaneWindow(in_seq_len, out_seq_len) #type: ignore
   win.load(data)

   from sklearn.preprocessing import MinMaxScaler

   scaler = MinMaxScaler()
   scaler.fit(data[:, :-1])

   seq_pairs = list(win.iter(lfn=lambda x: scaler.transform(x[:, :-1])))
   Xl, yl = [list(_) for _ in unzip(seq_pairs)]
   Xnp:ndarray
   ynp:ndarray 
   Xnp, ynp = np.asanyarray(Xl), np.asanyarray(yl).squeeze(1)

   print(Xnp.shape, ynp.shape)
   ynp = ynp
   ynp:ndarray = pl_binary_labeling(ynp[:, 3])
   print(ynp)
   Xnp, ynp = Xnp, ynp

   X, y = torch.from_numpy(Xnp), torch.from_numpy(ynp)
   X, y = X.float(), y.float()
   X, y = shuffle_tensors_in_unison(X, y)


   randsampling = torch.randint(0, len(X), (2000,))
   X = X.swapaxes(1, 2)
   X, y = X[randsampling], y[randsampling]

   test_split = config.val_split
   spliti = round(len(X) * test_split)
   X_test, y_test = X[-spliti:], y[-spliti:]
   X, y = X[:-spliti], y[:-spliti]

   X_test, X, y_test, y = X, X_test, y, y_test
   print(X.shape, y.shape)

   core_kw = dict(in_channels=num_input_channels, num_pred_classes=2)

   model_cores = [
      ResNetBaseline,
      FCNBaseline,
      # FCNNaccBaseline,
   ]

   rates = [
      0.00005,
      0.00015,
      0.0001,
   ]

   losses = [
      BCEWithLogitsLoss
   ]

   import random
   combos = list(itertools.product(model_cores, rates, losses))

   model_variants = (
      (
         base_factory,
         rate,
         crit_factory(),
         Sequential(
            base_factory(**core_kw)
         )
      )
      
      for (base_factory, rate, crit_factory) in combos
   )

   experiments:List[Dict[str, Any]] = []

   for mventry in model_variants:
      base_ctor, learn_rate, criterion, model = mventry
      model:Sequential = model
      hist, _ = fit(model, X, y, criterion=criterion, eval_X=X_test, eval_y=y_test, lr=learn_rate, epochs=config.epochs)
      metrics:Struct = evaluate(model, X_test, y_test)
      hist:DataFrame

      print('\n'.join([
         f'baseline="{base_ctor.__qualname__}"', 
         f'learn_rate={learn_rate}', 
         f'loss_fn={criterion}'
         # f'activation={activfn}'
      ]))
      print('final accuracy score:', metrics.score)#type: ignore
      
      
      
      print(hist.sort_values(by='accuracy', ascending=False))
      from coinflip_backtest import backtest
      
      trading_perf = backtest(stock=config.symbol, model=model)
      
      experiments.append(dict(
         model=model,
         model_state=model.state_dict(),
         mdl_config=Struct(loss_type=criterion, model_type=base_ctor.__qualname__),
         exp_config=config,
         score=trading_perf.roi,
         logs=hist
      ))

   best_loop = maxby(experiments, key=lambda e: e['score'])
   score, model, mdl_config = gets(best_loop, 'score', 'model', 'mdl_config')


   #* save the best-qualified classifier
   torch.save(model.state_dict(), './classifier_pretrained_state')
   torch.save(model, './classifier_pretrained.pt')
   accuracy = evaluate(model, X_test, y_test)

   pprint(config)

   print('accuracy of final exported model is ', accuracy)
   return best_loop

def generate_figures():
   import matplotlib.pyplot as plt
   from coinflip_backtest import backtest
   
   winners = pickle.load(open('./experiment_winners.pickle', 'rb'))
   for symbol, best_loop in winners.items():
      print(best_loop.keys())
      model:Module = best_loop['model']
      model.load_state_dict(best_loop['model_state'])
      
      trade_sess = backtest(stock=symbol, model=model)
      
      pprint(dict(
         symbol=symbol,
         pl_ratio=trade_sess.pl_ratio,
         roi=trade_sess.roi
      ))
      
      logs:pd.Series = trade_sess.trade_logs #type: ignore
      logs['datetime'] = logs.index
      
      logs.plot(
         title=f'CoinFlipNet({symbol}) Trading Balance', 
         x='datetime',
         y=['close', 'balance'],
         figsize=(24, 16), 
         legend=True, 
         stacked=True
      )
      
      plt.savefig(f'./figures/{symbol}_coin_flip_net.png')
      
      plt.close('all')

if __name__ == '__main__':
   import sys 
   argv = sys.argv[1:]
   
   if 'figures' in argv:
      generate_figures()
      exit()
   
   symbols = [
      # 'MSFT', 'GOOG', 'GOOGL', 'TSLA', 'AAPL', 'SONY', 'INTC', 
      'NVDA', 'AMD',
      # 'NTDOY', 'HPE'     
   ]
   
   winners = {}
   for symbol in symbols:
      try:
         winners[symbol] = shotgun_strategy(ExperimentConfig(
            in_seq_len=14,
            symbol=symbol
         ))
      
      except Exception as error:
         pprint(error)
         continue
   
   pickle.dump(winners, open('./experiment_winners.pickle', 'wb+'))
   
   for sym, best_row in winners.items():
      print(f'Highest-Achieved ROI for [{sym.upper()}]  =  {best_row["score"]:,.2f}%')
   
   generate_figures()