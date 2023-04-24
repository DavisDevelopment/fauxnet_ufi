from typing import Iterable, Tuple
import pandas as pd
import numpy as np
from torch import Tensor, tensor
import torch
from nn.data.sampler import DataFrameSampler
from operator import methodcaller
from functools import partial
from itertools import zip_longest
from fn import F
from tools import nor, unzip, nn, isiterable
from typing import *

class Backtester:
   def __init__(self, model, df: pd.DataFrame = None, samples = [], init_balance: float = 100.0, pos_type: str = 'long', on_sampler=None):
      self.model = model
      self.df = df
      self.samples = samples
      self.init_balance = init_balance
      self.pos_type = pos_type
      self.on_sampler = on_sampler
      
      self.borrow_price = None
      self.holdings = 0.0
      self.logs = []
      self.balances = []
      self.dollars = self.init_balance
      self.dollars_init = self.dollars
      self.last_time = None
      self._has_run = False
      self._batched_samples = False
      
      if (samples is None or len(samples) == 0) and self.df is not None:
         self.sampler = DataFrameSampler(self.df)
         if callable(self.on_sampler):
               self.on_sampler(self.sampler)
         self.samples = list(self.sampler.samples())
         
      elif samples is not None:
         if isinstance(samples, tuple) and len(samples) == 3:
            self.samples = samples
            self._batched_samples = True
         elif isiterable(samples):
            self.samples = list(self.samples)
      else:
         mirror = lambda s: (s + s[::-1])
         raise ValueError('You must be ' + mirror('po') + 'ing me')
      
      self.backtest_on = self.samples
      self.wrong = 0
      self.right = 0
            
   def numpify(self, xxx):
      return np.asanyarray(list(map(lambda v: (v.numpy() if isinstance(v, Tensor) else v), xxx)), dtype='float64')
   
   def transact(self, kind, ts, volume, price):
      entry = (kind, ts, volume, price)
      # print(entry, f'${self.dollars + (self.holdings * price):,.2f}')
      self.logs.append(entry)
      
   def close_long(self, t):
      vol = self.holdings
      if vol == 0:
         return
      
      # self.holdings = 0
      
      today_close = self.df.close.loc[t]
      self.dollars += (vol * today_close)
      self.holdings -= vol
      
      self.transact('S', t, vol, today_close)

   def open_long(self, t):
      #TODO refactor to support multiple stocks
      today_close = self.df.close.loc[t]
      
      if self.holdings > 0 or self.dollars == 0:
         return
      
      vol = (self.dollars / today_close)
      self.holdings += vol
      self.dollars -= (vol * today_close)
      self.transact('B', t, vol, today_close)

   def open_short(self, t):
      today = self.df.loc[t]
      self.borrow_price = today.close
      vol = (self.dollars / self.borrow_price)
      self.holdings = -vol

   def close_short(self, t):
      today = self.df.loc[t]
      buy_cost = self.borrow_price * (-self.holdings)
      sell_cost = (-self.holdings) * today.close
      profit = (buy_cost - sell_cost)
      self.dollars += (buy_cost - sell_cost)
      self.borrow_price = None
      self.holdings = 0

   def each_sample(self, inc_y=True)->Tuple[pd.Timestamp, Tensor, Tensor]:
      """
      -
       Generator for handling iteration over given backtesting samples

       Parameters
       ----------
       inc_x : bool, optional
           _description_, by default True
       inc_y : bool, optional
           _description_, by default True

       Yields
       ------
       Tuple[Timestamp, Tensor] | Tuple[Timestamp, Tensor, Tensor]
          a tuple of (
             "time" - the Timestamp index, 
             "features | X" - the input features, if any
             "target | label" - the output target (regression target, or classification label), if any
          ) 
       e.g. `(time, X, y)`
      """
      n_sample_items = None #? `None | 2 | 3`
      #* initially, whether the samples include the target labelling is unknown
      y_na = tensor(-1, dtype=torch.long)
      
      for sample_entry in self.backtest_on:
         if n_sample_items is None:
            assert isinstance(sample_entry, (list, tuple)), f'each sample (as passed to "Backtest(..., samples=?)") must be tuple, got {type(sample_entry).__name__} instead'
            n_sample_items = len(sample_entry)
            assert n_sample_items in (2, 3), f'Expected [2|3]-tuple, got a {n_sample_items}-tuple instead'
         
         time:pd.Timestamp
         X:Tensor
         y:Optional[Tensor] = None
         
         if n_sample_items == 2:
            time, X = sample_entry
            if inc_y:
               yield time, X, y_na
            else:
               yield time, X
            
         elif n_sample_items == 3:
            time, X, y = sample_entry
            if inc_y:
               yield time, X, y
            else:
               yield time, X


   def loop(self):
      ypred = self.generate_signals()
      self.total = self.right = self.wrong = 0
      
      if self._batched_samples:
         bT, bX, by = self.samples
         
         for i in range(min(len(bT), len(bX), len(by))-1):
            t, x, yt = bT[i], bX[i], by[i]
            yp = ypred[i]
            
            if yt == yp:
               self.right += 1
            else:
               self.wrong += 1
            self.total += 1
            
            yield i, t, x, yt, yp
      else:
         for i, (t, x, yt) in enumerate(self.each_sample()):
            yp = ypred[i]
            
            if yt == yp:
               self.right += 1
            else:
               self.wrong += 1
            self.total += 1
            
            yield i, t, x, yt, yp

   def generate_signals(self):
      """
      generates the signals for each input sample
      """
      if self._batched_samples:
         times, bX, by = self.samples
         # print(tuple(map(lambda x: (type(x).__qualname__, len(x)), (times, bX, by))))
         ypred = self.model(bX).detach().long()
         return ypred
      
      else:
         ypred = tensor([self.model(x).detach().long()[0] for t,x in self.each_sample(inc_y=False)])
         return ypred
      

   def run(self):
      self.total = 0
      self.right = 0
      self.wrong = 0
      
      for i, time, X, ytrue, ypred in self.loop():
         self.total += 1
         if (ytrue if isinstance(ytrue, int) else ytrue.item()) == ypred:
            self.right += 1
         else:
            self.wrong += 1

         if self.holdings > 0:
            if ypred == 0:
               self.close_long(time)
               
         if ypred == 0: #* LOSS
            if self.pos_type in ('short', 'both'):
               self.open_short(time)
         elif ypred == 1: #* GAIN
            if self.pos_type in ('long', 'both'):
               self.open_long(time)

         self.balances.append((time, self.dollars, self.holdings))

      if self.holdings > 0: #* stock held; position is still OPEN
         self.close_long(self.df.index[-1])
         self.balances.append((self.df.index[-1], self.dollars, self.holdings))

      if self.holdings < 0:
         self.close_short(self.df.index[-1])
         self.balances.append((self.df.index[-1], self.dollars, self.holdings))

      # df = pd.DataFrame(self.balances, columns=['time', 'balance']).set_index('time')
      # df['returns'] = df.balance.pct_change().fillna(0.0)
      self._has_run = True
      return self.logs, self.balances
   
   def summarize(self, plot=True, savefig=None, showfig=True, block=True):
      import matplotlib.pyplot as plt
      
      df:pd.DataFrame = self.df
      summary:pd.DataFrame = pd.DataFrame.from_records(self.balances, columns=('datetime', 'balance', 'held')).set_index('datetime', drop=False)
      close = df.close.loc[summary.index]
      summary['close'] = close
      summary['delta'] = close.pct_change()
      summary['USD'] = summary['balance']
      summary.loc[summary.USD == 0, 'balance'] = (summary.close * summary.held)
      
      bal_init = self.init_balance
      close_init = close.iloc[0]
      # close_final = close.iloc[-1]
      summary['baseline_roi'] = (close / close_init)
      summary['roi'] = (summary.balance / bal_init)
      # print(summary)
      
      if plot:
         pltd:pd.DataFrame = summary[['close', 'balance']]
         pltd.plot(y=['close', 'balance'])
         
         if showfig:
            plt.show(block=block)
            
         if savefig is not None:
            plt.savefig(savefig)
         
      return summary