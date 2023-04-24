#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pprint
from typing import Iterable, Tuple
import pandas as pd
import numpy as np
from torch import Tensor, tensor
import torch
from nn.data.sampler import DataFrameSampler
from operator import methodcaller
from functools import partial
from itertools import zip_longest
from fn import F, _
from tools import nor, unzip, nn, isiterable, Struct, safe_div
from typing import *
from faux.backtesting.common import ensure_eq_sized, ensure_equivalence, renormalize

from tree_tests import ensure_eq_sized
from cytoolz.dicttoolz import merge, assoc, dissoc, keyfilter, valmap, valfilter, itemfilter, itemmap
from pandas import DataFrame, Series
from termcolor import colored

from faux.backtesting.backtester import Backtester as Sub

# In[3]:




# In[4]:


# import pandas as pd
from coretools import list_stonks, load_frame
from faux.pgrid import PGrid
from faux.features.ta.loadout import Indicators, IndicatorBag

# winners = pd.read_pickle('minirocket_exp_results.pickle')

# print(winners)
# winners = winners.iloc[0:11]
# setups = []
# for _, winner in winners.iterrows():
#    winning_analyzer = Indicators(*winner.indicators)
#    winning_params   = winner.hyperparams
#    setups.append((winning_params, winning_analyzer))
   


# In[5]:


# symbols = list_stonks('./sp100')[:25]
# frames = [load_frame(sym, './sp100') for sym in symbols]
# for (sym, df) in zip(symbols, frames):
#    df.name = sym
#    print(df)


# In[6]:


import pandas as pd

def samples_from(df:pd.DataFrame, params, transform=None, date_begin=None, date_end=None):
   from faux.backtesting.common import samples_for, split_samples
   if transform is None:
      transform = lambda x: x
   
   symbol = df.name
   ts_idx, X, y = samples_for(
      symbol=df, 
      analyze=transform, 
      xcols_not=['open', 'close', 'datetime'],
      x_timesteps=params['seq_len']
   )
   
   idx:pd.DatetimeIndex = pd.DatetimeIndex([d.date() for d in ts_idx])
   # print(idx)
   X = torch.from_numpy(X)
   y = torch.from_numpy(y)
   
   #split the data 
   if date_begin is not None or date_end is not None:
      begin_index = None
      end_index = None
      
      for i, ts in enumerate(ts_idx):
         if date_begin is not None and begin_index is None and ts >= date_begin:
            begin_index = i
         
         if date_end is not None and end_index is None and ts >= date_end:
            end_index = i
            break
            
      ts_idx = ts_idx[begin_index:end_index]
      X = X[begin_index:end_index]
      y = y[begin_index:end_index]
   
   #* train the model on 85% of the available data, evaluating on the remaining 15%
   # train_X, train_y, test_X, test_y = split_samples(X=X, y=y, pct=val_split, shuffle=False)
   # train_X, test_X = tuple((v.unsqueeze(1) if v.ndim == 2 else v.swapaxes(1, 2)) for v in (train_X, test_X))
   (X,) = tuple((v.unsqueeze(1) if v.ndim == 2 else v.swapaxes(1, 2)) for v in (X,))
   
   return (ts_idx, X, y)

# samples_from(frames[0], setups[0][0], setups[0][1], date_begin=pd.Timestamp('01-01-2010'))


# In[15]:


# params, transform = setups[0]

# samples = [samples_from(df, params, transform, date_begin=pd.Timestamp('01-01-2005')) for df in frames]


# In[35]:


from collections import namedtuple
from tools import maxby, minby, flatten_dict

BTSubSample = namedtuple('BTSubSample', 'step time X ytrue ypred')
BTSample = namedtuple('BTSample', 'symbol step time X ytrue ypred')

from frozendict import frozendict
from bidict import bidict
from collections import UserDict

class BTFrames(UserDict):
   def __init__(self, symbols, frames):
      self.symbols = symbols[:]
      self.sym2idx = bidict({sym:i for i, sym in enumerate(symbols)})
      
      if isinstance(frames, list):
         ensure_eq_sized(symbols, frames)
         frames = zip(range(len(symbols)), frames)
      
      elif isinstance(frames, Mapping):
         frames = [(self.sym2idx[k], df) for k, df in frames.items()]
         
      else:
         raise ValueError('man, shit')
      
      super().__init__(dict(frames))
   
   def __getitem__(self, k:Union[int, str]) -> DataFrame:
      assert (k in self.keys() or k in self.sym2idx.keys()), KeyError(k)
      if isinstance(k, str):
         k = self.sym2idx[k]
      return super().__getitem__(k)

class PolySymBacktester:
   datasets: BTFrames
   
   def __init__(self, symbols=None, datasets=None, samples=None, models=None, init_balance=100.0, pos_type='long', on_sampler=None):
      # self.init_balance = init_balance
      self.pos_type = pos_type
      self.on_sampler = on_sampler
      
      self.wrong = 0
      self.right = 0
      
      self.trade_logs = []
      self.holdings_logs = []
      
      self._has_run = False
      self._batched_samples = False
      
      assert datasets is not None or symbols is not None
      self.symbols = list(symbols) if symbols is not None else ([df.name for df in datasets])
      
      self.datasets = BTFrames(symbols, datasets)
      #...

      
      if isinstance(init_balance, dict):
         init_balance.setdefault('$USD', 100.0)
         self.init_balance = init_balance['$USD']
         self.holdings = dissoc(init_balance, *(set(init_balance.keys()) - set(self.symbols) - {'$USD'}))
      else:
         self.init_balance = init_balance
         self.holdings = {
            '$USD': init_balance
         }
      
      if samples is not None:
         self.samples = samples
         if isinstance(samples, Mapping):
            self.samples = [samples[sym] for sym in self.symbols]
         assert len(self.samples) == len(self.symbols)
      
      assert samples is not None, f'In-house automatic sampling is not yet implemented because I\'m lazy'
      
      if models is None:
         def dfl(x:Tensor):
            return torch.zeros((len(x),))
         print('WARNING: No models specified, loss will be predicted for every sample')
         self.models = [dfl for sym in self.symbols]
      elif callable(models):
         self.models = [models for sym in self.symbols]
      else:
         self.models = models
      
      self.mount_subs(symbols=self.models, datasets=self.datasets, samples=self.samples, models=self.models)
      
   #TODO: write method for computing the 'state-encoding' for the current state
      
   def run(self):
      for sample in self.loop:
         time = sample.time
         signals = sample.signals
         # print(signals)
         assert isiterable(signals), f'{type(signals).__qualname__}'
         # pprint.pprint(sample, depth=2)
         
         #* enumerate signals
         buy_signals = set()
         sell_signals = set()
         sigvec = ''.join([str(label.item()) for label in signals.values()])
         print('SIGNALS=', sigvec)
         
         for symId, signal in signals.items():
            # print(f'SIGNAL(symbol="{symId}", label={signal})')
            if signal == 1:
               buy_signals.add(symId)
               
            elif signal == 0:
               sell_signals.add(symId)
         
         #* liquidate positions in stock which is projected to go downward
         held_pre = self.portfolio
         for symId in sell_signals: 
            #TODO close both LONG and SHORT positions (when they are not projected to be profitable)
            sym = self.symbols[symId]
            if sym in held_pre:
               self.close_long(time, sym)
         
         if len(buy_signals) > 0:
            print(set(map(self.symbols.__getitem__, buy_signals)))
         
         investable_liquidity = min(self.dollars, 250000.0)
         max_exposure_per_symbol:float = safe_div(investable_liquidity, len(buy_signals))
         
         #* open positions in stock which is projected to go upward
         for symId in buy_signals:
            sym = self.symbols[symId]
            
            if self.pos_type in ('long', 'both'):
               self.open_long(time, sym, bp=max_exposure_per_symbol)
               
            else:
               print(f'WARNING: Unknown position_type "{self.pos_type}"')
         
         held_post = self.portfolio
         
         self.holdings_logs.append((time, self.dollars, self.dollars + self.portfolio_value(time), (held_pre, held_post)))
         
      print('DONE')
      summary:DataFrame = DataFrame.from_records(self.holdings_logs, columns=['time', 'dollars', 'balance', 'portfolio']).set_index('time', drop=False)
      
      bal_init = self.init_balance
      sym_init = {sym:self.datasets[symId].close.iloc[0] for symId, sym in enumerate(self.symbols)}
      
      roi = summary['roi'] = (summary.balance / bal_init)
      
      for id, symbol in enumerate(self.symbols):
         summary[symbol] = self.datasets[id].close.loc[summary.index]
      
      cumval = summary['$$IDX'] = summary[self.symbols].sum(axis=1)
      cumval_init = cumval.iloc[0]
      baseline_roi = summary['baseline_roi'] = (cumval / cumval_init)
      vs_market = summary['vs_market'] = (roi / baseline_roi)
      final_vs_market = vs_market.iloc[-1]
      
      summary = summary.drop(columns=self.symbols)
      
      plot = False
      block = True
      showfig = True
      savefig = f'./figures/{",".join(self.symbols)}.png'
      
      if plot:
         import matplotlib.pyplot as plt
         
         pltd:pd.DataFrame = summary[['balance', *self.symbols]]
         pltd.plot(y=['balance', *self.symbols], legend=True)
         
         if showfig:
            plt.show(block=block)
         if savefig is not None:
            plt.savefig(savefig)
      
      return summary
   
   def mount_subs(self, symbols=[], datasets=[], samples=[], models=[]):
      ensure_eq_sized(symbols, datasets, samples, models)
      # from .backtester import Backtester as Sub
      
      subs = self.subs = []
      
      for (symbol, df, smpls, model) in zip(symbols, datasets, samples, models):
         sub = Sub(model=model, df=df, samples=smpls)
         sub.total = 0
         subs.append(sub)
      
      self.loop = ConcurrentBTLoopEnumeration(self)
      
   def sub_accuracy(self, symbol:Union[int, str])->float:
      sub:Sub = self.subs[self.symbols.index(symbol) if isinstance(symbol, str) else symbol]
      if sub.total == 0:
         return 1.0
      else:
         print(sub.total, sub.right, sub.wrong)
         accuracy = safe_div(sub.right, sub.total) * 100.0
         return accuracy
      
   def transact(self, symbol, kind, time, volume, price):
      if isinstance(symbol, int):
         symbol = self.symbols[symbol]
      entry = (symbol, kind, time, volume, price)
      print(*entry)
      self.trade_logs.append(entry)
      
   def close_long(self, t, symbol):
      if isinstance(symbol, int):
         symbol = self.symbols[symbol]
      vol = self.holdings[symbol]
      if vol == 0:
         return 
      today_close = self.datasets[symbol]['close'].loc[t]
      self.dollars += (vol * today_close)
      self.holdings[symbol] -= vol
      
      self.transact(symbol, 'S', t, vol, today_close)
      
   def open_long(self, t, symbol, w=None, bp=None):
      """
      Buy as many shares of the given stock as possible using the available
      dollars, at the current market price at time t.

      Args:
         t (datetime): The time at which to open the long position.
         stock (str): The ticker symbol of the stock to buy.
         w (float): The proportion of buying power

      Returns:
         None.

      Raises:
         KeyError: If the given ticker symbol is not found in the dataset.
      """
      if isinstance(symbol, int):
         symbol = self.symbols[symbol]
      
      symMdlAccuracy:float = self.sub_accuracy(symbol)
      
      today_close = self.datasets[symbol]['close'].loc[t]
      bp:float = bp if bp is not None else self.dollars
      avail = round(self.dollars, 6)
      bp = min(avail, round(bp, 6))
      
      w:float = w if w is not None else (renormalize(symMdlAccuracy, 50.0, 100.0, 0.0, 1.0))
      
      assert avail >= bp, f'Insufficient funds. Specified buying power exceeds available balance (${bp} > {avail})'
      
      if bp == 0:
         print('man, you broke')
         return
      elif w == 0:
         print(colored('you have no idea what you are doing', 'light_red', attrs=['bold']))
         return

      if symbol not in self.holdings:
         self.holdings[symbol] = 0

      #TODO factor trading fees into the transaction
      if self.holdings[symbol] >= 0:
         print(f'{symbol} accuracy: {symMdlAccuracy:.2f}%')
         #* Open a new long position
         vol:float = (bp / today_close) #? no. of shares that can be purchased with specified buying power
         vol = (vol * w) #? scaled proportionally with the confidence (for now, simply accuracy) level of the model which suggested the position
         
         self.holdings[symbol] += vol #? add these shares to our portfolio
         self.dollars -= (vol * today_close) #? subtract the cost of those shares from our pool of dollars
         self.transact(symbol, 'B', t, vol, today_close)
      
      else:
         # Close the existing short position and open a new long position
         vol = -self.holdings[symbol]
         self.dollars += (vol * today_close)
         self.holdings[symbol] = vol
         self.transact(symbol, 'B', t, vol * 2, today_close)

   def open_short(self, t, symbol):
      """
      Sell as many shares of the given stock as possible, shorting the stock if
      necessary, at the current market price at time t.

      Args:
         t (datetime): The time at which to open the short position.
         stock (str): The ticker symbol of the stock to sell short.

      Returns:
         None.

      Raises:
         KeyError: If the given ticker symbol is not found in the dataset.
      """
      if isinstance(symbol, int):
         symbol = self.symbols[symbol]
      today_close = self.datasets[symbol]['close'].loc[t]

      if self.dollars == 0:
         return

      if symbol not in self.holdings:
         self.holdings[symbol] = 0

      if self.holdings[symbol] <= 0:
         # Open a new short position
         vol = (self.dollars / today_close)
         self.holdings[symbol] -= vol
         self.dollars -= (vol * today_close)
         self.transact(symbol, 'S', t, vol, today_close)
      else:
         # Close the existing long position and open a new short position
         vol = self.holdings[symbol]
         self.dollars += (vol * today_close)
         self.holdings[symbol] = -vol
         self.transact(symbol, 'S', t, vol, today_close)

   @property
   def dollars(self):
      return self.holdings['$USD']
   @dollars.setter
   def dollars(self, val):
      self.holdings['$USD'] = float(val)
      
   @property
   def portfolio(self)->Dict[str, float]:
      #TODO: represent portfolio more robustly than this
      return frozendict(valfilter(_ != 0, dissoc(self.holdings, '$USD')))
   
   def portfolio_value(self, t)->float:
      held = self.portfolio
      return sum(held[sym] * self.datasets[self.symbols.index(sym)]['close'].loc[t] for sym in held.keys())
      
class ConcurrentBTLoopEnumeration:
   def __init__(self, owner:PolySymBacktester):
      self.owner = owner
      self.testers = owner.subs
      self.loops = [itertools.starmap(BTSubSample, t.loop()) for t in self.testers]
      self.terminated = {}
      self.lastYields = {}
      
   def reshape_loops(self):
      while len(self.testers) > len(self.terminated):
         values = []
         
         for symbolId, loop in enumerate(self.loops):
            if symbolId not in self.terminated:
               
               try:
                  (loop_step, l_time, l_x, l_yt, l_yp) = next(loop)
                  
                  values.append(BTSample(symbolId, loop_step, l_time, l_x, l_yt, l_yp))
                  
               except StopIteration:
                  self.terminated[symbolId] = True
                  values.append(None)
            else:
               values.append(None)
         if len(self.testers) > len(self.terminated):
            yield values
         
   def synchronize_loops(self):
      lastsample:Dict[int, BTSubSample] = {}
      start_date = None
      for symId, loop in enumerate(self.loops):
         sample = lastsample[symId] = next(loop)
         if start_date is None or sample.time > start_date:
            start_date = sample.time
      print(f'Synchronizing loops to {start_date}')
      
      for symId, sample in lastsample.items():
         while sample.time < start_date:
            sample = lastsample[symId] = next(self.loops[symId])
         self.loops[symId] = push(sample, self.loops[symId])
   
   def __iter__(self):
      self.synchronize_loops()
      
      
      for raw in self.reshape_loops():
         # if len(raw) == 0:
         time = sample_time(raw)
         signals = sample_signals(raw)
         
         yield Struct(
            time=time,
            signals=signals,
            samples=raw
         )
   

def push(a, b):
   yield a
   yield from b
   
from fn import _, F

def sample_signals(loop_sample):
   return frozendict((symId, e.ypred) for symId, e in enumerate(loop_sample))

def sample_time(sample):
   return ensure_equivalence([e.time for e in sample if e is not None])
   
import itertools
from tools import isiterable

def joinits(a, b):
   if not isiterable(a):
      a = [a]
   if not isiterable(b):
      b = [b]
   return itertools.chain(a, b)

def asdatetime(x):
   if x is None:
      return x
   return pd.to_datetime(x)

def backtest(symbols:List[str], params, transforms, model=None, dir='./stonks', date_begin=None, date_end=None):
   if isinstance(symbols[0], DataFrame):
      frames:List[DataFrame] = [cast(DataFrame, df) for df in symbols]
      symbols = [df.name for df in frames]
   else:
      frames:List[DataFrame] = [load_frame(symbol, dir) for symbol in symbols]
      
   date_begin, date_end = (asdatetime(date_begin), asdatetime(date_end))
   transforms = [(f[0] if isinstance(f, list) else f) for f in transforms]
   samples = [samples_from(df, p, transform, date_begin=date_begin, date_end=date_end) for (df, transform, p) in zip(frames, transforms, params)]
   
   tester = PolySymBacktester(symbols, datasets=frames, samples=samples, models=model)
   summary = tester.run()
   
   return tester, summary


# In[ ]:




