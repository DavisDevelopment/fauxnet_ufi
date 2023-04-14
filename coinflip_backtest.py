from fn import F, _
from pprint import pprint
import sys, os

from termcolor import colored

from datatools import norm_batches, percent_change, quantize, rescale
from nn.ts.classification.fcn_baseline import FCNNaccBaseline, FCNBaseline
P = os.path
import torch
import numpy as np
import pandas as pd
from numpy import ndarray
from nn.data.core import TwoPaneWindow

from torch import Tensor, from_numpy, tensor
from tools import Struct, unzip
from operator import methodcaller

from sklearn.preprocessing import MinMaxScaler
from typing import *

def load_dataframe(symbol:str, dir='./', format='csv', **read_kwargs):
   dir = './stonks'
   readfn = getattr(pd, f'read_{format}')
   datapath = P.join(dir, f'{symbol}.{format}')
   if not P.exists(datapath):
      raise FileNotFoundError(datapath)
   
   df:pd.DataFrame = readfn(datapath, **read_kwargs)
   if not isinstance(df.index, pd.DatetimeIndex):
      df['datetime'] = pd.to_datetime(df['datetime'])
      df = df.set_index('datetime', drop=True)
   
   return df
      
def count_classes(y: ndarray):
   ydelta = percent_change(y[:, 3])
   thresh = 0.5
   
   E = np.argwhere((ydelta <= thresh)&(ydelta >= -thresh))
   P = np.argwhere(ydelta > 2)
   L = np.argwhere(ydelta < -2)
   
   return len(P), len(L)

pos_types = ('long', 'short', 'both')

# def run_backtest(model, df:pd.DataFrame, init_balance:float=100.0, pos_type='long', on_sampler=None):
def run_backtest(model, df:pd.DataFrame=None, samples:Iterable[Tuple[pd.Timestamp, Tensor, Tensor]]=[], init_balance:float=100.0, pos_type='long', on_sampler=None):
   steps = []
   dollars = init_balance
   dollars_init = dollars
   
   borrow_price = None
   holdings = 0.0
   logs = []
   balances = []
   
   def transact(kind, ts, volume, price):
      entry = (kind, ts, volume, price)
      # nonlocal doll
      print(entry, f'${dollars + (holdings * price):,.2f}')
      logs.append(entry)
   
   def close_long(t):
      nonlocal holdings
      vol = holdings
      if vol == 0:
         return
      
      holdings = 0
      today = df.loc[t]
      
      nonlocal dollars
      dollars += (vol * today.close)
      
      transact('S', t, vol, today.close)
   
   def open_long(t):
      today = df.loc[t]
      nonlocal holdings, dollars
      if holdings > 0:
         #! really, this shouldn't be happening at all
         pass
      vol = (dollars / today.close)
      # print(f'vol = ({dollars} / {today.close})')
      holdings += vol
      dollars -= (vol * today.close)
      
      transact('B', t, vol, today.close)
      
   def open_short(t):
      today = df.loc[t]
      nonlocal holdings, dollars, borrow_price
      
      borrow_price = today.close
      vol = (dollars / borrow_price)
      holdings = -vol
      
   def close_short(t):
      today = df.loc[t]
      nonlocal holdings, dollars, borrow_price
      
      buy_cost = borrow_price * (-holdings)
      sell_cost = (-holdings) * today.close
      profit = (buy_cost - sell_cost)
      # print(f'closing SHORT position for ${profit:,.2f}')
      
      dollars += (buy_cost - sell_cost)
      borrow_price = None
      holdings = 0
      
   samples = list(samples)
   
   if len(samples) == 0:
      from nn.data.sampler import DataFrameSampler
      
      sampler = DataFrameSampler(df)
      if callable(on_sampler):
         on_sampler(sampler)
      samples = list(sampler.samples())
   else:
      pass   
         
   last_time = None
   backtest_on = samples[:]
   print(f'running backtest on {len(backtest_on)} samples')
   wrong = 0
   right = 0
   
   bts, bX, by = tuple(map(F(list), unzip(backtest_on)))
   # bX, by = tuple(map(F(methodcaller('numpy')) >> np.asanyarray, (bX, by)))
   numpify = lambda xxx: np.asanyarray(list(map(lambda v: (v.numpy() if isinstance(v, Tensor) else v), xxx)), dtype='float64')
   bX, by = numpify(bX), numpify(by)
   # print(bX)
   
   for time, X, y in backtest_on:
      ypred = model(X).detach().long()[0].item()
      # print(y.item(), ypred)
      if (y if isinstance(y, int) else y.item()) == ypred:
         right += 1
      else:
         wrong += 1
      
      if holdings > 0:
         #TODO: only close when profit is not predicted
         if ypred == 0:
            close_long(time)
      
      elif holdings < 0:
         #TODO: only close when loss is not predicted
         if ypred == 1:
            close_short(time)
      
      balances.append((time, dollars))
      
      if ypred == 0: #* Loss, Downward Movement 
         if pos_type in ('short', 'both'):
            open_short(time)
                        
      elif ypred == 1: #* Gain, Upward Movement
         if pos_type in ('long', 'both') and holdings == 0:
            open_long(time)
      
      else:
         raise Exception('Invalid value for ypred; Invalid class label')
      
      last_time = time
      
   if holdings > 0:
      close_long(last_time)
      
   elif holdings < 0:
      close_short(last_time)
   
   balances.append((last_time, dollars))
   
   return_on_investment = ((dollars / dollars_init) * 100)
   
   print('\n'.join([
      f'roi = {return_on_investment:,.2f}%',
      f'acc = {right / (right + wrong) * 100:,.2f}%'
   ]))
   
   return logs, balances

# if __name__ == '__main__':
# def backtest(stock:Union[str, pd.DataFrame]='AAPL', model=None, pos_type='long', on_sampler=None):
def backtest(stock:Union[str, pd.DataFrame]='AAPL', model=None, pos_type='long', on_sampler=None, samples=[], **kwargs):
   ticker:str = 'AAPL'
   from faux.backtesting.backtester import Backtester
   
   if isinstance(stock, str):
      ticker = stock
      stock:pd.DataFrame = load_dataframe(str(stock), './sp100', format='feather')[['open', 'high', 'low', 'close', 'volume']]
   else:
      pass

   
   if model is None:
      model_state = torch.load('./classifier_pretrained_state')
      model = torch.load('./classifier_pretrained.pt')
      model.load_state_dict(model_state)
      # print(model)
   
   bal_init = stock.close.iloc[0]
   tester = Backtester(model, df=stock, samples=samples, init_balance=bal_init, pos_type=pos_type)
   logs, balances = tester.run()
   blogs = tester.summarize(plot=True, showfig=False)
   
   baseline_roi = blogs.baseline_roi.iloc[-1]
   bals:pd.Series = blogs.balance
   bal_final = bals.iloc[-1]
   rois = bals.diff().iloc[1:]
   
   P = rois[rois > 0].mean()
   L = rois[rois < 0].abs().mean()
   pl_ratio = (P / L)
   final_roi = (bal_final / bal_init)
   
   vs_market = float(round(final_roi/baseline_roi, 3))
   
   print('Trading vs Holding' + colored(f'({ticker})', 'cyan') + ': ', vs_market)
   
   score = (final_roi - baseline_roi) / final_roi * 100.0
   
   did_place_orders = (len(logs) > 1)
   did_beat_market = (vs_market > 1)
   won = (did_beat_market and did_place_orders)
   
   print(
      colored('is winner: ', 'green', 'on_blue', attrs=['bold', 'dark', 'underline']), 
      colored(str(won), 'green' if won else 'red', attrs=['bold'])
   )
   
   return Struct(
      pl_ratio=pl_ratio, 
      roi=final_roi,
      baseline_roi=baseline_roi,
      trade_logs=blogs,
      score=score,
      did_beat_market=did_beat_market,
      did_place_orders=did_place_orders,
   )