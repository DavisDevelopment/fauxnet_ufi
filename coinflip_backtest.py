
import sys, os

from datatools import norm_batches
P = os.path
import torch
import numpy as np
import pandas as pd
from numpy import ndarray
from nn.data.core import TwoPaneWindow

from torch import Tensor, from_numpy, tensor
from tools import unzip

def load_dataframe(symbol:str, dir='./', format='csv', **read_kwargs):
   readfn = getattr(pd, f'read_{format}')
   datapath = P.join(dir, f'{symbol}.{format}')
   if not P.exists(datapath):
      raise FileNotFoundError(datapath)
   
   df:pd.DataFrame = readfn(datapath, **read_kwargs)
   if not isinstance(df.index, pd.DatetimeIndex):
      df['datetime'] = pd.to_datetime(df['datetime'])
      df = df.set_index('datetime', drop=True)
   
   return df

def get_model_inputs(df: pd.DataFrame):
   df = df.copy()
   data:ndarray = df.to_numpy()

   num_input_channels = 4
   num_predicted_classes = 3
   in_seq_len = 7
   out_seq_len = 1
   
   for i in range(1+in_seq_len, len(data)-out_seq_len-1):
      tstamp = df.index[i]
      X = from_numpy(data[i-in_seq_len:i]).unsqueeze(0).float().swapaxes(1, 2)
      yield tstamp, X

def run_backtest(model, df:pd.DataFrame):
   steps = []
   dollars = 50.0
   dollars_init = dollars
   holdings = 0.0
   logs = []
   balances = []
   
   def liq(t):
      nonlocal holdings
      vol = holdings
      if vol == 0: 
         return
      
      holdings = 0
      today = df.loc[t]
      
      nonlocal dollars
      dollars += (vol * today.close)
      
      logs.append(('S', t, vol, today.close))
   
   def buy(t):
      today = df.loc[t]
      nonlocal holdings, dollars
      
      vol = (dollars / today.close)
      holdings += vol
      dollars -= (vol * today.close)
      logs.append(('B', t, vol, today.close))
         
   for time, X in get_model_inputs(df):
      X = norm_batches(X)
      ypred = model(X).argmax()
      today = df.loc[time]
      print(X[-1], ypred)
      
      liq(time)
      balances.append((time, dollars))
      
      if ypred == 2:
         buy(time)
               
      elif ypred == 0:
         continue
      
      elif ypred == 1:
         continue
         
      else:
         raise Exception('Invalid value for ypred; Invalid class label')
      
   liq(time)
   balances.append((time, dollars))
   
   return_on_investment = ((dollars / dollars_init) * 100)
   print(f'roi=${return_on_investment:,.2f}')
   
   return logs, balances


stock = load_dataframe('TSLA', './stonks', format='feather')[['open', 'high', 'low', 'close']]
print(stock)

model = torch.load('./classifier_pretrained')
print(model)

logs, balances = run_backtest(model, stock)
balances = pd.DataFrame.from_records(balances, columns=('datetime', 'balance'))
balances = balances.set_index('datetime', drop=True)
print(balances)