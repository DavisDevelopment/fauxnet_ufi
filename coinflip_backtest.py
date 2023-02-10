
from pprint import pprint
import sys, os

from datatools import norm_batches, percent_change, rescale
from nn.ts.classification.fcn_baseline import FCNNaccBaseline, FCNBaseline
P = os.path
import torch
import numpy as np
import pandas as pd
from numpy import ndarray
from nn.data.core import TwoPaneWindow

from torch import Tensor, from_numpy, tensor
from tools import Struct, unzip

from sklearn.preprocessing import MinMaxScaler
from typing import *

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

def get_model_inputs(df:pd.DataFrame, state_dict:Optional[Dict[str, Any]]=None):
   df = df.copy()
   data:ndarray = df.to_numpy()

   scaler = MinMaxScaler()
   scaler.fit(data)
   
   p, l = count_classes(data)
   print(f'Loaded dataset contains {p} P-samples, and {l} L-samples')

   #*TODO define ModelConfig class to pass around instead of duplicating these variables endless all around town
   in_seq_len = 7
   out_seq_len = 1
   
   for i in range(1+in_seq_len, len(data)-out_seq_len-1):
      tstamp = df.index[i]
      
      X = from_numpy(scaler.transform(data[i-in_seq_len:i])).unsqueeze(0).float().swapaxes(1, 2)
      
      yield tstamp, X
      
def count_classes(y: ndarray):
   ydelta = percent_change(y[:, 3])
   thresh = 0.5
   
   E = np.argwhere((ydelta <= thresh)&(ydelta >= -thresh))
   P = np.argwhere(ydelta > 2)
   L = np.argwhere(ydelta < -2)
   
   return len(P), len(L)

def run_backtest(model, df:pd.DataFrame, init_balance:float=100.0):
   steps = []
   dollars = init_balance
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
      ypred = model(X).argmax()
      
      liq(time)
      balances.append((time, dollars))
      
      if ypred != 0:
         pass
      
      if ypred == 1:
         buy(time)
               
      elif ypred == 0:
         continue
      
      elif ypred == -1:
         continue
         
      else:
         raise Exception('Invalid value for ypred; Invalid class label')
      
   liq(time)
   balances.append((time, dollars))
   
   return_on_investment = ((dollars / dollars_init) * 100)
   print(f'roi = {return_on_investment:,.2f}%')
   
   return logs, balances

# if __name__ == '__main__':
def backtest(stock:Union[str, pd.DataFrame]='AAPL', model=None):
   if isinstance(stock, str):
      stock:pd.DataFrame = load_dataframe(str(stock), './stonks', format='feather')[['open', 'high', 'low', 'close']]
   else:
      pass

   if model is None:
      model_state = torch.load('./classifier_pretrained_state')
      model = torch.load('./classifier_pretrained.pt')
      model.load_state_dict(model_state)
      print(model)
   
   bal_init = 100.0
   logs, balances = run_backtest(model, stock, init_balance=bal_init)
   blogs = pd.DataFrame.from_records(balances, columns=('datetime', 'balance'))
   blogs = blogs.set_index('datetime', drop=True)
   
   close = stock.loc[blogs.index]['close']
   blogs['close'] = close
   print(blogs)
   
   bals:pd.Series = blogs.balance
   bal_final = bals.iloc[-1]
   rois = bals.diff().iloc[1:]
   
   P = rois[rois > 0].mean()
   L = rois[rois < 0].abs().mean()
   pl_ratio = (P / L)
   final_roi = (bal_final / bal_init)
   
   return Struct(
      pl_ratio=pl_ratio, 
      roi=final_roi,
      trade_logs=blogs
   )