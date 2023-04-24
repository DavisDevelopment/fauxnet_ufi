
from typing import *
import pandas as pd
import numpy as np
from pandas import DataFrame
from coretools import load_frame
from tools import isiterable

def renormalize(n, min1=None, max1=None, min2=None, max2=None):
   # if min1 is not None and max1 is not None and min2 is None and max2 is None:
      # min2, max2 = min1, max1
      # min1, max1 = n.min(), n.max()
   
   assert min1 is not None and max1 is not None, f'first range must be provided via min1 and max1 arguments'
   assert min2 is not None and max2 is not None, f'second range must be provided via min2 and max2 arguments'
   
   delta1 = max1 - min1
   delta2 = max2 - min2
   
   return (delta2 * (n - min1) / delta1) + min2

def samples_for(symbol:Union[str, DataFrame], analyze:Callable[[DataFrame], DataFrame], xcols=None, xcols_not=[], x_timesteps=1, y_type='pl_binary', y_lookahead=1):
   if isinstance(symbol, str):
      df:DataFrame = load_frame(symbol)
      df.name = symbol
   elif isinstance(symbol, DataFrame):
      df:DataFrame = symbol
      symbol = df.name
   else:
      raise ValueError('baw')
   
   # df['future_close'] = df['close'].shift(-y_lookahead)
   # df['tmrw_close'] = df['close'].shift(-1)
   
   if y_type == 'pl_binary':
      df['$LABEL'] = (df['close'].shift(-y_lookahead) > df['close']).astype('int')
   else:
      raise NotImplemented(f'Handling of y_type="{y_type}" :C')
   
   df = analyze(df)
   df = df.fillna(method='bfill').dropna()
   
   if xcols is None:
      xcols = list(set(df.columns) - set(xcols_not) - {'$LABEL'})
   
   df = df.drop(columns=list(xcols_not))
   cols = list(df.columns)
   xcols = [cols.index(c) for c in xcols]
   ycol = cols.index('$LABEL')
   
   data:np.ndarray = df.to_numpy()
   
   #TODO scaling the data
   
   tsi:List[pd.Timestamp] = []
   if x_timesteps == 1:
      X = data[:, xcols]
      assert X.shape[1] == len(xcols), f'{X.shape[1]} != {len(xcols)}'
      tsi = df.index.tolist()
   
   else:
      #TODO breaking the data up into episodes of {x_timesteps} length
      n_episodes = len(data) - x_timesteps
      X = np.zeros((n_episodes, x_timesteps, len(xcols)), dtype=np.float64)
      for i in range(n_episodes):
         X[i, :, :] = data[i:i+x_timesteps, xcols]
         assert X.shape[2] == len(xcols), f'{X.shape[2]} != {len(xcols)}'
         ts = df.index[i+x_timesteps]
         tsi.append(ts)
      
   y = data[:, ycol]
   
   return tsi, X, y

def split_samples(X=None, y=None, index=None, pct=0.2, shuffle=True): #type:ignore
   assert (X is not None and y is not None)
   
   import torch
   if shuffle:
      sampling = torch.randint(0, len(X), (len(X),))
      # X, y = X[sampling], y[sampling]
   else:
      sampling = torch.arange(len(X))
   # print(sampling)
   
   X, y = X[sampling], y[sampling]
   assert len(X) == len(y)
   
   # if index is None and (pct is None or pct == 0):
   #    return X, y
   
   if index is not None:
      index = [index[i] for i in sampling.tolist()]
      assert len(index) == len(X)
   
   test_split = pct
   spliti = round(len(X) * test_split)
   
   
   test_X, test_y = X[-spliti:], y[-spliti:]
   train_X, train_y = X[:-spliti], y[:-spliti]
   print(len(train_X), len(test_X))

   split_vals = (
      *ensure_eq_sized(train_X, train_y), 
      *ensure_eq_sized(test_X, test_y)
   )
   
   if index is not None:
      train_index, test_index = (
         index[:-spliti:],
         index[-spliti:], 
      )
      
      split_vals = (
         *ensure_eq_sized(train_index, train_X, train_y), 
         *ensure_eq_sized(test_index, test_X, test_y)
      )
   
   return split_vals

def eq_sized(*els):
   n = None
   for el in els:
      if n is None:
         n = len(el)
         continue
      elif n != len(el):
         return False
   return True


def ensure_eq_sized(*els):
   assert eq_sized(*els), f'Inconsistent sizing: {list(map(len, els))}'
   return els


# In[33]:


def ensure_equivalence(*args):
    if len(args) == 1 and isiterable(args[0]):
        args = args[0]
    if len(args) == 0:
        return None
    first_arg = args[0]
    for i, arg in enumerate(args[1:], start=1):
        if first_arg != arg:
            if not hasattr(first_arg, '__eq__') or not first_arg.__eq__(arg):
                raise ValueError(f"Arguments at index {i} and 0 are not equivalent; {arg} != {first_arg}")
    return args[-1]
