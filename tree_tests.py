from math import floor
import numpy as np
# from numpy import typename
import torch
from torch import Tensor, tensor, typename

from main import list_stonks, load_frame, ExperimentConfig
from nn.data.sampler import DataFrameSampler
from s2s_transformer import samples_for

from nn.trees.binary_tree import TorchDecisionTreeRegressor, TorchDecisionTreeClassifier
from nn.trees.random_forest import TorchRandomForestClassifier, TorchRandomForestRegressor

from typing import *
from fn import F, _
from cytoolz import *

from pandas import DataFrame, Series
from tools import flatten_dict, unzip

def samples_for(symbol:str, seq_len:int, columns:List[str]):
   df:DataFrame = load_frame(symbol, './sp100')
   df['abs_volume'] = df['volume']
   df['volume'] = df['volume'].pct_change()
   df = df.iloc[1:]
   df = df[columns]
   df['tmrw_close'] = df['close'].shift(-1)
   df = df.drop(columns=['low', 'high'])
   df['SIG'] = (df['tmrw_close'] > df['close']).astype('int')
   df = df.dropna()
   lookback_periods = [*list(range(1, 7)), 14, 21, 30]
   lbcs = []
   for lookback in lookback_periods:
      lb_change = df['close'].pct_change(periods=lookback)
      lb_col = f'close_lb{lookback}'
      lbcs += [lb_col]
      
      df[lb_col] = lb_change
   df = df[['close', *lbcs, 'SIG']].dropna()
   datacols = list(df.columns)
   x_columns = ['close', *lbcs]
   y_columns = ['SIG']
   data:np.ndarray = df.to_numpy()
   x = data[:, list(map(lambda c: datacols.index(c), x_columns))]
   y = data[:, datacols.index(y_columns[0])]
   return x, y

def samples_for2(symbol:str, analyze:Callable[[DataFrame], DataFrame], xcols=None, xcols_not=[], ycol='SIG'):
   df:DataFrame = load_frame(symbol)
   df['tmrw_close'] = df['close'].shift(-1)
   df['SIG'] = (df['tmrw_close'] > df['close']).astype('int')
   
   df = analyze(df)
   df = df.fillna(method='bfill').dropna()
   
   if xcols is None:
      xcols = list(set(df.columns) - set([ycol]) - set(xcols_not))
   
   df = df.drop(columns=list(xcols_not))
   cols = list(df.columns)
   xcols = [cols.index(c) for c in xcols]
   ycol = cols.index(ycol)
   data:np.ndarray = df.to_numpy()
   print(data)
   
   X = data[:, xcols]
   y = data[:, ycol]
   
   return X, y

def fmt_samples(x:Tensor, y:Tensor) -> Tuple[Tensor, Tensor]:
   print('x.shape = ', x.shape)
   print('y.shape = ', y.shape)
   
   return x, y

def split_samples(X, y, pct=0.2, shuffle=True):
   if shuffle:
      randsampling = torch.randint(0, len(X), (len(X),))
      X, y = X[randsampling], y[randsampling]

   test_split = pct
   spliti = round(len(X) * test_split)
   test_X, test_y = X[-spliti:], y[-spliti:]
   train_X, train_y = X[:-spliti], y[:-spliti]
   
   return train_X, train_y, test_X, test_y

from ta_experiments import mkAnalyzer, expandps, implode
# if __name__ == '__main__':
def test_indicator_config(symbol:str, items):
   from sklearn.metrics import accuracy_score
   
   indicators = mkAnalyzer(items)
   
   df = load_frame(symbol)
   print(indicators(df))
   X, y = samples_for2(symbol, indicators, xcols_not=['open', 'high', 'low', 'close', 'datetime'])
   X = torch.from_numpy(X)
   y = torch.from_numpy(y)
   train_X, train_y, test_X, test_y = split_samples(X, y, 0.83)
   
   from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
   from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
   from sklearn.neural_network import MLPClassifier
   
   models = [
      DecisionTreeClassifier(criterion='gini'),
      DecisionTreeClassifier(criterion='entropy'),
      DecisionTreeClassifier(criterion='log_loss'),
      RandomForestClassifier(n_estimators=100, criterion='entropy', n_jobs=-1),
      RandomForestClassifier(n_estimators=100, criterion='gini', n_jobs=-1),
      RandomForestClassifier(n_estimators=100, criterion='log_loss', n_jobs=-1),
      HistGradientBoostingClassifier(),
      ExtraTreeClassifier(criterion='entropy', splitter='best'),
      ExtraTreesClassifier(n_estimators=200, criterion='entropy', n_jobs=-1),
      
      MLPClassifier(hidden_layer_sizes=(120, 70, 32), learning_rate='adaptive'),
   ]
   
   for model in models:
      print(f'Fitting {typename(model)} on {len(train_X)} samples...')
      # print(train_X.shape, train_y.shape)
      model.fit(train_X, train_y)
      
      y_pred = model.predict(test_X)
      
      acc = 100.0 * accuracy_score(
         test_y.numpy(),
         y_pred
      )
      
      print(f'{typename(model)} Accuracy: {acc:.2f}%')
      
if __name__ == '__main__':      
   pspace = [
      ('zscore', dict(length=[14])),
      ('rsi', dict(length=[3, 7])),
      # ('rsi', dict(length=7)),
      # ('bbands', dict(length=7, std=0.82, mamode='wma')),
      ('bbands', dict(length=[7, 14, 20], std=1.25, mamode=['ema', 'wma'])),
      ('accbands', dict(length=[7, 14], mamode=['sma', 'ema']))
   ]
   pspace = flatten_dict(dict(pspace))
   print(pspace)
   configs = list(map(implode, expandps(pspace)))
   print(configs)
   
   for cfg in configs:
      test_indicator_config('AMZN', cfg.items())