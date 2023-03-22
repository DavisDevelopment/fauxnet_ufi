from functools import partial
from math import floor
from random import sample
from pprint import pprint
import numpy as np
# from numpy import typename
import torch
from torch import Tensor, tensor, typename
from faux.pgrid import ls

from main import list_stonks, load_frame, ExperimentConfig
from nn.data.sampler import DataFrameSampler
from s2s_transformer import samples_for

from nn.trees.binary_tree import TorchDecisionTreeRegressor, TorchDecisionTreeClassifier
from nn.trees.random_forest import TorchRandomForestClassifier, TorchRandomForestRegressor

from typing import *
from fn import F, _
from cytoolz import *

from pandas import DataFrame, Series
from tools import flatten_dict, unzip, maxby

import gc
from tqdm import tqdm

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

def samples_for2(symbol:Union[str, DataFrame], analyze:Callable[[DataFrame], DataFrame], xcols=None, xcols_not=[], ycol='SIG', x_timesteps=1):
   if isinstance(symbol, str):
      df:DataFrame = load_frame(symbol)
   elif isinstance(symbol, DataFrame):
      df:DataFrame = symbol
   else:
      raise ValueError('baw')
   
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
   # print(data)
   
   #TODO scaling the data
   
   if x_timesteps == 1:
      X = data[:, xcols]
   else:
      #TODO breaking the data up into episodes of {x_timesteps} length
      n_episodes = len(data) - x_timesteps
      X = np.zeros((n_episodes, x_timesteps, len(xcols)))
      for i in range(n_episodes):
         X[i, :, :] = data[i:i+x_timesteps, xcols]
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

def scale_dataframe(df:DataFrame, nq=500):
   from sklearn.preprocessing import QuantileTransformer
   scaler = QuantileTransformer(n_quantiles=nq)
   r = df.copy()
   # print(df)
   sX = scaler.fit_transform(X=df).T
   for i, c in enumerate(r.columns.tolist()):
      r[c] = sX[i]
   return r

# from ta_experiments import PGrid, Symbol, mkAnalyzer, expandps, implode, lrange
# if __name__ == '__main__':
def test_indicator_config(symbol, indicators, config={}):
   from sklearn.metrics import accuracy_score
   
   # indicators = mkAnalyzer(items)
   indicators = F(indicators.apply)
   
   # df = load_frame(symbol)
   # print(indicators(df))
   X, y = samples_for2(symbol, indicators, xcols_not=['open', 'high', 'low', 'close', 'datetime'])
   X = torch.from_numpy(X)
   y = torch.from_numpy(y)
   train_X, train_y, test_X, test_y = split_samples(X, y, 0.83)
   
   from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
   from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
   from sklearn.neural_network import MLPClassifier
   # from tsai.models import MINIROCKET
   
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
   
   results = []
   for model in models:
      # print(f'Fitting {typename(model)} on {len(train_X)} samples...')
      # print(train_X.shape, train_y.shape)
      model.fit(train_X, train_y)
      
      y_pred = model.predict(test_X)
      
      acc = 100.0 * accuracy_score(
         test_y.numpy(),
         y_pred
      )
      
      results.append(dict(
         model_type=type(model),
         model_args=model.get_params(),
         indicators=indicators,
         accuracy=acc
      ))
      # print(f'{typename(model)} Accuracy: {acc:.2f}%')
      
   results = DataFrame.from_records(results)
   results = results.sort_values('accuracy', ascending=False, ignore_index=True)
   results = results[['accuracy', 'model_args', 'model_type', 'indicators']]
   print(results)
   
   return results.iloc[0]

def test_minirocket(data=None):
   from tsai.models.MINIROCKET import MiniRocketClassifier, MiniRocketVotingClassifier
   from tsai.basics import get_UCR_data, timer
   
   if data is None:
      # Univariate classification with sklearn-type API
      dsid = 'OliveOil'
      X_train, y_train, X_valid, y_valid = get_UCR_data(dsid)   # Download the UCR dataset
   
   else:
      X_train, y_train, X_valid, y_valid = data
   
   # print(X_train.shape)
   # print(y_train)
   
   # Computes MiniRocket features using the original (non-PyTorch) MiniRocket code.
   # It then sends them to a sklearn's RidgeClassifier (linear classifier).
   model = MiniRocketClassifier()
   timer.start(False)
   model.fit(X_train, y_train)
   t = timer.stop()
   
   acc = model.score(X_valid, y_valid)
   print(f'valid accuracy    : {acc:.3%} time: {t}')
   # del model
   
   return dict(
      accuracy=acc,
      model=model
   )
      
if __name__ == '__main__':
   # test_minirocket()
   # exit()
    
   from faux.pgrid import PGrid
   from features.ta.loadout import Indicators, IndicatorBag
   
   hpg = PGrid(dict(
      seq_len=[10]
   ))
   
   #TODO: introduce a function to generate all combinations of N indicators
   ib = IndicatorBag()
   # ib.add('vwap', length=[10])
   
   ib.add('rsi', length=[2, 3])
   ib.add('zscore', length=[2, 3])
   
   ib.add('bbands', length=[10], mamode=['vwma', 'wma'], std=[1.25])
   # ib.add('accbands', length=[10])
   
   ib.add('bop')
   ib.add('mom')
   ib.add('inertia')
   #TODO add more T/A options
   
   ibcl = ls([x for x in ib.sampling()], True)
   print(ibcl)
   
   # TODO: convert the `expand` methods to generator functions, so that there aren't as many local functions allocated 
   # TODO ...as there are combinations of indicator options
   # ibcl = ib.expand()
   idll = map(lambda d: Indicators(*d.items()), ibcl)
   # 
   hpdl = list(hpg.expand())
   bests = []

   from time import sleep, time
   from olbi import printing, configurePrinting
   
   train_symbol = 'AMZN'
   pbar = tqdm(total=len(ibcl) * len(hpdl))
   for data_transform in idll:
      for hyperparams in hpdl:
         pbar.update()
         X, y = samples_for2(train_symbol, F(data_transform.apply), xcols_not=['open', 'high', 'low', 'close', 'datetime'], x_timesteps=hyperparams['seq_len'])
         X = torch.from_numpy(X)
         y = torch.from_numpy(y)
         train_X, train_y, test_X, test_y = split_samples(X, y, 0.83)
         train_X, test_X = tuple((v.unsqueeze(1) if v.ndim == 2 else v.swapaxes(1, 2)) for v in (train_X, test_X))
         
         data = tuple(map(lambda v: v.numpy(), (train_X, train_y, test_X, test_y)))
         
         best = test_minirocket(data=data)
         best['indicators'] = data_transform.items()
         best['hyperparams'] = hyperparams
         del best['model']
         
         pprint(best)
         
         bests.append(best)
         gc.collect()
   
   explogs = DataFrame.from_records(bests)
   alltimebest = maxby(bests, _['accuracy'])
   print(alltimebest)
   explogs:DataFrame = explogs.sort_values('accuracy', ascending=False)
   explogs.to_pickle('minirocket_exp_results.pickle')
   # explogs['indicators'] = explogs['indicators'].apply(lambda i: dict(i.items()))
   print(explogs)
   #TODO now, take the best configuration and fit MINIROCKET on an aggregate dataset
   #TODO ...then, evaluate that on a comprehensive list of symbols, and document performance statistics
   #TODO ...on each symbol individually and en aggregate

   winner = explogs.iloc[0]
   winning_analyzer = Indicators(*winner.indicators)
   winning_params   = winner.hyperparams
   
   eval_symbols =  sample(list_stonks(), 50)
   
   tests = []
   for symbol in eval_symbols:
      X, y = samples_for2(
         symbol=symbol, 
         analyze=F(winning_analyzer.apply), 
         xcols_not=['open', 'high', 'low', 'close', 'datetime'], 
         x_timesteps=winning_params['seq_len']
      )
      
      X = torch.from_numpy(X)
      y = torch.from_numpy(y)
      
      train_X, train_y, test_X, test_y = split_samples(X, y, 0.83)
      train_X, test_X = tuple((v.unsqueeze(1) if v.ndim == 2 else v.swapaxes(1, 2)) for v in (train_X, test_X))
      
      data = tuple(map(lambda v: v.numpy(), (train_X, train_y, test_X, test_y)))
      
      res:Dict[str, Any] = test_minirocket(data=data)
      res['symbol'] = symbol
      # best['indicators'] = winning_analyzer.items()
      # best['hyperparams'] = winning_params
      # del best['model']
      
      # pprint(best)
      
      tests.append(res)
   
   tests = DataFrame.from_records(tests).sort_values('accuracy', ascending=False)['symbol', 'accuracy', 'model']
   print(tests)
   tests.to_pickle('minirocket_results.pickle')
   
   exit()
   
   
   symkey = lambda item: (Symbol(id=item[0]), expandps(item[1]))
   pspace = list(map(symkey, [
      ('zscore', dict(length=[2, 14])),
      ('zscore', dict(length=[2, 14])),
      # ('rsi', dict(length=[3, 7])),
      # ('bbands', dict(length=lrange(3, 7, 1), std=[0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0], mamode=['wma'])),
      # ('accbands', dict(length=[3, 7, 14], mamode=['wma'])),
      # ('bop', dict()),
      # ('intertia', dict()),
      # ('mom', dict(length=[3, 5, 7, 9], offset=[0, 1, 2])),
   ]))
   
   def wimplode(d):
      top = {(k.name if isinstance(k, Symbol) else str(k)):implode(v) for k, v in d.items()}
      return top
   
   configs = list(map(wimplode, expandps(pspace)))
   pprint(configs)
   print(len(configs), ' possible configurations')
   input()
   
   from tqdm import tqdm
   
   for cfg in tqdm(configs):
      test_indicator_config('AMZN', cfg.items())