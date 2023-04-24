#!/usr/bin/env python
# coding: utf-8

# In[1]:


from functools import partial
from math import floor
from random import sample
from pprint import pprint
import random
import numpy as np
import pandas as pd
# from numpy import typename
import torch
from torch import Tensor, tensor, typename
from faux.pgrid import ls


# In[2]:


from coretools import list_stonks, load_frame


# In[3]:


from nn.trees.binary_tree import TorchDecisionTreeRegressor, TorchDecisionTreeClassifier
from nn.trees.random_forest import TorchRandomForestClassifier, TorchRandomForestRegressor


# In[4]:


from typing import *
from fn import F, _
from cytoolz import *


# In[5]:


from pandas import DataFrame, Series
from tools import flatten_dict, hasmethod, unzip, maxby, Struct


# In[6]:


import gc
from tqdm import tqdm
import pickle 


# In[7]:


import sys, os
P = os.path
from time import sleep, time
from olbi import printing, configurePrinting
import faux.backtesting.common as fauxbt_common
from faux.backtesting.common import split_samples


# In[8]:


from tsai.models.ROCKET import RocketClassifier
from tsai.models.MINIROCKET import MiniRocketClassifier, MiniRocketVotingClassifier


# In[9]:


samples_for2 = fauxbt_common.samples_for


# In[10]:


def scale_dataframe(df:DataFrame, nq=500):
   from sklearn.preprocessing import QuantileTransformer
   scaler = QuantileTransformer(n_quantiles=nq)
   r = df.copy()
   # print(df)
   sX = scaler.fit_transform(X=df).T
   for i, c in enumerate(r.columns.tolist()):
      r[c] = sX[i]
   return r


# In[11]:


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


# In[12]:


def test_minirocket(data=None, n_estimators=1):
   from tsai.models.MINIROCKET import MiniRocketClassifier, MiniRocketVotingClassifier
   # from tsai.models.MINIROCKET_
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
   model = (MiniRocketClassifier() if n_estimators == 1 else MiniRocketVotingClassifier(n_estimators=n_estimators, n_jobs=1))
   
   timer.start(False)
   model.fit(X_train, y_train)
   t = timer.stop()
   
   acc = model.score(X_valid, y_valid)
   # MiniRocketClassifier()
   print(f'valid accuracy    : {acc:.3%} time: {t}')
   # del model
   
   return dict(
      accuracy=acc,
      model=model
   )


# In[13]:


def fittest(data=None, ctor=MiniRocketClassifier, **ctor_kw):
   # from tsai.models.MINIROCKET_
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
   model = ctor(**ctor_kw)
   
   timer.start(False)
   model.fit(X_train, y_train)
   t = timer.stop()
   
   acc = model.score(X_valid, y_valid)
   print('\n'.join([
      f'valid accuracy  :  {acc:.3%}',
      f'time            :  {t}'
   ]))
   # del model
   
   return dict(
      accuracy=acc,
      model=model
   )


# In[14]:


def wrapped_minirockets(self, X:Tensor):
   if hasmethod(self, 'predict'):
      model = self
   elif isinstance(self, dict) and 'model' in self:
      model = self['model']
   else:
      raise ValueError('invalid self')
   if X.ndim == 2:
      X = X.unsqueeze(0)
   elif X.ndim == 3:
      pass
   
   # assert X.shape[2] == winning_params['seq_len'], f'expected {winning_params["seq_len"]}, but got {X.shape[2]}'
   
   ypred = model.predict(X.numpy())
   
   return torch.from_numpy(ypred)      


# In[15]:


def dcpy(target, src, *keys):
   for k in keys:
      target[k] = src[k]
      
from coinflip_backtest import backtest
from time import sleep
      
def backtest_minirockets(symbol, ts_idx, test_X, test_y, model, n_eval_days=90):
   totlist = lambda a: [a[i] for i in range(len(a))]
         
   slbeg, slend = -(n_eval_days * 2), -n_eval_days
   
   if n_eval_days < len(test_X):
      times, bX, by = (
         ts_idx[-n_eval_days:],
         test_X[len(test_X) - n_eval_days - 1:],
         test_y[len(test_X) - n_eval_days - 1:],
      )
   else:
      times = ts_idx[:]
      bX = test_X[:]
      by = test_y[:]
   
   wmodel = partial(wrapped_minirockets, model)
   btres = backtest(symbol, wmodel, samples=(times, bX, by), pos_type='long')
   
   return btres


# In[16]:


def cfgOptStep(train_symbol, hyperparams, data_transform, model=None, val_split=0.25, n_eval_days=365, unlink=True, backtest=True):
   ts_idx, X, y = samples_for2(
      symbol=train_symbol, 
      analyze=F(data_transform.apply) if not callable(data_transform) else data_transform, 
      xcols_not=['open', 'close', 'datetime'],
      x_timesteps=hyperparams['seq_len']
   )
   X = torch.from_numpy(X)
   y = torch.from_numpy(y)
   
   #* train the model on 85% of the available data, evaluating on the remaining 15%
   train_X, train_y, test_X, test_y = split_samples(X=X, y=y, pct=val_split, shuffle=False)
   train_X, test_X = tuple((v.unsqueeze(1) if v.ndim == 2 else v.swapaxes(1, 2)) for v in (train_X, test_X))
   
   if model is None:
      data = tuple(map(lambda v: v.numpy(), (train_X, train_y, test_X, test_y)))
      
      #best = test_minirocket(data=data, n_estimators=hyperparams['n_estimators'])
      best = fittest(data=data, ctor=MiniRocketClassifier)
      
      model = best['model']
      if unlink:
         del best['model']
      
   else:
      best = Struct()
   
   best['indicators'] = (data_transform if callable(data_transform) else data_transform.items())
   best['hyperparams'] = hyperparams
   
   if backtest:
      start_time = time()
      btres = backtest_minirockets(train_symbol, ts_idx, test_X, test_y, model, n_eval_days=n_eval_days)
      # print(f'took {time() - start_time}secs to run backtest')
      dcpy(best, btres, 'roi', 'baseline_roi', 'vs_market', 'pl_ratio')
   
   return best
   
def evaluateConfigurationOn(symbols:List[str], params, transform, model=None, val_split=0.25, n_eval_days=60):
   print(transform, params)
   entries = [
      cfgOptStep(sym, params, transform, model=model, val_split=val_split, n_eval_days=n_eval_days) for sym in symbols
   ]
   
   entries = [(e.asdict() if not isinstance(e, dict) else e) for e in entries]
   dfe:DataFrame = pd.DataFrame.from_records(entries)
   # print(dfe)
   
   res = Struct()
   for c, ctype in dfe.dtypes.to_dict().items():
      if ctype.name.startswith('float'):
         minc, meanc, maxc = (dfe[c].min(), dfe[c].mean(), dfe[c].max())
         res[c] = meanc
         res[f'min_{c}'] = minc
         res[f'max_{c}'] = maxc
      else:
         res[c] = dfe[c][0]
   
   print(res)
   return res


# In[17]:


def finetune_for(symbol:str, param_space, transforms, model_ctor=None, val_split=0.25, n_eval_days=365, n_winners=15):
   from tqdm import tqdm
   from faux.pgrid import PGrid
   from faux.features.ta.loadout import Indicators, IndicatorBag
   from cytoolz.itertoolz import topk
   
   #* prepare (hyper)parameter space
   if isinstance(param_space, PGrid):
      param_space = param_space.expand()
   elif isinstance(param_space, list):
      param_space = param_space
   elif isinstance(param_space, dict):
      param_space = [param_space]
   
   #* prepare space of possible analyses of the input data
   if isinstance(transforms, IndicatorBag):
      transforms = ls([x for x in transforms.sampling(4)], True)
   elif isiterable(transforms):
      transforms = list(transforms)
   else:
      transforms = [transforms]
      
   #* ensure that we know what type of model we'll be using
   if model_ctor is None:
      model_ctor = MiniRocketClassifier
   #* create progress bar, so we'll know where we're at
   pbar = tqdm(total=len(transforms) * len(param_space))
   
   attempts = []
   
   #TODO: replace this nested loop with some sort of non-brute-force grid search, using parallelism to accelerate computation
   for transform in transforms:
      for params in param_space:
         pbar.update()
         
         if model_ctor is None:
            model = model_ctor(**params)
         else:
            model = None
            
         res = cfgOptStep(symbol, params, transform, model=model, val_split=val_split, n_eval_days=n_eval_days)
         # res['params'] = params
         res['transform'] = transform
         
         attempts.append(res)
   winners = list(topk(n_winners, attempts, _['vs_market']))
   # print(winners[0])
   
   return winners


# In[18]:


def main():   
   from faux.pgrid import PGrid
   from faux.features.ta.loadout import Indicators, IndicatorBag
   
   last_run_path = 'minirocket_exp_results.pickle'
   best_config = None
   
   if P.exists(last_run_path):
      try:
         #* load last-run-results
         lrr:DataFrame = pd.read_pickle(last_run_path)
         print(lrr)
         best_config = lrr.iloc[0]
      except Exception:
         os.remove(last_run_path)
         pass
    
   if best_config is None:
      # Run GridSearch of all possible configurations
      hpg = PGrid(dict(
         seq_len=[15, 35],
         n_estimators=[1]
      ))
      
      #TODO: introduce a function to generate all combinations of N indicators
      ib = IndicatorBag()
      
      ib.add('rsi', length=[3])
      ib.add('zscore', length=[3])
      
      ib.add('bbands', length=[10, 15], mamode=['vwma'], std=[1.25, 1.75])
      # ib.add('donchian', length=[10])
      # ib.add('accbands', length=[10, 15])
      
      ib.add('delta_vwap')
      # ib.add('vwap')
      # ib.add('bop')
      # ib.add('mom')
      # ib.add('inertia')
      # ib.add('atr')
      #TODO add more T/A options
      
      ibcl = ls([x for x in ib.sampling(4)], True)
      # ibcl = ls(ib.expand(), True)
      print(ibcl)
      
      # TODO: convert the `expand` methods to generator functions, so that there aren't as many local functions allocated 
      # TODO ...as there are combinations of indicator options
      # ibcl = ib.expand()
      idll = [x for x in map(lambda d: Indicators(*d.items()), ibcl)]
      # ito = Indicators()
      # ito.add('rsi', length=2)
      # ito.add('zscore', length=2)
      # ito.add('bop', length=2)
      # ito.add('delta_vwap')
      # ito.add('bbands', length=10, mamode='vwma', std=1.25)
      # # ito.add('accbands', length=10)
      
      # idll = [ito]
      hpdl = list(hpg.expand())
      train_symbols = [
         'NOC',
         'PAYC', 'ETSY',
         'GE', 'AMZN',
      ]
      
      from tqdm import tqdm
      pbar = tqdm(total=len(idll) * len(hpdl))
      
      bests = []
      for data_transform in idll:
         for hyperparams in hpdl:
            pbar.update()
            res = evaluateConfigurationOn(train_symbols, hyperparams, data_transform, val_split=0.25, n_eval_days=365)
            # res['']
            print(res)
            bests.append(res.asdict())
   
      explogs = DataFrame.from_records(bests)
      # alltimebest = maxby(bests, _['roi'])
      # print(alltimebest)
      explogs:DataFrame = explogs.sort_values('vs_market', ascending=False)
      # explogs.to_pickle(last_run_path)
      # explogs['indicators'] = explogs['indicators'].apply(lambda i: dict(i.items()))
      # print(explogs)
      
      # winner = explogs.iloc[0]
      
      winners = explogs.iloc[:5]
      print(winners)
      #TODO evaluate top-K winners on each symbol in the loop below, saving the configuration which worked best for each given symbol
      
   elif best_config is not None:
      winner = best_config
      
   
   else:
      raise Exception('unreachable')
   
   #TODO now, take the best configuration and fit MINIROCKET on an aggregate dataset
   #TODO ...then, evaluate that on a comprehensive list of symbols, and document performance statistics
   #TODO ...on each symbol individually and en aggregate
   winning_analyzer = Indicators(*winner.indicators)
   winning_params   = winner.hyperparams
   
   eval_symbols =  sample(list_stonks(), 25)
   
   tests = []
   #TODO: aggregate training samples into a buffer during this loop, and then train a model on the aggregate afterwards to see if performance is improved
   # from nn.data.core import TensorBuffer
   
   data_buffer = []
   
   for symbol in eval_symbols:
      try:
         ts_idx, X, y = samples_for2(
            symbol=symbol, 
            analyze=F(winning_analyzer.apply), 
            xcols_not=['open', 'close', 'datetime'], 
            x_timesteps=winning_params['seq_len']
         )
         
         X = torch.from_numpy(X)
         y = torch.from_numpy(y)
         
         train_idx, train_X, train_y, test_idx, test_X, test_y = split_samples(index=ts_idx, X=X, y=y, pct=0.12, shuffle=False)
         train_X, test_X = tuple((v.unsqueeze(1) if v.ndim == 2 else v.swapaxes(1, 2)) for v in (train_X, test_X))
         # train_y = train_y[:-train_X.shape[2]]
         data = tuple(map(lambda v: v.numpy(), (train_X, train_y, test_X, test_y)))
         
         print(len(X))
         # data_buffer.append((train_X, train_y, test_X, test_y))
         data_buffer.append((X.unsqueeze(1).numpy() if X.ndim == 2 else X.swapaxes(1, 2).numpy(), y.numpy()))
         continue
      
         res:Dict[str, Any] = test_minirocket(data=data, n_estimators=1)
         res['symbol'] = symbol
         
         
         #TODO run the backtest on the model
         
         btres = backtest_minirockets(symbol, ts_idx, test_X, test_y, res['model'], n_eval_days=365)
         pickle.dump(res['model'], open(f'.pretrained_models/trained_for_{symbol}.pickle', 'wb+'))
                           
         dcpy(res, btres, 'roi', 'baseline_roi', 'pl_ratio', 'vs_market')
         tests.append(res)
      
      
      
      except ValueError as e:
         pprint(e)
         if input('(y/n)  continue?').lower().strip().startswith('n'):
            raise e
         else:
            continue
   
   if len(tests) > 0:
      tests = DataFrame.from_records(tests)
      tests = tests.sort_values('vs_market', ascending=False)[['symbol', 'vs_market', 'roi', 'baseline_roi', 'accuracy', 'model']]
      print(tests)
      tests.to_pickle('minirocket_results.pickle')
      
      model_efficacy = len(tests[tests.roi > 1.0])/len(tests)
      
      rois = tests.roi
      # P = rois[rois > 0].mean()
      # pl_ratio = 
      print('model efficacy: ' + '\n'.join(map(lambda x: f'  {x}', [
         f'pct-profitable: {model_efficacy*100:.3f}%',
         f'mean ROI:  {rois.mean()*100:3f}%',
      ])))
   
   #? take only most recent N-years of data, and train on data from multiple symbols
   trunc = lambda N, nd: nd[-N:] if N is not None else nd
   #* aggregate training-sample chunks
   # _train_X = np.concatenate([v[0].numpy() for v in data_buffer])
   # _train_y = np.concatenate([v[1].numpy() for v in data_buffer])
   # _test_X =  np.concatenate([v[2].numpy() for v in data_buffer])
   # _test_y =  np.concatenate([v[3].numpy() for v in data_buffer])
   
   _X = np.concatenate([x for x,_ in data_buffer])
   _y = np.concatenate([y for _,y in data_buffer])
   Nt = 12000
   train_X, train_y, test_X, test_y = split_samples(X=trunc(Nt, _X), y=trunc(Nt, _y), shuffle=True, pct=0.05)
   
   print(f'training MINIROCKET on {len(train_X)} samples..')
   res = test_minirocket(data=(train_X, train_y, test_X, test_y), n_estimators=1)
   pickle.dump(res, open('.res.pickle', 'wb+'))
   model = res['model']
   wmdl = partial(wrapped_minirockets, model)
   
   from faux.backtesting.multibacktester import PolySymBacktester, backtest
   
   start_date = pd.to_datetime('2022-01-01')
   tester, summary = backtest(eval_symbols, winning_params, winning_analyzer, wmdl, date_begin=start_date)
   print(summary)
   
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
      
def get_config_space():
   from faux.pgrid import PGrid
   from faux.features.ta.loadout import IndicatorBag, Indicators
   
   # Run GridSearch of all possible configurations
   hpg = PGrid(dict(
      seq_len=[15, 25, 35],
      n_estimators=[1]
   ))
   
   #TODO: introduce a function to generate all combinations of N indicators
   ib = IndicatorBag()
   ib.add('rsi', length=[2])
   ib.add('zscore', length=[2])
   ib.add('bbands', length=[10], mamode=['vwma'], std=[1.25])
   ib.add('delta_vwap')
   
   ib.add('stoch')
   # ib.add('atr')
   ib.add('obv')
   
   ibcl = ls([x for x in ib.sampling(5)], True)
   
   feature_specs = [x for x in map(lambda d: Indicators(*d.items()), ibcl)]
   random.shuffle(feature_specs)
   param_specs = list(hpg.expand())
   
   return param_specs, feature_specs[:18]


# In[ ]:





# In[20]:


def polyfinetune():
   from faux.features.ta.loadout import Indicators
   # from faux.backtesting.multibacktester import samples_from
   pofn = 'symbol_optima.pickle'
   pretrained_model_file = 'models_finetuned.pickle'
   initial_tunings_file = '.initial_tunings.pickle'
   
   if P.exists(pretrained_model_file):
      models:Dict[str, Any] = pickle.load(open(pretrained_model_file, 'rb'))
      symbols = list(models.keys())
   
   else:
      if P.exists(pofn):
         something_mhmm = pickle.load(open(pofn, 'rb'))
      
      if P.exists(initial_tunings_file):
         symbols, pspecs, fspecs = pickle.load(open(initial_tunings_file, 'rb'))
         pspecs = pspecs[0:1]
         optima = {}
      
      else:
         optima = {}
         symbols = sample(list_stonks(), 20)
         pspecs, fspecs = get_config_space()
         print(fspecs)
         print(len(fspecs))
         
         results = finetune_for(symbols[0], pspecs, fspecs, model_ctor=None, val_split=0.05, n_eval_days=100)
         for r in results:
            print(tuple(r.values()))
         # input()
         
         pspecs = [r['hyperparams'] for r in results]
         fspecs = [r['transform'] for r in results]
         pspecs = pspecs[0:1]
         
         optima[symbols[0]] = (pspecs[0], results[0]['transform'])
         pickle.dump((symbols, pspecs, fspecs), open(initial_tunings_file, 'wb+'))
         # input()
      
      failures = dict()
      for i in range(0, len(symbols)):
         symbol = symbols[i]
         
         try:
            print(f'TUNING OPTIMA for {symbol}')
            sym_results = finetune_for(symbol, pspecs, fspecs, val_split=0.05, n_eval_days=365)
            optima[symbol] = (sym_results[0]['hyperparams'], sym_results[0]['indicators'])
         
         except Exception as error:
            optima[symbol] = None
            failures[symbol] = error
            continue
         
         break
      
      pickle.dump(optima, open(pofn, 'wb+'))
      pprint(failures)
      print('OPTIMA-TUINING COMPLETE')
      
      models = {}
      for sym in symbols:
         if (sym not in optima or optima[sym] is None):
            continue
         
         (pspec, ispecs) = optima[sym]
         print(type(ispecs).__qualname__)
         
         tspec = ispecs if callable(ispecs) else Indicators(*ispecs)
         
         res = cfgOptStep(sym, pspec, tspec, val_split=0.15, n_eval_days=365, unlink=False, backtest=False)
         model = res['model']
         
         models[sym] = dict(
            model_ctor=type(model),
            model_ctor_args=(model.get_config() if hasmethod(model, 'get_config') else None),
            model=model,
            indicators=ispecs,
            hyperparameters=pspec
         )
         
      pickle.dump(models, open(pretrained_model_file, 'wb+'))
      
   assert models is not None
   
   #* load models
   model_entries = models
   pprint(model_entries)
   
   models = {sym:(model_entries[sym]['model']) for sym in symbols if (sym in model_entries and model_entries[sym] is not None)}
   params = [model_entries[sym] for sym in symbols if (sym in model_entries and model_entries[sym] is not None)]
   params = [p.get('hyperparameters', None) for p in params]
   transforms = [model_entries[sym]['indicators'] for sym in symbols if (sym in model_entries)]
   transforms = list(map(lambda t: Indicators(*t) if isiterable(t) else t, transforms))
   
   from faux.backtesting.multibacktester import PolySymBacktester, backtest
   
   print(transforms)
   backtest(symbols, params=params, transforms=transforms, model=models, date_begin='2022-01-01')
                  
   return models
   
if __name__ == '__main__':
   # main()
   polyfinetune()

