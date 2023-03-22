from olbi import printing

import itertools
from functools import reduce, wraps
import pickle
from time import sleep
from pprint import pprint
from numpy import asanyarray, ndarray
import termcolor
from torch import BoolTensor, random, renorm
from torch.jit._script import ScriptModule, ScriptFunction
from coinflip_backtest import backtest, count_classes
from nn.arch.transformer.TimeSeriesTransformer import TimeSeriesTransformer
from nn.common import *
from datatools import P, ohlc_resample, pl_binary_labeling, quantize, renormalize, unpack
   
from nn.namlp import NacMlp
from cytoolz import *
from cytoolz import itertoolz as iters
import pandas as pd

from datatools import get_cache, calc_Xy, load_ts_dump, rescale, norm_batches
from nn.data.core import TwoPaneWindow, TensorBuffer
from pandas import DataFrame
from typing import *
from nn.ts.classification.fcn_baseline import FCNBaseline2D, FCNNaccBaseline
# from nn.arch.transformer.TransformerDataset import TransformerDataset, generate_square_subsequent_mask
from tools import Struct, dotget, gets, maxby, minby, unzip, safe_div, argminby, argmaxby
from nn.arch.lstm_vae import *
import torch.nn.functional as F
from nn.ts.classification import LinearBaseline, FCNBaseline, InceptionModel, ResNetBaseline
from nn.arch.transformer.TransformerDataset import generate_square_subsequent_mask

from sklearn.preprocessing import MinMaxScaler
from ttools.thunk import Thunk, thunkv, thunk

def list_stonks(stonk_dir='./stonks', shuffle=True)->List[str]:
   stonk_dir = './stonks'
   from pathlib import Path
   from random import shuffle
   
   tickers = [str(P.basename(n))[:-8] for n in Path(stonk_dir).rglob('*.feather')]
   shuffle(tickers)
   
   return tickers

from cachetools import cached, Cache, keys

@cached(cache=Cache(25))
def load_frame(sym:str, dir='./stonks')->DataFrame:
   dir = './stonks'
   #* read the DataFrame from the filesystem
   df:DataFrame = pd.read_feather(P.join(dir, '%s.feather' % sym))
   
   #* iterating forward through the frame, replace all NA values with the last non-NA value
   df = df.fillna(method='ffill')
   #* iterating backward through the frame, replace all NA values with the last non-NA value
   df = df.fillna(method='bfill')
   #* drop any remaining rows containing NA values
   df = df.dropna()
   
   #* reindex the frame by datetime
   df = df.set_index('datetime', drop=False)
   
   return df

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

from tqdm import tqdm
from nn.optim import ScheduledOptim, Checkpoints

max_epochs = 100

def fit(model:Module, X:Tensor, y:Tensor, criterion=None, eval_X=None, eval_y=None, lr=0.001, epochs=None, eval_symbols=None, on_sampler=None, **kwargs):
   import random as rand
   assert criterion is not None
   epochs = epochs if epochs is not None else max_epochs
   
   inner_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
   optimizer = Checkpoints(
      model,
      inner_optimizer
   )
   optimizer.n_warmup_steps = 2
   
   crit = criterion

   fit_iter = tqdm(range(epochs))
   metric_logs = OrderedDict()
   
   eval_symbol = rand.choice(eval_symbols if eval_symbols is not None else list_stonks('./sp100'))
   do_evaluation = (eval_X is not None and eval_y is not None) or (eval_symbols is not None)
   fast_evaluation = kwargs.pop('fast_evaluation', eval_symbols is None)
   
   for e in fit_iter:
      optimizer.zero_grad()
      
      y_pred = model(X)
      
      loss = crit(y.squeeze(), y_pred.squeeze())
      loss.backward()
      
      fit_iter.set_description(f'epoch {e}')
      
      le = metric_logs[e] = Struct(epoch=e, loss=loss.detach().item(), accuracy=None)
      
      if do_evaluation:
         if eval_symbols is None or fast_evaluation:
            eval_metrics = evaluate(model, eval_X, eval_y)
            le.metrics = tuple(f'{n:.2f}' for n in (eval_metrics.p, eval_metrics.l, eval_metrics.false_positives)) #type: ignore
            le.n_pos_ids_P = eval_metrics.p #type: ignore
            le.n_pos_ids_L = eval_metrics.l #type: ignore
            le.total_neg_ids = (eval_metrics.false_positives / len(eval_y)) #type: ignore
            le.accuracy = eval_metrics.score #type: ignore
         
         else:
            with printing(False):
               le.accuracy = trading_score( #type: ignore
                  model=model,
                  eval_symbols=[eval_symbol], 
                  on_sampler=on_sampler, 
               ) 
            
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
   y = y.flatten()
   y_eval = m(X).flatten()
   # assert (y.shape == y_eval.squeeze().shape), f'{y.shape} != {y_eval.squeeze().shape}'
   
   count = lambda *bool_exprs: reduce(torch.logical_and, bool_exprs).int().count_nonzero().item()
   select = lambda array, *bool_exprs: array[reduce(torch.logical_and, bool_exprs)]
   
   labels = quantize(y, 3)
   pred_labels = quantize(y_eval, 3)
   
   matched = (labels == pred_labels)
   mismatched = (pred_labels != labels)
   
   # print(len(y_eval), count(matched), count(mismatched))
   # print(matched.numpy())
   # q_p = (labels == 1)
   # q_l = (labels == 0)
   # matched:BoolTensor = (pred_labels == labels)
   
   n_p, n_l = count(labels == 2), count(labels == 1)
   n_e = count(labels == 0)
   
   # n_matched, n_correct_p, n_correct_l = (
   #    count(matched),
   #    count(matched, q_p),
   #    count(matched, q_l)
   # )
   
   # n_false_p, n_false_l = (
   #    count(torch.logical_not(matched), pred_labels == 1),
   #    count(torch.logical_not(matched), pred_labels == 0)
   # )
   
   # misidentified = count(pred_labels == 1, labels == 0) + count(labels == 1, pred_labels == 0)
   
   # misids = safe_div(misidentified, n_p+n_l)

   # accuracy = safe_div(n_correct_p + n_correct_l, n_p + n_l) * 100
   
   # pct_pos_p = safe_div(n_correct_p, n_p)
   # pct_pos_l = safe_div(n_correct_l, n_l)
   
   # accuracy = (pct_pos_p + pct_pos_l) - misids
   nM, nMM = count(matched), count(mismatched)
   accuracy = (nM / len(labels))
   
   return Struct(
      score=accuracy,
      p=safe_div(count(matched, pred_labels == 2), n_p),
      l=safe_div(count(matched, pred_labels == 1), n_l),
      false_positives=nMM
   )
   
from dataclasses import dataclass, asdict
from cytoolz.dicttoolz import merge

@dataclass
class ExperimentConfig:
   in_seq_len:Optional[int] = None
   out_seq_len:Optional[int] = 1
   num_input_channels:Optional[int] = 4
   # input_channels:Optional[tuple] = ('open', 'high', 'low', 'close')
   num_predicted_classes:Optional[int] = 2
   
   symbol:Optional[str] = None
   df:Optional[DataFrame] = None
   
   shuffle:bool = True
   val_split:float = 0.12
   epochs:int = 30
   
   X_train:Optional[Tensor] = None
   y_train:Optional[Tensor] = None
   X_test:Optional[Tensor] = None
   y_test:Optional[Tensor] = None
   
   def __post_init__(self):
      # assert not (self.df is None and self.symbol is None), "either symbol name or a DataFrame must be provided"
      # assert self.in_seq_len is not None, "size of input sequence must be provided"
      
      if self.symbol is not None and self.df is None:
         self.df = load_frame(self.symbol, './sp100')
            
   # @wraps(ExperimentConfig)
   def extend(self, **params):
      kw = merge(asdict(self), params)
      return ExperimentConfig(**kw)
            
def prep_data_for(config:ExperimentConfig):
   df:DataFrame = config.df
   df['delta_pct'] = df['close'].pct_change()
   data = df.to_numpy()[1:]

   num_input_channels, in_seq_len, out_seq_len = config.num_input_channels, config.in_seq_len, config.out_seq_len
   assert None not in (in_seq_len, out_seq_len)
   win = TwoPaneWindow(in_seq_len, out_seq_len) #type: ignore
   win.load(data)


   scaler = MinMaxScaler()
   scaler.fit(data[:, :])

   seq_pairs = list(win.iter(lfn=lambda x: scaler.transform(x[:, :-1])))
   Xl, yl = [list(_) for _ in unzip(seq_pairs)]
   Xnp:ndarray
   ynp:ndarray 
   Xnp, ynp = np.asanyarray(Xl), np.asanyarray(yl).squeeze(1)

   ynp = ynp
   ynp:ndarray = pl_binary_labeling(ynp[:, 3])
   Xnp, ynp = Xnp, ynp

   X, y = torch.from_numpy(Xnp), torch.from_numpy(ynp)
   X, y = X.float(), y.float()
   X, y = shuffle_tensors_in_unison(X, y)

   X = X.swapaxes(1, 2)
   randsampling = torch.randint(0, len(X), (len(X),))
   X, y = X[randsampling], y[randsampling]

   test_split = config.val_split
   spliti = round(len(X) * test_split)
   X_test, y_test = X[-spliti:], y[-spliti:]
   X, y = X[:-spliti], y[:-spliti]

   X_test, X_train, y_test, y_train = X, X_test, y, y_test
   
   return X_train, y_train, X_test, y_test
   
def trading_score(eval_symbols=None, on_sampler=None, model=None):
   assert eval_symbols is not None
   assert model is not None
   
   from coinflip_backtest import backtest
   logs = []
   for sym in eval_symbols:
      testres = backtest(stock=sym, model=model, on_sampler=on_sampler)
      logs.append(dict(symbol=sym, score=testres.score))
   
   logs = pd.DataFrame.from_records(logs)
   print(logs)
   
   return logs.score.mean()

def shotgun_strategy(config:ExperimentConfig, **kwargs):
   # df = load_frame('MSFT')[['open', 'high', 'low', 'close']]
   num_input_channels, in_seq_len, out_seq_len = config.num_input_channels, config.in_seq_len, config.out_seq_len
   if config.X_train is None:
      X, y, X_test, y_test = prep_data_for(config)
   else:
      X, y, X_test, y_test = config.X_train, config.y_train, config.X_test, config.y_test

   core_kw = dict(in_channels=num_input_channels, num_pred_classes=config.num_predicted_classes)

   model_cores = [
      ResNetBaseline,
      FCNBaseline,
   ]

   rates = [
      0.0006,
      0.001,
      0.0015,
      0.002,
      0.00225,
      # 0.001,
      # 0.0015,
      # 0.002,
      # 0.00015,
   ]

   losses = [
      # torch.nn.BCELoss,
      MSELoss,
      L1Loss,
      # BCEWithLogitsLoss
   ]

   combos = list(itertools.product(model_cores, rates, losses))

   model_variants = (
      (
         base_factory,
         rate,
         crit_factory(),
         Sequential(
            # Dropout(p=0.32333), #? randomly zero-out 5% of inputs
            base_factory(**core_kw)
         )
      )
      
      for (base_factory, rate, crit_factory) in combos
   )

   experiments:List[Dict[str, Any]] = []
   
   from nn.data.sampler import DataFrameSampler
   
   on_sampler = kwargs.pop('on_sampler', None)
   if on_sampler is None:
      def on_sampler(sampler):
         from nn.data.sampler import add_indicators
         
         #* constrain evaluation period to most recent 365 days
         mints, maxts = sampler.date_range()
         mints = (maxts - pd.DateOffset(days=365))
         
         sampler.configure(
            in_seq_len=config.in_seq_len,
            min_ts=mints
         )
      
   eval_symbols = kwargs.pop('eval_symbols', None)
      
   #* for each variation of the configuration
   for mventry in model_variants:
      base_ctor, learn_rate, criterion, model = mventry
      model:Sequential = model
      print('\n'.join([
         f'baseline="{base_ctor.__qualname__}"', 
         f'learn_rate={learn_rate}', 
         f'loss_fn={criterion}'
      ]))
      
      hist, _ = fit(model, X, y, 
                    criterion=criterion, 
                    eval_X=X_test, 
                    eval_y=y_test, 
                    lr=learn_rate, 
                    epochs=config.epochs, 
                    
                    eval_symbols=eval_symbols, 
                    on_sampler=on_sampler,
                    fast_evaluation=True
                  )
      
      from coinflip_backtest import backtest
      
      #TODO: backtest on multiple symbols for a more informative performance summary
      testres = backtest(stock=config.symbol, model=model, on_sampler=on_sampler)
      
      entry_score = float(testres.score)
      
      #* store some info on this permutation of the model
      experiments.append(dict(
         model=model,
         model_state=model.state_dict(),
         mdl_config=Struct(loss_type=criterion, model_type=base_ctor.__qualname__),
         exp_config=config,
         score=entry_score,
         did_beat_market=(testres.did_beat_market and testres.did_place_orders),
         logs=hist
      ))

   if not any(e['did_beat_market'] for e in experiments):
      pprint(Exception('Failed to generate a model which beats the market :C'))

   #* select the permutation with the highest accuracy rating
   best_loop = maxby(experiments, key=lambda e: e['score'])
   
   score, model, mdl_config = gets(best_loop, 'score', 'model', 'mdl_config')
   
   return best_loop

def evaluate_loop(model=None, loop=None, symbols=None, n_symbols=None):
   import matplotlib.pyplot as plt
   from coinflip_backtest import backtest
   import random as rand
   from termcolor import colored
   
   loop:Struct
   
   if isinstance(model, dict):
      loop = model
   elif loop is not None and model is None:
      model = loop['model']
      
   if symbols is None and n_symbols is not None:
      symbols = rand.sample(list_stonks('./sp100'), n_symbols)
   else:
      symbols = symbols if symbols is not None else list_stonks('./sp100')

   print(colored('EVALUATING', 'yellow', attrs=['dark']), f' on {len(symbols)} symbols')
   
   elogs = []
   
   for symbol in symbols:
      trade_sess = backtest(stock=symbol, model=model)
      
      pprint(dict(
         symbol=symbol,
         pl_ratio=trade_sess.pl_ratio,
         roi=trade_sess.roi
      ))
      
      logs:pd.Series = trade_sess.trade_logs #type: ignore
      logs['datetime'] = logs.index
      
      #* (WHERE APPLICABLE) compute the margin by which our agent outperformed a BUY/HOLD strategy
      final_roi_held = trade_sess.baseline_roi
      final_roi_traded = trade_sess.roi
      advantage = (final_roi_traded - final_roi_held)/final_roi_held*100.0
      absadv = abs(advantage)
      print('Agent ', colored('over' if advantage > 0 else 'under', attrs=['bold']) + f'performed by {absadv:.2f}%')
      
      #* store information about this eval-iteration
      elogs.append(dict(
         symbol=symbol,
         roi=final_roi_traded,
         baseline_roi=final_roi_held,
         score=advantage
      ))
      
      if not trade_sess.did_beat_market:
         continue
      
      elif final_roi_traded == 1.0 and not any(logs.roi != 1.0):
         #* skip iterations for which the Agent never receives a BUY signal
         #? ... which I could be doing in a more conceptually direct way
         continue
      
      try:
         #* Plot the symbol's close against the agent's trading balance
         logs.plot(
            title=f'CoinFlipNet({symbol}) Trading Balance', 
            x='datetime',
            y=['close', 'balance'],
            figsize=(24, 16), 
            fontsize=14,
            legend=True, 
            stacked=True
         )
         
         #* save the plot to a file
         plt.savefig(f'./figures/{symbol}_coin_flip_net.png')
      
      except Exception as error:
         #* when an exception is raised during plotting, print the data we were trying to plot for debugging purposes
         print(logs)
         pprint(error)
      
      print(f'BEAT THE MARKET by {advantage:,.2f}% on {symbol}')
   
   #* convert the evaluation logs to a DataFrame, and print it to the console
   elogs = DataFrame.from_records(elogs)
   
   if len(elogs) == 0:
      return None
   
   return elogs

def ensemble_stage1(cache_as='ensemble', rebuild=False, **kwargs):
   from fn import _, F
   from nn.arch.contrarion import ApexEnsemble, Args
   from cytoolz.dicttoolz import valmap
      
   proto = ExperimentConfig(
      in_seq_len = kwargs.pop('in_seq_len', 14),
      num_input_channels = kwargs.pop('num_input_channels', 4),
      num_predicted_classes=2,
      epochs=75,
      val_split=0.4
   )
   
   if rebuild or cache_as is None or not P.exists(f'.{cache_as}_pretrained.pt'):
      symbols = list_stonks()
      import random as rand
      
      symbols = rand.sample(symbols, 2)
      model_factory = F(train_models_for)
      
      configs = []
      for sym in symbols:
         configs.append(proto.extend(symbol=sym))
      
      pprint(configs)
      
      winners = model_factory(configs)
      winner_models = valmap(lambda x: x[1] if isinstance(x, tuple) else (x['model'] if isinstance(x, dict) else x), winners)
      
      ensemble = ApexEnsemble(winner_models, Args(
         input_shape=[proto.in_seq_len, proto.num_input_channels], #type: ignore
         **asdict(proto)
      ))
      
      torch.save(ensemble.state_dict(), f'.{cache_as}_state_dict.pt')
      torch.save(ensemble, f'.{cache_as}_pretrained.pt')
      
   elif P.exists(f'./.{cache_as}_pretrained.pt'):
      ensemble:ApexEnsemble = torch.load(f'.{cache_as}_pretrained.pt')
      ensemble_state = torch.load(f'.{cache_as}_state_dict.pt')
      ensemble.load_state_dict(ensemble_state)
      
   else:
      raise Exception('Ensemble not assembled')
   
   assert ensemble is not None
   
   for sym in ensemble.component_names:
      backtest(stock=sym, model=ensemble)

def ensemble_stage2(symbols=None, proto=None, cache_as='ensemble', rebuild=False, **kwargs):
   from fn import _, F
   from nn.arch.contrarion import ApexEnsemble, Args
   from cytoolz.dicttoolz import valmap
   from nn.data.agg import aggds
   import random as rand
   
   proto = proto if proto is not None else ExperimentConfig(
      in_seq_len = kwargs.pop('in_seq_len', 14),
      num_input_channels = kwargs.pop('num_input_channels', 5),
      num_predicted_classes=3,
      epochs=kwargs.pop('epochs', 7),
      val_split=0.15
   )
   
   symbols = thunkv(symbols) if (symbols is not None) else [
      'MSFT', 'GOOG', 'AAPL',
   ]
   
   ds = aggds(proto, symbols)
   proto, X_train, y_train, X_test, y_test = ds
   
   proto.X_train, proto.y_train, proto.X_test, proto.y_test = X_train, y_train, X_test, y_test
   
   print(f'No. of training samples: {len(X_train)}')
   
   from datatools import training_suitability
   
   #* select from only the most 'suitable' datasets for evaluation
   most_suiteds = topk(2, symbols, lambda sym: training_suitability(load_frame(sym, './sp100')))
   
   print(f'Most suitable symbol is {most_suiteds}')
   best_loop = shotgun_strategy(proto.extend(symbol=rand.choice(most_suiteds)), eval_symbols=most_suiteds)
   
   model = best_loop['model']
   
   n_eval_symbols = kwargs.pop('n_eval_symbols', 50)
   elogs = evaluate_loop(loop=best_loop, n_symbols=n_eval_symbols)
   
   import matplotlib.pyplot as plt
   
   return symbols, best_loop, elogs

def generate_ensemble_components(n_winners=25, n_symbols=25, filter_symols=None, **kwargs):
   import random as rand
   all_symbols = list_stonks('./sp100')
   
   viable = []
   
   bsymbols = None
   
   colored = termcolor.colored
   
   from time import time
   break_meter = 0
   
   while len(viable) < n_winners:
      #TODO: when a winning model is generated, ensure that all info necessary to replicate the winning results is provided, 
      #TODO ... and pass that configuration to another routine to be mutated iteratively, searching for improvements
      
      try: #* keyboardinterrupt try/catch block
         symbols = thunk(lambda : (bsymbols if bsymbols is not None else []) + rand.sample(all_symbols, n_symbols))
         
         print('\n\n', colored('SYMBOLS=', 'cyan', attrs=['bold']), '   ', colored(f'{symbols}', 'yellow'), '\n\n')
         
         tbeg = time()
         symbols, winner, elogs = ensemble_stage2(
            symbols=symbols,
            epochs=20,#rand.randint(2, 6),
            n_eval_symbols=50
         )
         tend = time()
         
         print('Took ', colored(f'{(tend - tbeg)/60}', 'yellow', attrs=['bold']) + 'min')
         
         if elogs is None:
            print(
               colored('Failed to produce a winning model', 'red', attrs=['bold']), 
               '\n',
               colored('Continuing...', 'cyan', 'on_red')
            )
            continue
         
         elogs:DataFrame
         
         viable_logs:DataFrame = elogs[elogs.score > 0.0]
         
         if (len(viable_logs)/len(elogs) < 0.5):
            #? reject the model if there were more symbols for which it did not facilitate profitable avtive trading than for which it did
            print(
               colored('Failed to produce a model which beats the market on more than half of the evaluated symbols', 'red', attrs=['bold']), 
               '\n',
               colored('Discarding iteration and continuing...', 'cyan', 'on_red')
            )

            continue
         
         viable_logs.sort_values('score', ascending=False, inplace=True)
         
         print(viable_logs)
         numb = lambda num: colored(f'{num}', 'yellow', attrs=['bold'])
         punc = lambda sym: colored(f'{sym}', 'magenta', attrs=['bold'])
         
         if len(viable_logs) > 0:
            print(
               colored('QUALIFIED!!', 'green', attrs=['bold']), '\n - ', 
               (numb(len(viable_logs)) + ' ' + punc('/') + ' ' + numb(len(elogs))),
               colored(' passed', 'green', attrs=['bold']),
            )
            
            viable.append(dict(
               symbols=symbols[:],
               eval_logs=elogs,
               loop=winner
            ))
            
            print(len(viable), ' viable candidates generated so far')
            print('candidate symbols are ', symbols)
            # input()
            
         #* each time a full iteration completes organically, decrement the BREAK-meter
         break_meter -= 1
         
      except KeyboardInterrupt:
         print(colored('Interrupted! Skipping..', 'cyan', attrs=['bold']))
         
         break_meter += 1 #* increment the BREAK-meter
         if break_meter > 10:
            break
         else:
            continue
   
   results_path = './ensemble_viable_results.pt'#'./ensemble3_results.pickle'
   torch.save(viable, results_path)
      
   results:DataFrame = DataFrame.from_records(viable)
   # print(results)
   # results.to_pickle(results_path)
   
   print('Ensemble-3 results persisted to filesystem at %s' % results_path)
   
   return results
      
def ensemble_stage4(stage3_results:DataFrame, **kwargs):
   r = stage3_results
   if len(stage3_results) == 0:
      raise Exception('baw, dafuq')
   
   r['loop']
   
def assemble_pretrained_ensemble(res_path):
   data = torch.load(res_path)
   components = []
   
   for i, entry in enumerate(data):
      model, model_state = dotget(entry, 'loop.model'), dotget(entry, 'loop.model_state')
      model:torch.nn.Module
      assert None not in (model, model_state)
      model.load_state_dict(model_state)
      
      components.append(model)
      
   print(components)
   from nn.arch.ensemble import TsClassifyingEnsemble
   
   ensemble = TsClassifyingEnsemble(
      components=components,
   )
   
   for sym in list_stonks('./sp100'):
      backtest(sym, model=ensemble, pos_type='long')
      #TODO: test with pos_type='both'
   
   return ensemble

if __name__ == '__main__':
   import sys 
   argv = sys.argv[1:]
   
   if 'figures' in argv:
      evaluate_loop()
      exit()
      
   elif 'ensemble' in argv:
      rebuild = ('rebuild' in argv)
      ensemble_stage2(symbols=list_stonks('./sp100'), rebuild=rebuild)
      exit()
      
   elif 'ensemble-3' in argv:
      generate_ensemble_components()
      exit()
      
   else:
      assemble_pretrained_ensemble('ensemble_viable_results.pt')
      
   
   