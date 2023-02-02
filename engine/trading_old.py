from time import sleep
from pymitter import EventEmitter
from datetime import datetime, timedelta
from typing import *
import kraken
import shelvex as shelve
from pprint import pprint
from math import floor, ceil
import numpy as np
import re
import pandas as pd
#import modin.pandas as pd
from numba import jit, njit
from numba.experimental import jitclass
from functools import *
import toolz as tlz
from cytoolz import *
from operator import attrgetter
from cytoolz import dicttoolz as dicts
from cytoolz import itertoolz as iters
from fn import _
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

from engine.utils import *
from engine.datasource import *
from engine.utypes import Order

import engine.backend as be
from engine.backend import enginemethod

import ds.utils.bcsv as bcsv
from ds.utils.book import Book, TsBook, FlatBook
from ds.forecaster import NNForecaster
from ds.model.ctrl import MdlCtrl
from ds.model_train import descale
import pickle
import os
P = os.path
dt, td = datetime, timedelta
kr = kraken


#TODO: split this into two classes, {paper,real}_trading_engine with a base class
class TradingEngineBase(EventEmitter):
   initial_balance: Dict[str, float] = None
   balance: Dict[str, float] = None
   db: FlatBook[TsBook] = None
   book: TsBook
   accdf: pd.DataFrame
   current_date: pd.datetime = None
   forecasters: Dict[str, NNForecaster]
   exec_mode: str

   def __init__(self, db: FlatBook[TsBook] = None, forecasters=None, exec_mode='test'):
      super().__init__()

      self.exec_mode = exec_mode
      self.liquidity_symbol = 'USD'
      self.deployed = False
      self.maximum_investment = 25000
      self.initial_balance = dict(USD=350.0)
      self.balance_history = []
      self.forecast_logs = {}
      self.forecast_history = []
      self.forecasts = None
      self._dates = None
      self.balance = self.initial_balance.copy()
      self.db = None
      self.book = None
      if isinstance(db, Book):
         self.db = None
         self.accdf = None
         self.book = cast(Book, db)
      elif db is not None:
         self.db = db

         self.book = db.book
         self.book.data = keymap(cleansym, self.book.data)
         self.accdf = db.data

      if notnone(self.book):
         self.current_date = self.book.map(lambda d: d.index.min()).min()
      self.forecasters = forecasters
      self.scalers = None

      if self.is_live:
         self.client = kraken.krakenClient()
         self.pairs = kraken.get_pairs(self.client)
      else:
         self.client = None
      self._validate()

   @cached_property
   def symbols(self):
      return self.book.keys()

   @property
   def liquidity(self):
      return self.balance[self.liquidity_symbol]

   @liquidity.setter
   def liquidity(self, v):
      if self.is_live:
         raise Exception('TradingEngine.liquidity cannot be assigned directly when running live. Place an order, dipshit')

      assert not np.isnan(v) and v > 0 and np.isfinite(v), 'Cannot set dollars to negative or NaN value'
      self.balance[self.liquidity_symbol] = v

   @property
   def is_live(self): return self.exec_mode == 'live'

   @property
   def is_paper(self): return self.exec_mode == 'test'

   def sync(self):
      if self.is_paper:
         return True

      elif self.is_live:
         bal = self.client.get_account_balance()
         bal = {k: v for k, v in bal.to_records()}
         bal = format_balance(bal)
         bal = valfilter(_ != 0, bal)
         self.balance.update(bal)

   def balOf(self, sym: str):
      sym = cleansym(sym)
      return self.balance.get(sym, 0)

   def addBalance(self, sym: str, amount: float):
      sym = cleansym(sym)
      if self.is_live:
         raise ValueError('Cannot use .addBalance in live mode')
      self.balance[sym] = self.balOf(sym) + amount

   def _validate(self):
      pass

   # invest = enginemethod()

   @enginemethod
   def invest(self, symbol=None, weight: float = 1.0):
      today = self.book[symbol].loc[self.current_date]
      price = today['close']
      investable_dollars = min(self.maximum_investment, self.liquidity)
      volume = ((investable_dollars * weight)/price) * 0.9
      cost = (volume * price)
      self.liquidity -= cost
      self.addBalance(symbol, volume)

   @invest.live
   def _(self, symbol=None, weight: float = 1.0, instant=False):
      if self.book is None:
         raise ValueError('No data available')  # for symbol %s' % symbol)
      today = self.book[symbol].loc[self.current_date]
      price = today['close']
      investable_dollars = min(self.maximum_investment, self.liquidity)
      volume = ((investable_dollars * weight)/price) * 0.9
      cost = (volume * price)

      symbol = ksymbol(symbol)
      pair_match = self.pairs[self.pairs.eval(
          f'(base == "{symbol}" or base == "X{symbol}") and quote == "ZUSD"')]
      # print(pair_match)
      pair_code = pair_match.index[0]

      o = Order(pair=pair_code, type='buy', ordertype='limit',
                volume=volume, price=price, validate=(not self.deployed))
      if instant:
         o.ordertype = 'market'
         o.price = None
      do = o.to_dict()
      print(do)
      res = self.client.add_standard_order(**do)
      print(res)
      return res

   @enginemethod
   def divest(self, symbol, weight: float = 1.0, **kwargs):
      volume = (self.balOf(symbol) * weight)
      try:
         today = self.book[symbol].loc[self.current_date]
         price = (today['open'] + today['close'])/2.0
         value = (volume * price)
      except KeyError:
         # today = self.client.get_ohlc_data()
         #TODO this is not acceptable
         today = None
         price = None
         value = None
      symbol = cleansym(symbol)

      if self.is_paper:
         self.balance[symbol] -= volume
         self.liquidity += value
         return self.liquidity

      elif self.is_live:
         symbol = ksymbol(symbol)
         pair_match = self.pairs[self.pairs.eval(
             f'(base == "{symbol}" or base == "X{symbol}") and quote == "ZUSD"')]
         if len(pair_match) == 0:
            print('yo wtf u talkin bout, bro')
            return None

         pair = pair_match.iloc[0]
         # volume = np.round(volume, pair.pair_decimals)
         if pair.ordermin > volume:
            print(
                f'Cannot sell {symbol}; amount held is less than the minimum allowed order volume')
            return None
         pair_code = pair_match.index[0]
         fmt = "{:."+str(pair.pair_decimals)+"}"
         o = Order(pair=pair_code, type='sell', ordertype='market', volume=fmt.format(
             volume), starttm=(60 * 5), validate=(not self.deployed))
         o.update(**kwargs)
         print(o.to_dict())
         try:
            res = self.client.add_standard_order(**o.to_dict())
         except Exception as e:
            if 'Insufficient funds' in e.args[0]:
               o.volume = round(o.volume)
               res = self.client.add_standard_order(**o.to_dict())
            else:
               print(o.to_dict())
               raise e
         print(res)
         return res

   def liquidate_lowvolume(self, symbol):
      if self.is_live:
         symbol = ksymbol
      pair_match = self.pairs[self.pairs.eval(
          f'(base == "{symbol}" or base == "X{symbol}") and quote == "ZUSD"')]
      if len(pair_match) == 0:
         print('yo wtf u talkin bout, bro')
         return None

      pair = pair_match.iloc[0]
      pair_code = pair_match.index[0]
      o = Order(pair=pair_code, type='buy', ordertype='market',
                volume=pair.ordermin, validate=(not self.deployed))
      self.client.add_standard_order(**o.to_dict())
      sleep(1.0)
      self.sync()
      return self.divest(symbol)

   def liquidate(self):
      liqorders = []
      self.deployed = True
      for sym in self.balance.keys():
         if sym != 'USD':
            sellout = self.divest(sym)
            if sellout is None and self.is_live:
               sellout = self.liquidate_lowvolume(sym)
            liqorders.append(sellout)

      self.deployed = False
      self.sync()

   def build_scalers(self):
      if self.scalers is not None:
         return self.scalers

      def scalermap():
         return {k: MinMaxScaler() for k in 'open high low close'.split(' ')}

      bk = self.book
      scalers = {name: scalermap() for name in bk.keys()}
      # colstoscale = [(nnf.params.target_column if nnf.params.univariate else nnf.params.target_columns) for nnf in self.forecasters.values()]

      # for k, sm in scalers.items():
      for name in bk.keys():
         doc = bk[name]
         for c in scalers[name].keys():
            data = doc[c].to_numpy()
            sc = scalers[name][c]
            sc.fit(data.reshape(-1, 1))
      self.scalers = scalers

   def get_todays_args(self, nnf: NNForecaster):

      # df: pd.DataFrame = self.accdf
      bk: TsBook = self.book
      names = self.symbols
      args = {}
      findToday = indexOf(self.current_date)
      todayIdxs = bk.map(compose_left(attrgetter('index'), findToday))

      hp = nnf.params
      def symcols(name): return [f'{name}_{col}' for col in self.book.columns]
      column = hp.target_columns

      ret = {}

      for name in names:
         todayIdx = todayIdxs[name]
         if todayIdx == -1:
            ret[name] = None
            continue

         doc: pd.DataFrame = bk[name][bk.columns]
         subdf = doc.iloc[(todayIdx - hp.n_steps):todayIdx]
         if len(subdf) < hp.n_steps:
            ret[name] = None
            continue

         cdata = subdf[column].to_numpy()
         ret[name] = cdata

      return ret

   def apply_scaling(self, nnf: NNForecaster, args: Dict[str, np.ndarray], invert=False):
      if args is None:
         return None
      args = valfilter(notnone, args)
      outputs = args.copy()
      columns = nnf.params.target_columns
      for name in args.keys():
         features = args[name]
         if nnf.params.univariate:
            sc = self.scalers[name][nnf.params.target_column]
            scaled_features = (sc.transform if not invert else sc.inverse_transform)(
                features.reshape(-1, 1))[:, 0]
         else:
            sm = {i: self.scalers[name][col] for i, col in enumerate(columns)}
            scaled_features = descale(sm, features, inverse=invert)

         if not invert and (scaled_features[scaled_features > 1].sum() != 0 or scaled_features[scaled_features < 0].sum() != 0):
            print(ValueError('wtf', name, features, scaled_features))
            scaled_features[scaled_features > 1] = 1
            scaled_features[scaled_features < 0] = 0

         outputs[name] = scaled_features
      return outputs

   def convert_forecasts_to_rows(self, nnf, forecasts: Dict[str, np.ndarray]) -> Dict[str, pd.Series]:
      columns = nnf.params.target_columns
      d = self.book[self.book.keys()[0]]
      ex_row = d.loc[d.index.min()]
      freq = d.index.freq
      assert freq is not None

      ret = {}

      for name in forecasts.keys():
         pred = forecasts[name][0, :]
         pred_row = pd.Series(data=pred, index=columns)
         pred_row['time'] = (self.current_date + (1 * freq))

         ret[name] = pred_row

      return ret

   def generate_forecasts(self, nnf: NNForecaster, args: Dict[str, np.ndarray]):
      column = nnf.params.target_columns if not nnf.params.univariate else nnf.params.target_column

      outputs = {}

      for name in args.keys():
         input = args[name]
         try:
            output = nnf([input])
            outputs[name] = output
         except Exception as e:
            print(input)
            raise e

      outputs = self.apply_scaling(nnf, outputs, invert=True)

      return outputs

   @cached_property
   def correct_forecasts(self):
      bk = self.book
      entries = []

      cols = {}
      for name, doc in bk.items():
         price_mvmnt = ((doc.open + doc.close)/2).pct_change()
         cols[name] = price_mvmnt

      rois = pd.DataFrame.from_dict(cols)
      res = pd.DataFrame(data=None, index=rois.index)
      res['best_roi'] = None
      res['correct_investment'] = None

      for date, row in rois.iterrows():
         res.loc[date].best_roi = row.max()
         res.loc[date].correct_investment = row.idxmax()

      return res

   def evaluate_forecasts(self):
      hist = self.forecast_history
      real = self.correct_forecasts
      # real.rename(columns={})
      pivot = real.copy()
      pivot['pred_best_roi'] = None
      pivot['pred_correct_investment'] = None

      for entry in hist:
         date = entry.pop('date')
         recs = self.recommendations(entry, date=date)
         entry['date'] = date

         if len(recs) > 0:
            pred_best_name, pred_best_ret = recs[0]
            pivot['pred_correct_investment'].loc[date] = pred_best_name
            pivot['pred_best_roi'].loc[date] = pred_best_ret

      #TODO...
      return pivot

   def trade_on_forecasts(self, forecasts: Dict[str, np.ndarray]):
      todays = {cleansym(
          sym): self.book[sym].loc[self.current_date] for sym in self.book.keys()}
      forecasts = self.convert_forecasts_to_rows(
          self.forecasters['a'], forecasts)

      if self.is_paper:
         self.liquidate()
         print(f'day #{self.currentDateLoc + 1}, ${self.liquidity:,.2f}')

      projected_returns = {}
      for sym, tmrw in forecasts.items():
         today = todays[sym]
         # tmrw = forecasts[sym]
         #TODO something more nuanced than this would likely be superior
         cur_price = today.close
         nxt_price = tmrw.close
         # pred_price = pred
         projected_returns[sym] = (nxt_price / cur_price)-1.0

      projected_returns = valfilter(lambda x: x > 0, projected_returns)

      if self.liquidity < self.maximum_investment:
         k = 3
      else:
         k = int(self.liquidity // self.maximum_investment)

      ranked = list(topk(k, projected_returns.items(), key=lambda x: x[1]))

      # w = (1.0 / k)
      candkeys = [k for k, v in ranked]
      candidates = {k: v for k, v in ranked}

      if len(candkeys) >= 1:
         for k in candkeys:
            self.invest(k)

   def applyfc(self, fcname: str = 'a'):
      nnf = self.forecasters[fcname]
      args = self.get_todays_args(nnf)
      scaled_args = self.apply_scaling(nnf, args)
      scaled_args = valfilter(notnone, scaled_args)

      forecasts = self.generate_forecasts(nnf, scaled_args)

      self.forecast_history.append(
          merge(forecasts, {'date': self.current_date}))

      return forecasts

   def update_cache(self):
      if self.is_live:
         self.book = kraken_cache(update=True)
         self.emit('update.cache', self)

   def recommendations(self, forecasts: Dict[str, np.ndarray], n: int = None, date=None):
      date = date if date is not None else self.current_date
      todays = {cleansym(sym): self.book[sym].loc[date]
                for sym in self.book.keys()}
      forecasts = self.convert_forecasts_to_rows(
          self.forecasters['a'], forecasts)

      projected_returns = {}
      for sym, tmrw in forecasts.items():
         today = todays[sym]
         # tmrw = forecasts[sym]
         #TODO something more nuanced than this would likely be superior
         cur_price = today.close
         nxt_price = tmrw.close
         # pred_price = pred
         projected_returns[sym] = (nxt_price / cur_price)-1.0

      projected_returns = valfilter(lambda x: x > 0, projected_returns)

      if n is None:
         if self.liquidity < self.maximum_investment:
            n = 3
         else:
            n = int(self.liquidity // self.maximum_investment)

      ranked = list(topk(n, projected_returns.items(), key=lambda x: x[1]))

      return [(k, (v, forecasts[k])) for k, v in ranked]

   def select_investments(self):
      return self.recommendations(self.forecasts)

   def live_steps(self):
      yield None

      while True:
         self.sync()
         self.liquidate()
         self.current_date = pd.to_datetime(datetime.now().date())
         self.update_cache()
         self.sync()

         self.forecasts = self.applyfc('a')

         finadv = self.select_investments()
         (sym, (projected_return, tmrw)) = finadv[0]
         self.invest(sym, 0.2)
         self.sync()
         print(self.balance)
         #TODO render a time-elapsed progress bar in the interim
         print('bot.step completed! sleeping for the next 20hrs')
         sleep(4 * 60)  # sleep for 20hrs
         yield None

   def run_live(self):
      stepper = self.live_steps()
      self.build_scalers()

      # next(stepper)
      while True:
         next(stepper)

   def step(self):
      self.sync()

      forecasts = self.applyfc('a')

      self.trade_on_forecasts(forecasts)

   def run_paper(self):
      start, end = self.book.time_range
      self._dates = dates = pd.date_range(start, end)
      df = self.accdf

      self.build_scalers()

      for i in range(len(dates)):
         self.current_date = dates[i]
         assert self.current_date is not None
         self.liquidate()
         # self.balance_history[self.current_date] = self.balance.copy()
         self.balance_history.append(
             dict(date=self.current_date, balance=self.liquidity))

         self.step()

      self.history = hist = pd.DataFrame.from_records(
          self.balance_history, index='date')
      hist['roi'] = (hist['balance'].pct_change() * 100.0)
      hist['balance_formatted'] = hist.balance.apply(
          lambda s: "${:,.1f}".format(s))
      hist['roi_formatted'] = hist.roi.apply(lambda v: "{:.2f}%".format(v))
      hist.to_csv('paper_sess.csv')
      print(hist.to_string())

   def run(self):
      if self.is_live:
         self.run_live()

      elif self.is_paper:
         self.run_paper()

   @property
   def currentDateLoc(self):
      return self._dates.get_loc(self.current_date)

# for k,f in be._registry_.items():
   # setattr(TradingEngine, k, f)
# print()
