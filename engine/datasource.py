from ds.utils.book import Book, TsBook, FlatBook
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
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

import kraken
kr = kraken

from fn import Stream

from engine.utils import *
from tools import *

def raw_full_dataset():
   bk = bcsv.load(open('cache/all_data.dat', 'rb'))
   bk = TsBook(data=bk)
   return bk


def load_full_dataset(truncate=None, refresh=False):
   if P.exists('full_dataset.pickle') and not refresh:
      return pickle.load(open('full_dataset.pickle', 'rb'))

   ld = bcsv.load(open('cache/all_data.dat', 'rb'))
   bk = TsBook(data=ld)
   # print(len(list(bk.keys())))

   if truncate is not None:
      bk = truncate(bk)

   bk = TsBook(bk.data)
   pickle.dump(bk, open('full_dataset.pickle', 'wb+'))

   return bk


def literally_flatten(db: TsBook):
   start, end = db.time_range
   # acc = pd.DataFrame(data=None, index=pd.date_range(start, end))
   acc = {}
   for name, doc in db.items():
      if name.endswith('usd'):
         name = name.replace('usd', '')

      for col in db.columns:
         acc[f'{name}_{col}'] = doc[col]
   
   acc = pd.DataFrame(data=acc, index=pd.date_range(start, end))

   return acc


def flat_full_dataset():
   now = dt.now().date()
   end = pd.to_datetime(now - td(days=floor(365 / 2)))
   start = pd.to_datetime(end - td(days=floor(365 * 1.5)))

   db = load_full_dataset(truncate=has_range(start, end), refresh=True)
   for doc in db.values():
      # fill NaN values using linear interpolation
      doc.interpolate(method='linear', inplace=True)
      # drop all but selected rows
      doc.drop(doc[doc['time'] < start].index, inplace=True)
      assert doc['time'].min() == start
      doc.drop(doc[doc['time'] > end].index, inplace=True)
      assert doc['time'].max() == end
      # doc.drop(doc[(doc['time'] < start)|(doc['time'] > end)].index, inplace=True)
      print(doc)

   to_drop = []
   names = db.keys()
   exclude_pairs = ['btc', 'eth', 'ust', 'eos', 'gbp', 'eur', 'jpy', 'xch']
   for name in names:
      for p in exclude_pairs:
         if name != p and name.endswith(p):
            to_drop.append(name)
            continue

      if name.endswith('usd'):
         wo_usd = name.replace('usd', '')
         to_drop.append(wo_usd)

   for name in to_drop:
      del db[name]
   db._refresh_time_range()

   print({name: len(doc) for name, doc in db.items()})

   dbstart, dbend = db.time_range
   assert start == dbstart and end == dbend, f'{dbstart}!={start}, {dbend}!={end}'

   print(len(db), (dbend - dbstart).days)
   return literally_flatten(db)


def kraken_cache(update=False):
   df_dict = kr.load_cache() if not update else kr.update_cache()
   df_dict = keymap(cleansym, df_dict)
   for k, df in df_dict.items():
      if type(df.index) is pd.RangeIndex:
         df.set_index('dtime', drop=True, inplace=True)
      
      df['timecode'] = df['time']
      df['time'] = df.index
      
      if not isinstance(df.index, pd.DatetimeIndex):
         print(df)
         raise ValueError('Chooooo, baw u like craayzeyy stoopid, sha')
      
      if df.index.freq is None:
         freq = pd.infer_freq(df['time'])
         if freq is None:
            pass
         else:
            idx = df.index.copy()
            idx.freq = freq
            df.set_index(idx, inplace=True)

   df_dict = keymap(cleansym, df_dict)
   precleansize = len(df_dict)
   pairs:pd.DataFrame = kr.get_pairs()
   real_syms = pairs['base'].unique().tolist()
   syms = list(df_dict.keys())
   
   # print(real_syms)
   # print(set(syms) & set(real_syms))
   # print(set(syms) - set(real_syms))
   for s in syms:
      if s not in real_syms:
         del df_dict[s]
   
   # print(set(syms) | set(real_syms))
   # for sym in syms:
   #    if sym.startswith('X') and sym[1:] in syms:
   #       del df_dict[sym]
   print(len(df_dict), precleansize - len(df_dict))
   # exit()
   df_dict = valfilter(lambda d: len(d) >= 720, df_dict)

   return df_dict

from typing import *

class DataSource(TsBook):
   freq:pd.DateOffset
   _agg:Optional[pd.DataFrame]
   
   def __init__(self, data=None):
      self._rawdata = data
      if data is not None:
         data = {k:self._inline_map(d) for k,d in data.items()}
         
      super().__init__(data=data)
      
      self.pairs = None
      self._agg = None
      self.__flat()
      
   def __flat(self):
      flatdf:pd.DataFrame = literally_flatten(self)
      print(flatdf.columns)
      self._agg = flatdf
      
   def _inline_map(self, data_component:pd.DataFrame):
      return data_component
   
   def isuptodate(self)->bool:
      raise NotImplementedError()
   
   def notuptodate(self) -> Iterable[str]:
      raise NotImplementedError()
   
   def update(self):
      pass
   
   def at(self, date:pd.Timestamp)->Dict[str, pd.Series]:
      #TODO return a DataFrame object, indexed by symbol
      res = {}
      date = pd.to_datetime(date).normalize()
      flat_at = self._agg.loc[date]
      for k in self.keys():
         cols = getcolsmatching(f'{k}_*', self._agg)
         r:pd.Series = flat_at[cols]
         res[k] = r.rename(lambda s: s.replace(f'{k}_', ''))
      return res
   
      for key, doc in self.items():
         if date in doc.index:
            row = doc.loc[date]
            res[key] = row
         else:
            res[key] = None
      return res
   
   def drop(self, sym):
      if isinstance(sym, str):
         del self.data[sym]
      elif isinstance(sym, (list, tuple)):
         for s in sym:
            del self.data[s]
      self._refresh_time_range()
   
class KrakenDataSource(DataSource):
   def __init__(self, interval=None):
      self.interval = interval if interval is not None else kr.TimeInterval.Day
      self.kc = kr.KrakenCache(interval=self.interval)
      self.freq = pd.tseries.frequencies.to_offset(str(self.interval))
      self.nsamples = 200
      
      docs = self.kc.load_cache()
      
      docs = keymap(lambda k: nonkrakenSymbol(k), keyfilter(lambda k: nonkrakenSymbol(k) is not None, docs))
      
      super().__init__(data=docs)
      
      self.pairs = self.kc.get_pairs()
      
      self._validate()
         
      self._refresh_time_range()
      
   def _validate(self):
      for name, d in self.items():
         if str(d.index.dtype) != 'datetime64[ns]':
            print(name)
            print(d)
            raise ValueError('only timeseries are supported')
         
   def _set_data(self, data=None):
      super()._set_data(data=data)
      self._validate()
      self._refresh_time_range()
      
   def reload(self):
      self._set_data(data=self.kc.load_cache())
      # self._refresh_time_range()
      
   def _inline_map(self, df:pd.DataFrame):
      # self.freq.nanos()
      # r = df.copy(deep=True)
      if self.nsamples is not None:
         r = df[df.index > (df.index.max() - (self.freq * self.nsamples))].copy(deep=True)
      else:
         r = df.copy(deep=True)
      
      #? ensure that the data is in proper chronological order
      r = r.sort_index()
      
      return r
      
   def isuptodate(self, deep=True):
      freq_secs = tsUnitToSeconds(self.freq)
      
      stale = set(self.notuptodate())
      return (len(stale) == 0)
   
   def freshness(self) -> Iterator[str]:
      freq_secs = tsUnitToSeconds(self.freq)
      corrupted = []
      
      for name in self.keys():
         latest_date: pd.Timestamp = self[name].index.max()
         if isinstance(latest_date, int):
            corrupted.append(name)
            continue

         age: timedelta = (datetime.now() - latest_date.to_pydatetime())
         staleness = (age.total_seconds() / freq_secs)
         freshness = 1.0 - staleness
         
         yield name, freshness

      for c in corrupted:
         self.kc.delete(c)
         del self.data[c]
      
      self._refresh_time_range()
      
   def notuptodate(self) -> Iterator[str]:
      freq_secs = tsUnitToSeconds(self.freq)
      corrupted = []
      for name in self.keys():
         if not self.kc.has_cache_for(name):
            yield name
         latest_date:pd.Timestamp = self[name].index.max()
         if isinstance(latest_date, int):
            corrupted.append(name)
            continue
         
         age:timedelta = (datetime.now() - latest_date.to_pydatetime())
         if age.total_seconds() >= freq_secs:
            yield name
      
      for c in corrupted:
         self.kc.delete(c)
         del self.data[c]
      self._refresh_time_range()
   
   def update(self, force=False, test=False):
      if self.isuptodate() and not (force or test):
         return None

      unupdated_cache = self.data
      updated_cache = self.data if test else self.kc.update_cache(cache=unupdated_cache, symbols=tuple(self.notuptodate()))
      
      from termcolor import colored
      
      for sym,df in updated_cache.items():
         print('[========', colored(sym, 'blue', attrs=['bold', 'underline']), '========]')
         print(df.dtypes)
         print(df.index.dtype)
         
      self._set_data(updated_cache)
      
      # import krakendb as kdb
      # kdb.upsert_book(updated_cache)
      
      
   