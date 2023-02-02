
from enum import Enum
import sys, os, shutil
import krakenex
from pykrakenapi import KrakenAPI
from keychain import getkeys
from functools import lru_cache, cached_property, wraps, partial
import functools
import numpy as np
import pandas as pd
#import modin.pandas as pd
from datetime import datetime, timedelta
from typing import *
from cachetools import cached, LRUCache, TTLCache, cachedmethod
import pickle, json
from ds import t
from ds.model.spec import Hyperparameters

from glob import glob
# from gen_forecasts import creation_date
from cytoolz import *
from tools import maxby, have_internet, closure, once
from ds.forecaster import NNForecaster
import atexit
P = os.path
from operator import attrgetter, methodcaller
from time import time, time_ns, sleep, monotonic
from tools import getcolsmatching
from fn import F, _
# import fn
from ratelimit import limits, sleep_and_retry
import shelve
kcache = shelve.open('kcache', writeback=True)

def _onclose():
   print('_onclose')
   kcache.sync()
   kcache.close()
atexit.register(_onclose)

def loadparams(path):
   with open(path, 'rb') as f:
      return pickle.load(f)
   
tmpcache = TTLCache(-1, 211000)

from tools import *
import enum
from cytoolz import *
#1, 30, 60, 240, 1440, 10080, 21600

class TimeInterval(Enum):
   Minute = 1
   HalfHour = 30
   Hour = 60
   SixHours = 240
   Day = 1440
   Week = 10080
   Month = 21600
   
   def __str__(self):
      return _str2TI.get(self)
      
   @staticmethod
   def from_str(s):
      return {v:k for k,v in _str2TI.items()}.get(s)
   
   @staticmethod
   def from_int(n: int):
      return TimeInterval(n)

_str2TI = {
   TimeInterval.Minute: '1min',
   TimeInterval.HalfHour: '30min',
   TimeInterval.Hour: '1H',
   TimeInterval.SixHours: '6H',
   TimeInterval.Day: '1D',
   TimeInterval.Week: '1W',
   TimeInterval.Month: '1M'
}
   
class FreqSensitiveFunc:
   def __init__(self, func):
      self.func = func
      self.last_call_time = None
      
   def since_last_call(self):
      return (time() - self.last_call_time) if self.last_call_time is not None else None
   
   def __call__(self, *args, **kwargs):
      try: 
         ret = self.func(*args, **kwargs)
         self.last_call_time = time()
         return ret
      except Exception as e:
         self.last_call_time = time()
         raise e
      
   def rateLimited(self, onceper:float, *args, **kwargs):
      if self.last_call_time is None:
         return self(*args, **kwargs)
      
      if self.since_last_call() >= onceper:
         return self(*args, **kwargs)
      else:
         delay = onceper - self.since_last_call()
         sleep(delay)
         return self(*args, **kwargs)
      
   def withRateLimit(self, limit:float=1.0):
      return partial(self.rateLimited, limit)

def freq_sensitive(func):
   wfunc = sleep_and_retry(limits(calls=1, period=2)(func))
   return wfunc
   
# @once
# @memoize
def krakenClient():
   from keychain import getkeys
   import krakenex
   from pykrakenapi import KrakenAPI
   
   keys = getkeys()
   api = krakenex.API(key=keys['kraken'][0], secret=keys['kraken'][1])
   k = KrakenAPI(api)
   
   return k

class ExchangeError(Exception):
   pass

class InvalidArgumentError(ExchangeError):
   def __init__(self, argument:List[str]=[], *agrs, **kwargs):
      super().__init__((argument, *args), **kwargs)
      
@closure
def apiException():
   import re
   
   table = {
      'EGeneral:Invalid arguments': (':(.+)$', lambda m: InvalidArgumentError([m.group(1)]))
   }
   table:Dict[str, Tuple[re.Pattern, Any]] = valmap(lambda x: (re.compile(x[0]), *x) if isinstance(x[0], str) else x, table)
   
   def convertException(apiError:Exception, defaultCls=ExchangeError):
      evals = apiError.args
      if len(evals) >= 1 and isinstance(evals[0], list) and isinstance(evals[0][0], str):
         api_err_msg:str = evals[0][0]
         for prefix, (pat, extract) in table.items():
            if api_err_msg.startswith(prefix):
               _,api_err_msg = api_err_msg.split(prefix, 1)
               m = pat.match(api_err_msg)
               if m is not None:
                  return extract(m)
      
      return apiError
   return convertException
               

def apiCall(private=False, method=None, api=None, **kwargs):
   if api is None:
      api = krakenClient()
   query = getattr(api.api, 'query_' + ('public' if not private else 'private'))
   assert method is not None, 'query method must be specified'
   res = query(method, data=kwargs)
   if len(res['error']) > 0:
      raise ExchangeError(res['error'])
   
   return res['result']

def privateCall(method=None, api=None, **kwargs):
   return apiCall(api=api, private=True, method=method, **kwargs)

def withdrawInfo(asset:str, key:str, amount:float):
   from engine.utils import krakenSymbol

   asset = krakenSymbol(asset)
   
   wd_inf = privateCall('WithdrawInfo', key=key, asset=asset, amount=amount)
   
   print(wd_inf)
   
   return wd_inf

def makeWithdrawal(asset:str, key:str, amount:float):
   from engine.utils import krakenSymbol

   asset = krakenSymbol(asset)
   wd_res = privateCall(method='Withdraw', asset=asset, key=key, amount=amount)
   
   return wd_res

def _slcmeth(self):
   return (time() - self.last_call_time) if self.last_call_time is not None else None

def merge_ts(a:pd.DataFrame, b:pd.DataFrame):
   a_times = set(a.index)
   b_times = set(b.index)
   
   uniqueToA = a_times - b_times
   uniqueToB = b_times - a_times
   common = (a_times & b_times)
   
   chunks = [
      a.loc[uniqueToA],
      a.loc[common],
      b.loc[uniqueToB]
   ]
   
   result = pd.concat(chunks, verify_integrity=True).sort_index()
   print(result)
   
   return result

def prep_remote_df(arg:Tuple[pd.DataFrame,Any]):
   print(arg)
   doc,latest = arg
   
   return (doc.sort_index(), latest)

def isuptodate(doc: pd.DataFrame):
   latest:pd.Timestamp = doc.index.max().to_pydatetime()
   actual_current = datetime.now().date()
   return latest == actual_current

def savedf(df:pd.DataFrame, path):
   try:
      df = df.reset_index(drop=False)
   except:
      df = df.reset_index(drop=True)
   df.to_feather(f'{path}.feather')

def ensureKcDirTree():
   base = './kraken_cache'
   paths = [base, P.join(base, 'historical')]
   for i in (1, 30, 60, 240, 1440, 10080, 21600):
      paths.append(P.join(base, 'historical', str(i)))
   for p in paths:
      if not P.exists(p):
         os.mkdir(p)
         
def pair_filter(pair):
   l, r = pair.wsname.split('/')
   return ('USD' not in l) and (r == 'USD')

class KrakenCache:
   def __init__(self, interval=TimeInterval.Day):
      self.interval = interval
      self.api = krakenClient()
      ensureKcDirTree()
      
   @cached_property
   def basedir(self):
      return f'./kraken_cache/historical/{self.interval.value}'
      
   def get_pairs(self, k:KrakenAPI=None):
      if k is None: k = self.api
      
      from engine.utils import age_of_path

      if P.exists('kraken_cache/tradable_pairs.csv'):
         if age_of_path('kraken_cache/tradable_pairs.csv').days >= 7 and have_internet():
            os.remove('kraken_cache/tradable_pairs.csv')
            return self.get_pairs(k)

         pairs = pd.read_csv('kraken_cache/tradable_pairs.csv')
         if 'Unnamed: 0' in pairs.columns:
            pairs.rename(columns={'Unnamed: 0': 'name'})
         pairs.set_index('name', inplace=True)
         cols = pairs.columns.tolist()
         pairs['name'] = pairs.index
         pairs = pairs[['name']+cols]
         return pairs
      
      else:
         pairs = k.get_tradable_asset_pairs()
         pairs.rename_axis(index='name', inplace=True)
         pairs.to_csv('kraken_cache/tradable_pairs.csv')
         return pairs
   
   def save_cache(self, cache:Dict[str, pd.DataFrame], format='feather'):
      base = P.join(self.basedir, '{name}.'+format)
      encode = attrgetter(f'to_{format}')
      
      for k, d in cache.items():
         p = base.format(name=k)
         try: 
            d = d.reset_index(drop=False)
         except ValueError: 
            d = d.reset_index(drop=True)
         encode(d)(p)
         
   def delete(self, name:str, format='*'):
      path = P.join(self.basedir, '{name}.'+format)
      import glob, shutil, os
      fnames = glob.glob(base)
      for p in fnames:
         os.remove(p)
         
   def load_cache_as(self, format='feather'):
      import glob
      
      glob_pattern = P.join(self.basedir, f'*.{format}')
      paths = glob.glob(glob_pattern)
      names = [P.splitext(P.basename(x))[0] for x in paths]
      names = [name for name in names if not name.endswith('USD')]
      decode = getattr(pd, f'read_{format}')
      cache = {}
      
      if len(paths) == 0:
         print(f'search of "{glob_pattern}" yielded empty list')
         return None
      
      for name, path in zip(names, paths):
         df:pd.DataFrame = decode(path)
         dtimedupes = [k for k in getcolsmatching('dtime*', df) if k != 'dtime']
         df = df.drop(columns=dtimedupes)
         df['dtime'] = pd.to_datetime(df['dtime'])
         df = df.set_index('dtime', drop=False)
         
         assert str(df['dtime'].dtype) == 'datetime64[ns]'
         
         cache[name] = df
      
      return cache
      
   def load_cache(self):
      cache = self.load_cache_as(format='feather')
      if cache is None:
         cache = self.load_cache_as(format='csv')
         if cache is None:
            cache = self.build_cache()
         self.save_cache(cache, format='feather')
      
      return cache
      
   def pairssel(self, verbose=True):
      from time import sleep
      import tqdm
      pairs = self.get_pairs(self.api)
      pairs = pairs.where(pairs.quote.str.endswith('USD')).dropna()
      exclude = ['ZAUD', 'ZJPY', 'ZGBP']
      sel = [(name.replace('USD', ''), pair) for name, pair in pairs.iterrows() if name.endswith('USD') and pair.base not in exclude]
      sel = tqdm.tqdm(iterable=sel)
      for name, pair in sel:
         sel.set_description(name)
         yield name, pair
      
   def clear_cache(self):
      from glob import glob
      paths = glob(P.join(self.basedir, '*.*'))
      for p in paths:
         os.remove(p)
         
   def has_cache_for(self, symbol:str, format='feather'):
      return P.exists(P.join(self.basedir, f'{symbol}.{format}'))
      
   def build_cache(self, k:KrakenAPI=None, verbose=True, autosave=True, format='feather'):
      from time import sleep
      import tqdm
      
      if k is None: 
         k = self.api
      
      base = P.join(self.basedir, '{name}.'+format)
      
      def encode(df, name):
         f = getattr(df, f'to_{format}')
         f(name)
      
      pairs = self.get_pairs(k)
      pairs = pairs.where(pairs.quote.str.endswith('USD')).dropna()
      exclude = ['ZAUD', 'ZJPY', 'ZGBP']
      cache = {}
      
      sel = [(name, pair) for name, pair in pairs.iterrows() if name.endswith('USD') and pair.base not in exclude and pair_filter(pair)]
      real_ids = [(name, pair.wsname.split('/')) for (name, pair) in sel]
      # for pid, (l, r) in real_ids:
      #    print(f'{l} ->  {r}')
      # input()
      duration_estimate = (1.0 * len(sel))/60.0
      
      print(f'building cache; this will take ~{duration_estimate}mins')
      
      sel = tqdm.tqdm(iterable=sel)
      fetch = partial(freq_sensitive(k.get_ohlc_data), interval=self.interval.value)
      
      for p_id, (token, usd) in real_ids:
         sel.set_description(token)
         
         df, _ = fetch(p_id)
         df:pd.DataFrame = df.sort_index()
         
         encode(
            df.reset_index(drop=False), 
            base.format(name=token)
         )
         cache[token] = df
         
         # print(f'[=================== {token}/{quote} ===================')
         # print(df)
         
      return cache

   def update_cache(self, k:KrakenAPI=None, cache=None, symbols=None):
      from time import sleep
      
      def inlineSave(d:pd.DataFrame, base:str):
         savedf(d, P.join(self.basedir, base))
      
      if k is None: k = krakenClient()
      if cache is None: 
         cache = self.load_cache()
      
      res = {}
      pairs = self.pairssel(k)
      
      fetch = F(freq_sensitive(partial(k.get_ohlc_data, interval=self.interval.value))) >> prep_remote_df
      
      for name, pair in pairs:
         locl = cache.get(pair.base, None)
         pairId = pair.name
         base, quote = pair.wsname.split('/')
         
         if symbols is not None and pair.base not in symbols and pair.base != '1INCH':
            if pair.base in cache:
               res[pair.base] = cache[pair.base]
            continue
         
         if locl is None:
            # print('fetching %s' % name)
            locl,_ = fetch(pairId)
            res[pair.base] = locl
            inlineSave(locl, pair.base)
            
            continue
         
         elif (base in symbols if symbols is not None else False) or not isuptodate(locl):
            remote,_ = fetch(pairId)
            try: 
               remote = remote.set_index('dtime', drop=False).sort_index()
            except Exception as e: 
               print(e)
               remote = remote.sort_index()
            
            merged = merge_ts(locl, remote).drop(columns=['dtime']).sort_index()
            res[pair.base] = merged
            inlineSave(merged, pair.base)
            
         else:
            res[pair.base] = cache[pair.base]
      
      self.save_cache(res)
      
      return res
   
from ds.datasets import register, LDExt, load_dataset

def optts(func):
   # from engine.utils import nonkrakenSymbol
   def wfunc(s: str):
      nks = func(s)
      if nks is not None:
         # print(f'transformed "{s}" = "{nks}"')
         return nks
      # print(f'"{s}"')
      return s
   return wfunc

class KrakenLDExt(LDExt):
   def __init__(self):
      self.k = KrakenCache()
      self.cache = None
      
   def exists(self, name, **kwargs):
      from engine.utils import krakenSymbol
      name = optts(krakenSymbol)(name.upper())
      return self.k.has_cache_for(name.upper())
   
   def get(self, name, **kwargs):
      from engine.utils import nonkrakenSymbol, krakenSymbol
      name = optts(krakenSymbol)(name.upper())
      if self.cache is None:
         self.cache = self.k.load_cache()
      r = self.cache.get(name.upper(), None)
      if r is not None:
         r = r.sort_index()
      return r
   
   def symbols(self):
      from engine.utils import nonkrakenSymbol, krakenSymbol
      if self.cache is None:
         self.cache = self.k.load_cache()
      return map(optts(nonkrakenSymbol), self.cache.keys())
   
   def cache_name(self, name, **kwargs):
      return f'kraken_{name}'
   
register('kraken', KrakenLDExt)

if __name__ == '__main__':
   print(load_dataset('kraken/zec'))