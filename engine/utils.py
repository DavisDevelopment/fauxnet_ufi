import socket
from pymitter import EventEmitter
from datetime import datetime, timedelta
from typing import *
import kraken
from pprint import pprint
from math import floor, ceil
import numpy as np
import re
import pandas as pd
#import modin.pandas as pd
from numba import jit, njit, float32, float64
from numba.experimental import jitclass
from functools import *
import toolz as tlz
from cytoolz import *
from operator import attrgetter, methodcaller
from cytoolz import dicttoolz as dicts
from cytoolz import itertoolz as iters
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

import ds.utils.bcsv as bcsv
from ds.utils.book import Book, TsBook, FlatBook
from ds.forecaster import NNForecaster
from ds.model.ctrl import MdlCtrl
from ds.model_train import descale
from ds.gtools import Struct
import pickle
import os, sys
P = os.path
dt, td = datetime, timedelta
kr = kraken
from fn import _
import fn
import shelve

@curry
def n_largest(n: int, bk: Book):
   items = list(bk.items())
   items = list(sorted(items, key=lambda x: len(x[1]), reverse=True))
   items = [(name, len(doc)) for name, doc in items]

   print(items[:100])
   for k, v in items[101:]:
      del bk[k]
   return bk


@curry
def has_range(start, end, bk: TsBook):
   flagged = []
   for name, doc in bk.items():
      min, max = doc.index.min(), doc.index.max()
      if start >= min and end <= max:
         continue
      else:
         flagged.append(name)
   for n in flagged:
      del bk[n]
   bk._refresh_time_range()
   return bk


def cleansym(sym):
   sym = sym.upper()
   if sym.endswith('USD'):
       sym = sym.replace('USD', '')
   return sym


@curry
def indexOf(value, snapping:bool, series:pd.Series):
   try:
      if snapping:
         todayIdx = series.get_loc(value, method='nearest')
      else:
         todayIdx = series.get_loc(value)
   except KeyError as ke:
      print(ke)
      todayIdx = -1
   # except Inde
   return todayIdx


def notnone(v):
   return v is not None


def price(today: pd.Series):
   return (today['open'] + today['close'])/2.0


def addrow(df: pd.DataFrame, row):
   df.loc[len(df.index)] = row

def format_balance(bal: Dict[str, float], invert=False):
   bal = keymap(partial(ksymbol, inverse=True), keymap(methodcaller('upper'), bal))
   
   #TODO rename the balance-keys where appropriate to allow for naming consistency
   tbl = _rename_table
   if invert:
      tbl = {v: k for k, v in tbl.items()}
   bal = keymap(lambda k: tbl.get(k, k), bal)
   
   return bal

def age_of_path(path:str)->timedelta:
   mtime = P.getmtime(path)
   age = (datetime.now() - datetime.fromtimestamp(mtime))
   return age

def trash(*paths):
   from bash import bash
   import shlex
   qpaths = list(map(compose(shlex.quote, P.abspath), paths))
   
   for name in qpaths:
      bash('trash %s' % name, stderr=0, stdout=0, sync=True)
   
# @jit
def npdate2date(x):
   return datetime.utcfromtimestamp(dt64.astype(int) * 1e-9)

# @jit(cache=True)
def _reducer(a:Optional[float64[:]], b:Optional[float64[:]], method='avg', bias=0.5):
   if a is None:
      return b
   elif b is None:
      return a
   else:
      a, b = a.T, b.T
      r = np.zeros(a.shape)
      channels = a.shape[0]
      
      for i in range(channels):
         larg:float64 = a[i][0]
         rarg:float64 = b[i][0]
         
         if method == 'avg':
            r[i] = (a[i] + b[i])/2.0
         
         elif method == 'max':
            r[i] = max(larg, rarg)
         
         elif method == 'min':
            r[i] = min(larg, rarg)
            
         elif method == 'lerp':
            r[i] = lerp(larg, rarg, bias)
         
         else:
            raise Exception('Unsupported reduction method')
      return r.T
   
# def flatten_tuple
def flatten_tuple(data):
    if isinstance(data, tuple):
        if len(data) == 0:
            return ()
        else:
            return flatten_tuple(data[0]) + flatten_tuple(data[1:])
    else:
        return (data,)

def dzip(a, b, acc=dict):
   _acc = acc()
   keys = list(set(a.keys()) | set(b.keys()))
   
   for k in keys:
      l, r = a.get(k, None), b.get(k, None)
      if l is None: 
         _acc[k] = (r,)
         continue
      elif r is None:
         _acc[k] = (l,)
         continue
      else:
         agg = tuple()
         if isinstance(l, tuple):
            agg = (agg, *l)
         else:
            agg = (l,)
         if isinstance(r, tuple):
            agg = (*agg, *r)
         else:
            agg = (*agg, r)
         
         _acc[k] = flatten_tuple(agg)
   
   return _acc
_super = dzip
def dzip(*dicts):
   if len(dicts) < 2:
      print(dicts)
      raise ValueError("not enough arguments")
   return reduce(_super, dicts)
# dzip

def dunitize(d:Dict[str, Iterable[Any]], reducer=None):
   red = lambda x: reduce(reducer, x)
   return valmap(red, d)

def dpck(pairs:Iterator[Tuple[Any, Any]], factory=dict):
   askw = {l:r for (l,r) in pairs}
   return factory(**askw)

def dtrange(start, end, step):
   cur = start
   # yield cur
   while cur <= end:
      yield cur
      cur += step

from tools import profile, noprofile, inspect, diffdf

dunder = re.compile('$__(.+)__^')
def notmagic(name:str):
   return dunder.match(name) is None

def for_all_methods(decorator):
   def decorate(cls):
      for attr in cls.__dict__:  # there's propably a better way to do this
         if callable(getattr(cls, attr)):
            setattr(cls, attr, decorator(getattr(cls, attr)))
      return cls
   return decorate

def profileclass(cls:type):
   if noprofile: return cls
   clsname = cls.__name__
   # members = inspect.get
   members = inspect.getmembers(cls, inspect.ismethod)
   print(members)
   # exit()
   
   newmembers = {}

   for name, mem in members:
      if notmagic(name):
         mem = profile(mem)
      # newmembers[name] = mem
      setattr(cls, name, mem)
   # wrapped = type(clsname, (cls,), newmembers)
   # return wrapped
   return cls

# profile.__
# def have_internet():
#    try:
#       import httplib
#    except:
#       import http.client as httplib
   
#    conn = httplib.HTTPConnection("www.google.com", timeout=5)
#    try:
#       # conn.
#       conn.request("HEAD", "/")
#       conn.close()
#       return True
#    except:
#       conn.close()
#       return False

@njit
def lerp(a:float, b:float, t:float) -> float:
    """Linear interpolate on the scale given by a to b, using t as the point on that scale.
    Examples
    --------
        50 == lerp(0, 100, 0.5)
        4.2 == lerp(1, 5, 0.8)
    """
    return (1 - t) * a + t * b

@njit
def inv_lerp(a: float, b: float, v: float) -> float:
    """Inverse Linar Interpolation, get the fraction between a and b on which v resides.
    Examples
    --------
        0.5 == inv_lerp(0, 100, 50)
        0.8 == inv_lerp(1, 5, 4.2)
    """
    return (v - a) / (b - a)
 
def npdropna(a):
   if a.ndim == 2:
      b = np.isnan(a[..., 1])|np.isnan(a[..., 0])
      return a[~b]
   else:
      return a[~np.isnan(a)]

def monasterisk(func):
   @wraps(func)
   def wrapper(arg):
      return func(*arg)
   return wrapper

def identity(x):
   return x

def tsUnitToSeconds(ts_unit:str)->int:
   do = pd.DateOffset(ts_unit) if not isinstance(ts_unit, pd.DateOffset) else ts_unit
   return (do.n * (do.nanos * 1e-9))

def once(func):
   s = [False, None]
   def _wf(*a,**kw):
      if not s[0]:
         s[1] = func(*a, **kw)
         s[0] = True
      return s[1]
   return _wf

@once
def once(func):
   s = [False, None]

   def _wf(*a, **kw):
      if not s[0]:
         s[1] = func(*a, **kw)
         s[0] = True
      return s[1]
   return _wf

def flipdict(d):
   return {v:k for k,v in d.items()}

# _rename_table = dict(
#     ZUSD='USD',
#     XXDG='DOGE',
#     XXBT='BTC',
#     XBT='BTC',
#     XETH='ETH'
# )

# def ksymbol(sym:str, inverse=False):
#    if not inverse:
#       tbl = flipdict(_rename_table)
#    else:
#       tbl = _rename_table
#    sym = sym.upper()
#    if sym not in tbl and f'X{sym}' in tbl:
#       return tbl[f'X{sym}']
#    return tbl.get(sym, sym)

from bidict import bidict

from collections import namedtuple

ksyminfo = namedtuple('ksyminfo', 'ksymbols,symbols,tmap,dmap')

@once
def ksym_info()->ksyminfo:
   from krakendb import ExchangePair
   ksymbols = set()
   symbols = set()
   
   # slim_map = bidict()
   tmap = bidict()
   dmap = dict()
   
   for p in ExchangePair.objects.all():
      p:ExchangePair = p
      kl, kr = p.base, p.quote
      sl, sr = p.wsname.split('/')
      if not kl in ksymbols:
         ksymbols.add(kl)
         tmap[kl] = sl
         
      if not kr in ksymbols:
         ksymbols.add(kr)
         tmap[kr] = sr
      
      if not sl in symbols:
         symbols.add(sl)
         
      if not sr in symbols:
         symbols.add(sr)
      dmap[p.wsname] = p.name
      
   return ksyminfo(ksymbols, symbols, tmap, dmap)

def krakenSymbol(sym:str):
   ks = ksym_info()
   if sym in ks.ksymbols:
      return sym
   elif sym in ks.symbols:
      return ks.tmap.inverse[sym]
   else:
      return None
      raise ValueError(f'No krakenSymbol for "{sym}"')
   
def nonkrakenSymbol(sym:str):
   ks = ksym_info()
   if sym in ks.ksymbols:
      return ks.tmap[sym]
   elif sym in ks.symbols:
      return sym
   else:
      return None
      raise ValueError(f'No nonkrakenSymbol for "{sym}"')
   
def krakenPair(lsym:str, rsym:str):
   ks = ksym_info()
   l, r = nonkrakenSymbol(lsym,), nonkrakenSymbol(rsym)
   return ks.dmap[f'{l}/{r}']