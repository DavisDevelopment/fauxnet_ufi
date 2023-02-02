import inspect
from cytoolz import *
from functools import *
from fn import F, _
import pandas as pd
#import modin.pandas as pd
import numpy as np
from numpy import ndarray
from numba import *
from typing import *

TFeature = TypeVar('TFeat')
TTarget = TypeVar('TTarget')

nojit = False

def _doc_to_ndarray(df):
   if isinstance(df, ndarray):
      return df
   elif isinstance(df, pd.DataFrame):
      return df.values
   else:
      raise ValueError(f'Unaccepted data type {type(df)}')
   
def nor(a, b):
   return a if a is not None else b

class ValueExtractor:
   __slots__ = ['head', 'body', 'tail']
   
   def __init__(self, main, head=None, tail=None):
      self.head = F(_doc_to_ndarray)
      self.body = main
      self.tail = F(lambda x: x)
      if head is not None:
         self.head = head
      if tail is not None:
         self.tail = tail
         
   def extend(self, head=None, body=None, tail=None, backref=False):
      if not backref:
         h = nor(head, self.head)
         b = nor(body, self.body)
         t = nor(tail, self.tail)
         return ValueExtractor(b, head=h, tail=t)
      else:
         raise NotImplementedError()
      
   def __call__(self, df:pd.DataFrame, *args, **kwargs):
      ndf = self.head(df)
      result = self.body(ndf, *args, **kwargs)
      result = self.tail(result)
      return result
      
def dffunc(func):
   ex = ValueExtractor(func)
   
   return ex

@dffunc
@jit(cache=True)
def feature_seqs(df: float32[:, :], n_steps:i4=7, idx:i4[:]=None)->Tuple[Tuple[i4, i4], float32[:,:,:]]:
    rows: i4 = df.shape[0]
    cols: i4 = df.shape[1]

    if idx is None:
      X: float32[:, :, :] = np.zeros((rows-n_steps, n_steps, cols))
      for i in prange(n_steps, rows):
         X[i-n_steps, :, :] = df[i-n_steps+1:i+1]
      return (n_steps, 0), X
    else:
      r = np.zeros((len(idx), n_steps, cols))
      for j in range(len(idx)):
         i = idx[j]
         r[j, :, :] = df[i-n_steps:i]
      return (0, 0), r

@dffunc
@jit(cache=True)
def target_seqs(df: float32[:, :], n_steps:i4=7, idx:i4[:]=None)->Tuple[Tuple[i4, i4], float32[:,:,:]]:
    rows: i4 = df.shape[0]
    cols: i4 = df.shape[1]
    if idx is None:
      y: float32[:, :] = np.zeros((rows-n_steps, cols))
      for i in prange(n_steps, rows-1):
         y[i-n_steps, :] = df[i+1, :]
      return (n_steps, 0), y
    else:
      r = np.zeros((len(idx), cols))
      for j in range(len(idx)):
         i = idx[j]
         r[j, ...] = df[i+1, :]
      return (0, 0), r
 
def regression(df:pd.DataFrame, feature_func=None, target_func=None, feature_columns=None, target_columns=None, *args, **kwargs):
   Xdf = df[feature_columns] if feature_columns is not None else df
   ydf = df[target_columns] if target_columns is not None else df
      
   idx = kwargs.pop('idx', None)
   idx_mode = (idx is not None)
   if idx is not None:
      npidx = _npdfidx(df, idx)
      # print(idx, npidx)
      kwargs['idx'] = npidx
      
   resdf = kwargs.pop('res', None)
   if resdf is None:
      resdf = pd.DataFrame(index=df.index, columns=['X', 'y'])
   
   resdf['X'] = None
   resdf['y'] = None
   
   if feature_func is not None:
      xskip, X = feature_func(Xdf, *args, **kwargs)
      
      dfa_kw = dict(var='X', skip=xskip, val=X)
      if idx_mode: dfa_kw['idx']= npidx
      
      df_assign(resdf, **dfa_kw)
      
   if target_func is not None:
      yskip, y = target_func(ydf, *args, **kwargs)
      # idxOfFuckup, isOk = check(X, y)
      # assert isOk, f'You fucked up at {idxOfFuckup}; You\'re probably stupid'
      
      dfa_kw = dict(var='y', skip=yskip, val=y)
      if idx_mode: dfa_kw['idx']= npidx
      
      df_assign(resdf, **dfa_kw)

   return resdf


# @njit(cache=True, parallel=True)
def check(X:np.ndarray, y:np.ndarray):
   X, y = np.asanyarray(X), np.asanyarray(y)
   assert len(X) == len(y)
   for i in range(0, len(X)):
      _X = X[i]
      _y = y[i]
      
      if _y in _X:
         loc, = np.where(np.all(_X == _y, axis=1))
         # print(loc)
         if len(loc) == 0:
            return None
         
         print(f'{_X[loc]} == {_y}')
         raise AssertionError(f"bad")
   
   return -1, True

def df_assign(df:pd.DataFrame, var='_', skip=None, val=None, idx:Iterable[int]=None):
   _size = len(df.index)
   indices = df.index[:]
   if skip == (0,0): skip = None
   if skip is not None:
      atstart, atend = skip
      if not (atend is None or atend == 0):
         indices = indices[:_size - atend]
      if not (atstart is None or atstart == 0):
         indices = indices[atstart:]
      df[var] = None
      c = df[var]
      df.loc[indices, var] = range(len(indices))
      df[var] = df[var].apply(lambda i: val[i] if i is not None else None)
   elif idx is not None:
      assert len(val) == len(idx)
      df[var] = None
      df[var].iloc[idx] = range(len(idx))
      
      df[var] = df[var].apply(lambda i: val[i] if i is not None else None)
   

@njit(cache=True, parallel=True)
def delta(x: float64[:]):
   y = np.copy(x)
   for i in prange(1, len(x)):
      y[i] = (x[i] - x[i - 1])
   return y

@njit(cache=True, parallel=True)
def dedelta(x: float64[:]):
   y = np.copy(x)
   for i in prange(1, len(x)):
      y[i] = (x[i-1] + x[i])
   return y

@njit(cache=True, parallel=True)
def delta2d(arr:ndarray):
   for i in range(arr.shape[0]):
      row = arr[i, :]
      arr[i, :] = delta(row)
   return arr

@dffunc
def feature_seqs2d(df:float32[:, :], n_steps:i4):
   df = df.T
   df = delta2d(df)
   
   rows:i4 = df.shape[1]
   cols:i4 = df.shape[0]
   res = np.zeros((*df.shape, 2))
   
   for i in range(n_steps, rows):
      sign = (df[:, i] > 0).astype('int')
      magn = np.abs(df[:, i])
      row = np.vstack((sign, magn)).T
      res[:, i, :] = row

   return res
      
def _attempt(func, handle, default=None):
   @wraps(func)
   def _wrapper(*args, **kwargs):
      try:
         return func(*args, **kwargs)
      except Exception as e:
         if isinstance(e, handle):
            print('Exception', type(e), e)
            return default
         else:
            raise e
   return _wrapper
      
def isiterable(x):
   try:
      i = iter(x)
      return True
   except TypeError as e:
      return False
   return isinstance(x, Iterable)
      
def _expand(loc):
   if loc is None: return loc
   elif isinstance(loc, int): return loc
   elif isinstance(loc, slice):
      loc:slice = loc
      rloc = range(loc.start, loc.stop)
      return [i for i in rloc]
   else: return loc
   
def _flat(locs):
   r = []
   for x in locs:
      if isiterable(x):
         for v in x: 
            r.append(v)
      else:
         r.append(x)
   return r
   
def _npdfidx(df:pd.DataFrame, idx:List[Any], method='nearest', tol=None)->List[Optional[int]]:
   locator = F(_attempt((lambda i: df.index.get_loc(i, method=method, tolerance=tol)), (KeyError, IndexError))) >> _expand
   task = F() >> (map, locator) >> _flat
   ret = task(idx)
   return ret


def ckw(hp, feature_func, target_func, **kwargs):
   Xargs = inspect.signature(feature_func).parameters
   yargs = inspect.signature(target_func).parameters
   hpattrs = dir(hp)
   kw = {}
   for k in Xargs.keys():
      if k in hpattrs:
         kw[k] = getattr(hp, k)
   # ykw =
   for k in yargs.keys():
      if k in hpattrs:
         kw[k] = getattr(hp, k)
   kw.update(kwargs)
   return kw

from ds.model.spec import Hyperparameters
from engine.utils import profile
class Extractor:
   def __init__(self, params:Hyperparameters, features:ValueExtractor, targets:ValueExtractor):
      self.params = params
      
      self.head = F()
      self.tail = F()
      self.features = features
      self.targets = targets
      # self.preprocessing_steps = []
      
      self.__post_init__()
      
   def __post_init__(self):
      pass
   
   def extend(self, head=None, tail=None):
      # h = nor(head, self.head)
      # t = nor(tail, self.tail)
      sub = Extractor(self.params, self.features, self.targets)
      if head is not None:
         sub.head >>= head
      if tail is not None:
         sub.tail >>= tail
      return sub
      
   def add(self, func, prepend=False):
      self.head <<= func
      return func
      
   def preprocess(self, df:pd.DataFrame, copy=True):
      df = self.head(df)
      return df
      
   @profile(fid='Extractor.call')
   def __call__(self, df:pd.DataFrame, **kwargs):
      hp = self.params
      columns = list(unique(hp.feature_columns + hp.target_columns))
      df = df[columns]
      df = self.params.apply_mods(df)
      df = self.preprocess(df)
      kw = ckw(self.params, self.features.body, self.targets.body, **kwargs)
      
      rdf = regression(df, 
         feature_func=self.features, 
         target_func=self.targets, 
         feature_columns=hp.feature_columns, 
         target_columns=hp.target_columns, 
         **kw
      )
      rdf = self.tail(rdf)
      return rdf
   
   def all(self, df:pd.DataFrame):
      return self(df)
   
   def sub(self, df:pd.DataFrame, idx:float64[:]):
      assert idx is not None and len(idx) > 0
      ret = self(df, idx=idx)
      print(ret)
      return ret

def compileFromHyperparams(hp, feature_func=feature_seqs, target_func=target_seqs)->Extractor:
   _f = Extractor(hp, feature_func, target_func)

   return _f