import numpy as np
from numpy import *
from numba import jit, njit, float32, float64, prange, generated_jit
from ds.gtools import ojit
from functools import lru_cache, wraps
from sklearn.base import TransformerMixin

def randomwalk(dims, step_n, step_set=[-1, 0, 1]):
   # step_set = [-1, 0, 1]
   origin = np.zeros((1, dims))
   # Simulate steps in 1D
   step_shape = (step_n, dims)
   steps = np.random.choice(a=step_set, size=step_shape)
   path = np.concatenate([origin, steps]).cumsum(0)
   start = path[:1]
   stop = path[-1:]
   return path

class _FunctionalTransform(TransformerMixin):
   def __init__(self, f, b=None):
      self._forward = f
      self._backward = b
      self._inverted = None
      
   def transform(self, *args, **kwargs):
      return self._forward(*args, **kwargs)
   
   def inverse_transform(self, *args, **kwargs):
      if self._backward is None:
         raise ValueError('Transform has no inverter')
      return self._backward(*args, **kwargs)
   
   def inverter(self, f):
      self._backward = f
      return f
   
   def __call__(self, *args, **kwargs):
      return self.transform(*args, **kwargs)
   
   def __invert__(self):
      if self._inverted is None:
         self._inverted = _FunctionalTransform(self._backward, self._forward)
      return self._inverted
   
   def __neg__(self): return self.__invert__()
   
class TChain(TransformerMixin):
   def __init__(self, a, b):
      self.a = a
      self.b = b
      
   def transform(self, x):
      x = self.a.transform(x)
      x = self.b.transform(x)
      return x
   
   def inverse_transform(self, x):
      x = self.b.inverse_transform(x)
      x = self.a.inverse_transform(x)
      return x

def Transform(func):
   t = _FunctionalTransform(func)
   return t

@Transform
@njit(cache=True, parallel=True)
def divdelta(x: float64[:]):
   y = np.copy(x)
   for i in prange(1, len(y)):
      prev = x[i-1]
      cur = x[i]
      if prev == 0:
         print(i, cur, prev)
      y[i] = cur/prev
   # nn = np.count_nonzero(np.isnan(y))
   # if nn != 0:
   #    print('divdelta generated nan values')
   return y

@divdelta.inverter
@njit(cache=True, parallel=True)
def ddprod(x: float64[:]):
   y = np.copy(x)
   for i in prange(1, len(y)):
      y[i] = y[i-1]*x[i]
   return y

@Transform 
@njit(cache=True, parallel=True)
def delta(x: float64[:]):
   y = np.copy(x)
   for i in prange(1, len(x)):
      y[i] = (x[i] - x[i - 1])
   return y

@delta.inverter
@njit(cache=True, parallel=True)
def dedelta(x: float64[:]):
   y = np.copy(x)
   for i in prange(1, len(x)):
      y[i] = (x[i-1] + x[i])
   return y


@njit(cache=True)
def shift(xs: ndarray, n: int, fv=np.nan, inplace=False):
   e = np.empty_like(xs) if not inplace else xs
   if n >= 0:
      e[:n] = fv
      e[n:] = xs[:-n]
   else:
      e[n:] = fv
      e[:n] = xs[-n:]
   return e

# @jit
def shift2(x:float64[:,:], n:int):
   x = np.roll(x, n, axis=0)
   x[n:] = np.zeros((x[n:].shape))
         
   return x