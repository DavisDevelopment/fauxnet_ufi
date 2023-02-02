from ds.common import *
import numba
from numba import jit, njit
from functools import reduce, partial, wraps
import functools
import numpy as np


def compose(*functions):
   return functools.reduce(lambda f, g: lambda *x: f(g(*x)), functions, lambda x: x)

class Outcome:
   def __init__(self, result, is_failure=None):
      self.value = result
      if is_failure is None:
         self.status = not isinstance(self.value, Exception)
      else:
         self.status = not is_failure
   
   @property
   def is_failure(self): return not self.status
   
   @property
   def is_success(self): return self.status
   
   def get(self):
      if self.is_failure:
         raise self.value
      return self.value
   
def noraise(userfunc):
   @wraps
   def safe(*args, **kwargs):
      try:
         r = userfunc(*args, **kwargs)
         return Outcome(r, is_failure=False)
      except Exception as e:
         return Outcome(e)
   return safe

def mpmap(f, it, partialargs=None, parallelize=True):
   if partialargs is not None:
      f = partial(f, *partialargs)
   if parallelize:
      from multiprocessing import Pool
      with Pool(11) as pool:
         r = pool.map(f, it)
         pool.close()
         return r
   else:
      return list(map(f, it))
   
def once(func):
   s = [False, None]
   def _wf(*a,**kw):
      if not s[0]:
         # nonlocal result
         s[1] = func(*a, **kw)
         s[0] = True
      return s[1]
   return _wf


from tools import Struct

def ojit(f):
   f = numba.jit(f, forceobj=True)
   return f

is_verbose = '--verbose' in sys.argv or '-V' in sys.argv
def vprint(*args):
   if is_verbose:
      print(*args)
   
   
class TupleOfLists(tuple):
   def __new__(self, init):
      if isinstance(init, int):
         # self.n = init
         l = ([] for x in range(init))
         # super().__init__(l)
         return tuple.__new__(TupleOfLists, l)
         
      elif isinstance(init, (list, tuple)):
         # self.n = len(init)
         l = init
         return tuple.__new__(TupleOfLists, l)
      
      else:
         raise TypeError(f'{type(init)} object is not iterable')
         
   def append(self, *row):
      for i, v in enumerate(row):
         self[i].append(v)
         
   def map(self, func):
      return TupleOfLists(tuple(map(func, self)))