from collections.abc import MutableMapping
from functools import *
from cytoolz import *
from fn import _, F

import inspect

from inspect import signature, Signature, getsource, getsourcefile, getsourcelines, isfunction, isclass, Parameter
from inspect import Parameter as P
from inspect import Signature as S
from numpy import typename, isscalar
from pandas import Series, DataFrame

from typing import *
import numpy as np
import pandas as pd
import torch

PK = (P.POSITIONAL_ONLY, P.KEYWORD_ONLY, P.POSITIONAL_OR_KEYWORD)

CT = ('torch', 'numpy')
def tmap(f, it):
   if isinstance(it, (dict, MutableMapping)):
      return valmap(f, it, type(it))
   return type(it)(map(f, it))

def _convert_to_tensor(x: Any)->Tensor:
   from torch import tensor, from_numpy, as_tensor, asarray, Tensor, is_tensor
   
   if is_tensor(x): 
      return x
   
   elif isscalar(x):
      return tensor(x)
   
   elif isinstance(x, list):
      return tensor(x)
   
   elif isinstance(x, (Mapping, Sequence)):
      return tmap(_convert_to_tensor, x)
   
   elif isinstance(x, ndarray):
      return from_numpy(x)
   
   elif isinstance(x, DataFrame):
      return from_numpy(x.to_numpy())
   
   else:
      t, tn = type(x), typename(x)
      raise TypeError(f'Failed to cast {tn} to Tensor.. because you suck')

def _convert_to_numpy(x: Any)->Union[ndarray, bool, float, int]:
   if x is None or isinstance(x, ndarray) or isscalar(x):
      return x
   elif isinstance(x, list):
      if 'ndarray' in tmap(typename, x):
         return asanyarray(x)
      return asarray(x)
   elif isinstance(x, (Sequence, Mapping)):
      return tmap(_convert_to_numpy, x)
   elif 
   
_CFM_ = {}

def _cast_():
   pass

def _pass_thru(argument:Any)->Any:
   return argument

def autocast(*args, **params):
   _to = params.pop('to', None)
   if _to is None:
      _to = 'torch'
   
   _convert = _pass_thru
   
   def autocasted(func):
      assert isfunction(func), TypeError(f'Cannot @autocast {typename(func)}')
      sig = signature(func)
      for name, p in sig.parameters.items():
         assert p.kind in PK
         lookup_type:int = PK.index(p)
   
   if len(params) == 0 and len(args) == 1:
      
      pass
   
   raise NotImplementedError("Not implemented completely")

import numba as nb
from ptmdl.ops import torch_vectorize

_VEC_BACKENDS = {
   'numpy': np.vectorize,
   'numba': nb.vectorize,
   'torch': torch_vectorize
}

def _vectorize_infer_backend_from_call_signature(*args, **kwargs):
   types = list(map(typename, chain(list(args), kwargs.values())))
   for t in types:
      if t == 'ndarray':
         return 'numpy'
      elif t == 'Tensor':
         return 'torch'
      continue
   raise TypeError('You are a complete and total Chad, dewd')      

def autovectorize(func):
   _v2be = {}
   
   @wraps(func)
   def jit_vectorized(*args, **kwargs):
      _be = _vectorize_infer_backend_from_call_signature(*args, **kwargs)
      if _be in _v2be:
         return _v2be[_be](*args, **kwargs)
      else:
         _f = _v2be[_be] = _VEC_BACKENDS[_be](func)
         return _f(*args, **kwargs)
      
   return jit_vectorized