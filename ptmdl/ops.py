import numpy as np
from cytoolz import *
from pathlib import Path
from torch.nn import *
from torch.jit import script, freeze, optimize_for_inference
from torch.nn import Module
from torch.autograd import Variable
from torch import Tensor, tensor, asarray
import torch.nn as nn
import torch
from typing import *
import os
import sys
import math
import random
import re
from tqdm import tqdm
import pandas as pd
import ta
from sklearn.preprocessing import MinMaxScaler, minmax_scale

from numpy import ndarray, unpackbits, packbits

P = os.path

def after(s, pref):
   if s.startswith(pref):
      return s[len(pref):]
   return s

def before(s, suff):
   if s.endswith(suff):
      return s[:-len(suff)]
   return s

def torch_vectorize(f, inplace=False, end_dim=None):
   #TODO: implement this such that it can be jitted by pytorch
   def wrapper(tensor:Tensor, **params) -> Tensor:
      x = tensor if inplace else tensor.clone()
      xshape = x.shape
      
      view = x.flatten() if end_dim is None else x.flatten(end_dim=end_dim)

      y = torch.reshape(f(view, **params), xshape)

      return y

   return wrapper

def binarized_size(a:ndarray, end_dim=-1, size:int=32)->Tuple[int]:
   if isinstance(a, Tensor):
      a = a.numpy()
   
   bytes_per_item:int = size
   bits_per_byte:int = 8
   
   if a.ndim == 1:
      return len(a) * bytes_per_item * bits_per_byte
   
   last = tensor(a.shape[end_dim:]).prod().item() * bytes_per_item * bits_per_byte
   
   return (*tuple(a.shape[:end_dim]), last)

def binarized_spec(shape:Tuple[int], end_dim=-1)->Tuple[int]:
   bytes_per_item:int = 8
   bits_per_byte:int = 8
   last = tensor(shape[end_dim:]).prod().item() * bytes_per_item * bits_per_byte
   return (*tuple(shape[:end_dim]), last)

def float2bytes(a: ndarray, size=32):
   ndim = a.ndim
   a = a.astype(f'float{size}')
   shape = a.shape
   a = a.flatten()
   buffer = a.tobytes()
   r:ndarray = np.frombuffer(buffer, dtype='uint8')
   if ndim > 1:
      rshape = (*shape[:-1], shape[-1] * 8)
      r = r.reshape(*rshape)
   return r

def bytes2float(a:ndarray, size=32):
   shape = a.shape
   a = a.flatten()
   buffer = a.tobytes()
   r = np.frombuffer(buffer, dtype=f'float{size}')
   return r

torch.float

def binarize_float(a:ndarray, size=None):
   shape, ndim = a.shape, a.ndim
   dtype = str(a.dtype)
   
   
   if dtype.startswith('float'):
      tn = after(dtype, 'float')
      _size = 32 if tn == '' else int(tn)
      if size is None: 
         size = _size
      elif size != _size:
         a = a.astype(f'float{size}')
   else:
      raise TypeError(f'array of floating-point values expected, got (dtype={a.dtype})')
   
   assert size in (16, 32, 64, 128)

   a = a.flatten()
   nels = binarized_size(a, size=size)
   b = float2bytes(a, size=size)
   
   return unpackbits(b, axis=0)

def binarize(x, size=32):
   if isinstance(x, Tensor):
      x = x.numpy()
   elif isinstance(x, ndarray):
      x:ndarray = x
   else:
      return binarize(tensor(x, dtype=torch.float32).numpy(), size=size)
   assert isinstance(x, ndarray)
   
   t = str(x.dtype)
   d:Tuple[int] = tuple(x.shape)
   r:Optional[ndarray] = None
   
   if t.startswith('float'):
      r = binarize_float(x, size=size)
   
   else:
      raise NotImplementedError(f'binarize(x: {t})')
   
   assert r is not None
   
   d = list(d)
   d[-1] = (d[-1] * size)
   r = r.reshape(*d)
   return r

def unbinarize(b:ndarray, dtype='float32', size=32):
   if b.dtype.name != 'uint8':
      b = b.astype('uint8')
   
   flat_b = b.flatten()
   
   d = list(b.shape)
   d[-1] = (d[-1] // size)
   
   flat_a = packbits(flat_b, axis=0)
   flat_a = bytes2float(flat_a)
   
   a = flat_a.reshape(*d)
   
   return a


assert len(binarize([math.pi], size=16)) == 16
assert len(binarize([math.pi], size=32)) == 32
assert len(binarize([math.pi], size=64)) == 64

_dtypes = ["float","float32","float64","double","complex64","cfloat","complex128","cdouble","float16","half","bfloat16","uint8","int8","int16","short","int32","int","int64","long","bool"]
def torchtype(t: Any):
   tnm = {name:getattr(torch, name, None) for name in _dtypes}
   if isinstance(t, np.dtype):
      t = t.name
   elif isinstance(t, torch.dtype):
      return t
   t = str(t)
   return tnm[t]