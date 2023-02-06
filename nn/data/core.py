import sys, os, random, json, pickle
import numpy as np
from numpy import ndarray, empty, zeros, asarray, asanyarray
import pandas as pd
from pandas import DataFrame, Series, Timestamp
from fn import F, _
from itertools import *
from cytoolz import *
from typing import *

DataLoadable = Union[ndarray, DataFrame]
DataLoadableIndex = Union[ndarray, pd.Index, Series]

class TwoPaneWindow:
   data:Optional[ndarray] = None
   index:Optional[ndarray] = None
   lpos:Optional[int] = None
   
   def __init__(self, lsize:int, rsize:int):
      self.lpos = None
      self.data = None
      self.index = None
      
      # self._size = (lsize, rsize)
      self.lsize = lsize
      self.rsize = rsize
      self.wsize = (lsize + rsize)
      
   def has_data(self)->bool:
      return (self.data is not None and self.lpos is not None)
   
   def __repr__(self)->str:
      r = f'TwoPaneWindow({self.lsize}, {self.rsize}'
      if self.has_data():
         r += f', {self.lpos}:{self.rpos})'
      else:
         r += r')'
      return r
   
   def is_empty(self)->bool:
      
      return not (self.lpos is not None and self.lpos < len(self.data) - self.wsize)
   
   def load(self, data:DataLoadable, index:Optional[DataLoadableIndex]=None, pos=0):
      self.lpos = pos
      if isinstance(data, ndarray):
         self.data = data
      elif isinstance(data, DataFrame):
         self.data = data.to_numpy()
         self.index = data.index.to_numpy()
      if index is not None:
         if isinstance(index, ndarray):
            self.index = index
         else:
            self.index = index.to_numpy()
      else:
         self.index = np.arange(0, len(self.data))
      return self
   
   def unload(self)->None:
      if self.data is None:
         raise Exception('Unload failed; No data loaded')
      self.data = None
      self.index = None
      self.lpos = None
   
   @property
   def rpos(self)->int:
      assert self.lpos is not None, f'TwoPaneWindow has no position property until load method is called'
      return (self.lpos + self.wsize)
   
   @property
   def pos(self)->int:
      return self.lpos + self.lsize
   
   @pos.setter
   def pos(self, v:int):
      nv = v - self.lsize
      
      if nv < 0 or nv > len(self.data)-self.wsize:
         raise IndexError(f'The pos attribute of {self} may not be less than 0 and may not exceed {len(self.data)-self.wsize}')
      
      self.lpos = nv
      
      return self.pos
   
   def seek(self, pos:int):
      if pos < 0:
         pos = (len(self.data)-self.rsize+pos)
      self.pos = pos
      
   def next(self):
      self.pos += 1
      return self
   
   def prev(self):
      self.pos -= 1
      return self
      
   def before(self)->Optional[ndarray]:
      assert self.data is not None and self.lpos is not None
      if self.lpos > 0:
         return self.data[:self.lpos]
      return None
   
   def left(self, fmtfn:Optional[Callable[[ndarray], Any]]=None):
      assert self.data is not None
      res = self.data[self.lpos:self.pos]
      assert len(res) == self.lsize, 'size mismatch: expected %d, got %d' % (self.lsize, len(res))
      if callable(fmtfn):
         res = fmtfn(res)
      return res
   
   def right(self, fmtfn:Optional[Callable[[ndarray], Any]]=None):
      assert self.data is not None
      res = self.data[self.pos+1:self.pos+1+self.rsize]
      assert len(res) == self.rsize, 'size mismatch: expected %d, got %d' % (self.rsize, len(res))
      if callable(fmtfn):
         res = fmtfn(res)
      return res
   
   def sample(self, lfn=None, rfn=None):
      l:ndarray = self.left(lfn)
      r:ndarray = self.right(rfn)
      
      return (l, r)
   
   def iter(self, lfn=None, rfn=None, fmtfn=None):
      if lfn is None and rfn is None and fmtfn is not None:
         lfn, rfn = fmtfn, fmtfn
      
      while not self.is_empty():
         yield self.sample(lfn, rfn)
         self.next()
   
   # def __next__(self):
   #    assert self.data is not None
   #    if self.is_empty():
   #       raise StopIteration()
      
   #    r = self.sample()
      
   #    self.next()
      
   #    return r
   
   def indices(self)->Tuple[int]:
      return (self.lpos, self.pos, self.rpos)
   
class TwoPaneWindowIterator:
   def __init__(self, w:TwoPaneWindow) -> None:
      self.w = w
      
   
import torch
from torch import Tensor, tensor, nn

El = Union[Tensor, ndarray]

class TensorBuffer:
   def __init__(self, size:int, item_shape:Tuple[int, int], dtype=None) -> None:
      self._capacity = size
      self.item_shape = item_shape
      self._pos = 0
      self.data = np.empty((size, *item_shape), dtype=dtype)
      
   def is_full(self)->bool:
      return (self._pos >= self._capacity)
   
   def capacity(self)->int:
      return (self._capacity - self._pos - 1)
   
   def size(self)->int:
      return self._capacity
   
   def __len__(self)->int:
      return self._pos
      
   def push(self, el:ndarray):
      self.grow()
      self.data[self._pos] = el
      self._pos += 1
      return self._pos
   
   def pop(self):
      self._pos -= 1
      r = self.data[self._pos]
      return r
   
   def shift(self):
      first = self.data[0]
      for i in range(1, self._pos):
         self.data[i-1] = self.data[i]
      return first
   
   def unshift(self, el:ndarray):
      for i in range(self._pos-1, 1, -1):
         self.data[i+1] = self.data[i]
      self.grow()
      self.data[0] = el
      self._pos += 1
      return self._pos
   
   def get(self)->ndarray:
      return self.data[:self._pos]
   
   def tcpy(self)->Tensor:
      return torch.from_numpy(self.data.copy())
   
   @property
   def T(self)->Tensor:
      return torch.from_numpy(self.get())
   
   def grow(self, mult:int=2):
      if self._pos == len(self.data):
         new_cap = (self._capacity * mult)
         data_size = len(self.data)
         # self.data.resize((new_cap, *self.item_shape))
         _d = self.data
         self.data = np.empty((new_cap, *self.item_shape))
         self._capacity = new_cap
         self.data[:data_size] = _d
      
      return self