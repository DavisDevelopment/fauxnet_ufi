import pandas as pd
#import modin.pandas as pd
import numpy as np
from tools import isiterable
# import ds.utils.bcsv as bkformat
from cytoolz import *

from typing import *

def mkbroadcaster(f, isvoid=False):
   def bf(d:Dict[str, pd.DataFrame], *args, **kwargs):
      if not isvoid:
         ret = {}
      for k, df in d.items():
         if not isvoid:
            ret[k] = f(df, *args, **kwargs)
         else:
            f(df, *args, **kwargs)
      if not isvoid:
         return ret
   return bf

class Book:
   """
   data type for storing a "book" of documents, represented as DataFrames, organized by name
   """
   def __init__(self, data=None):
      self.data = {}
      self._columns = None

      if data is not None:
         for name, doc in data.items():
            self.add(name, doc)
      
   def copy(self, deep=False):
      if not deep:
         return Book(self.data.copy())
      else:
         return Book({name:doc.copy() for name, doc in self.data.items()})
         
   @property
   def columns(self):
      if self._columns is None:
         if len(self.data) == 0:
            return None
         cols = set()
         for df in self.values():
            cols |= set(df.columns)
         self._columns = list(cols)
      return self._columns
         
   def validateColumns(self, doc:pd.DataFrame):
      if self.columns is None:
         return True
      for c in self.columns:
         if c not in doc.columns:
            raise ValueError(f'Column {c} not found in DataFrame {doc.columns}')
         
   def add(self, name:str, document:pd.DataFrame):
      self.validateColumns(document)
      self.data[name] = document
      
   def map(self, getter):
      return pd.Series([getter(self.data[n]) for n in self.keys()], index=self.keys())
   
   def apply(self, func, inplace=False):
      cls = type(self)
      data = self.data if not inplace else {}
      for name, doc in self.data.items():
         data[name] = func(doc)
      if not inplace:
         return self
      else:
         return cls(data=data)
      
   def keys(self): return list(self.data.keys())
   def items(self)->Tuple[str, pd.DataFrame]: return self.data.items()
   def values(self)->Iterator[pd.DataFrame]: 
      return self.data.values()
   
   def __iter__(self): return iter(self.data)
   
   def __getitem__(self, idx):
      if idx in self.keys():
         return self.data[idx]
      else:
         raise KeyError(idx)
   
   def __setitem__(self, idx, value):
      if isinstance(value, pd.DataFrame):
         self.add(idx, value)
         
   def __delitem__(self, idx):
      if idx in self.keys():
         del self.data[idx]
         
   def __len__(self):
      return len(self.data)
         
         
class TsBook(Book):
   def __init__(self, data=None):
      # super().__init__(data)
      self._columns = ['time', 'open', 'high', 'low', 'close', 'volume']
      self._set_data(data=data)
      
   def _set_data(self, data=None):
      self.data = {}
      self._mints = None
      self._maxts = None
      
      if data is not None:
         for name, doc in data.items():
            self.add(name, doc)
      self._refresh_time_range()
      return self.data

   def add(self, name, doc: pd.DataFrame):
      super().add(name, doc)
      if doc.index.dtype != 'datetime64[ns]':
         doc.set_index(doc['time'], inplace=True, drop=False)

      if self._mints is None and self._maxts is None:
         self._mints = doc.time.min()
         self._maxts = doc.time.max()
      else:
         cur_ts_range = (self._mints, self._maxts)
         self._mints = min(cur_ts_range[0], doc.time.min())
         self._maxts = max(cur_ts_range[1], doc.time.max())
         
   def _refresh_time_range(self):
      ranges = list(map(lambda d: (d.index.min(), d.index.max()), self.values()))
      mints = min(*[r[0] for r in ranges])
      maxts = max(*[r[1] for r in ranges])
      self._mints = mints
      self._maxts = maxts
      
   def __delattr__(self, idx):
      if idx in self.keys():
         del self.data[idx]
         # self._refresh_time_range()

   @property
   def time_range(self):
      # return pd.date_range(start=self._mints, end=self._maxts, freq='D')
      return (self._mints, self._maxts)
   
   @property
   def time_index(self):
      return pd.date_range(start=self._mints, end=self._maxts)

T = TypeVar('T')
class FlatBook(Generic[T]):
   book:T = None
   def __init__(self, orig:T=None, flat:pd.DataFrame=None):
      assert orig is not None
      assert flat is not None
      self.book = orig
      self.data = flat
      
      