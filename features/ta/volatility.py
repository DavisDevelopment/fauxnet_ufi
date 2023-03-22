
import pandas as pd
import pandas_ta as ta
from pandas import DataFrame, Series

from functools import *
from tools import getcolsmatching
from sys import float_info as sflt

def non_zero_range(high: Series, low: Series) -> Series:
    """Returns the difference of two series and adds epsilon to any zero values.  This occurs commonly in crypto data when 'high' = 'low'."""
    diff = high - low
    if diff.eq(0).any().any():
        diff += sflt.epsilon
    return diff

def globcol(df, pattern:str='*'):
   from fnmatch import filter
   matched = filter(list(df.columns), pattern)
   if len(matched) == 0:
      return None
   elif len(matched) == 1:
      return df[matched[0]]
   else:
      return tuple(df[c] for c in matched)

def iwrap(ifn, fwrap=True):
   def owrapped(wfn):
      # @wraps(ifn)
      def wrapper(*args, **kwargs):
         return wfn(ifn, *args, **kwargs)
      if fwrap:
         wrapper = wraps(ifn)(wrapper)
         wrapper.__doc__ = ifn.__doc__
      return wrapper
   return owrapped

@iwrap(ta.bbands)
def bbands(_super, close, output='BBP_*', **kwargs):
   res = _super(close, **kwargs)
   out = res[getcolsmatching(output, res)[0]]
   return out

@iwrap(ta.donchian)
def donchian(_super, high, low, close, **kwargs):
   res = _super(high, low, **kwargs)
   l, u = globcol(res, 'DCL_*'), globcol(res, 'DCU_*')
   ulr = non_zero_range(u, l)
   p = non_zero_range(close, l) / ulr
   return p

@iwrap(ta.accbands)
def accbands(_super, high, low, close, **kwargs):
   res = _super(high, low, close, **kwargs)
   l, u = globcol(res, 'ACCBL_*'), globcol(res, 'ACCBU_*')
   ulr = non_zero_range(u, l)
   p = non_zero_range(close, l) / ulr
   return p

@iwrap(ta.atr)
def atr(_super, high, low, close, percent=True, **kwargs):
   return _super(high, low, close, percent=percent, **kwargs)

@iwrap(ta.vwap)
def delta_vwap(_super, high, low, close, volume, anchor=None, offset=None):
   abs_res:Series = _super(high, low, close, volume, anchor, offset)
   rel_res = abs_res.pct_change()
   rel_res.name = f'delta_{abs_res.name}'
   
   return rel_res