import numpy as np
import pandas as pd
#import modin.pandas as pd
from cytoolz import *
from numba import jit, njit, vectorize

@vectorize(target='cpu', cache=True)
# @njit
def rescale(OldValue, OldMin: float, OldMax: float, NewMin: float, NewMax: float):
   """
   (sweet-ass perk of doing it this way is that it will work for any shape at all)
   """

   OldRange = (OldMax - OldMin)
   NewRange = (NewMax - NewMin)
   NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
   return NewValue

@jit(cache=True)
def rollingscale_minmax(src:np.ndarray, at=-1, sample_length=50, target_length=22, target_range=(0, 1), sample_range=None):
   i = at
   target:np.ndarray = src[(i - target_length):i]
   
   if sample_range is None:
      sample:np.ndarray = src[(i - sample_length):i]
      
      local_min = sample.min()
      local_max = sample.max()
   else:
      local_min, local_max = sample_range
   
   result = rescale(target, local_min, local_max, target_range[0], target_range[1])
   
   return (local_min, local_max), result

@jit(cache=True)
def rowRollingScale(src:np.ndarray, target_range=(0, 1), sample_range=None):
   target:np.ndarray = src.copy()
   if sample_range is None:
      sample: np.ndarray = src

      local_min = sample.min()
      local_max = sample.max()
   else:
      local_min, local_max = sample_range
   target_min, target_max = target_range
   target = rescale(target, local_min, local_max, target_min, target_max)
   return (local_min, local_max), target

class RollingMinMaxScaler:
   def __init__(self, length=50):
      pass
   
   # def transform