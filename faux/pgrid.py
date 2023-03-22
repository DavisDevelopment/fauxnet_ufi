
from random import shuffle
from tools import isiterable
from itertools import product
# from tools import closure, dotget, dotgets, before, after, gets, isiterable, once
from numpy import typename
from cytoolz import juxt
from functools import cached_property
from fn import F
from typing import *

class PDef:
   def __init__(self, name=None, value_space:Any=None, constraints=None):
      self.name = name
      self.value_space = value_space if isiterable(value_space) else [value_space]
      self.eliminated_values = []
      self.constraints = None
      if constraints is not None:
         if callable(constraints):
            self.constraints = [constraints]
         elif isiterable(constraints):
            self.constraints = list(constraints)
         else:
            raise TypeError(typename(constraints))
         
   def expand(self):
      need_chk = False
      chk = lambda x: True
      if self.constraints is not None:
         chk = F(juxt(*self.constraints)) << all
         need_chk = True
      
      for v in self.value_space:
         if self.eliminated_values is None or (v not in self.eliminated_values):
            if not need_chk or (need_chk and chk(v)):
               yield v
      
class PEmbeddedGrid(PDef):
   def __init__(self, name=None, parameters=None):
      super().__init__(name)
      
      pg = parameters #TODO convert to PGrid
      self.parameters = pg
      
   @cached_property
   def value_space(self):
      return list(self.parameters.expand())
   
def ls(a, shuffled=False):
   l = list(a)
   if shuffled:
      shuffle(l)
   return l

class PGrid:
   def __init__(self, params=None):
      self.params = []
      
      if params is not None:
         items = get_items(params)
         for k, v in items:
            self.add(name=k, values=v)
      
   def expand(self, shuffled=False):
      k = []
      g = []
      # names = [p.name for p in self.params]
      for p in self.params:
         k.append(p.name)
         a = list(p.expand())
         # print(a)
         g.append(ls(a, shuffled=shuffled))
      
      all_poss = ls(product(*g), shuffled=shuffled)
      all_poss = list(map(lambda x: dict(zip(k, x)), all_poss))
      return all_poss
   
   def add(self, name=None, values:Any=None, parameters=None):
      assert name is not None or values is not None
      if parameters is not None:
         if not isinstance(parameters, (PEmbeddedGrid, PGrid)):
            parameters = PGrid(parameters)
         
         P = PEmbeddedGrid(name=name, parameters=parameters)
      else:
         P = PDef(name=name, value_space=values)
      
      self.params.append(P)
      
      return self
   
def recursiveMap(x, func):
   if isinstance(x, (dict, Mapping)):
      AccCls = type(x)
      try:
         accumulator = AccCls((key, recursiveMap(val, func)) for key, val in x.items())
      except Exception:
         accumulator = {}
         for key, value in x.items():
            accumulator[key] = recursiveMap(value, func)
      return accumulator
   
   elif isinstance(x, Sequence):
      accumulator = []
      for item in x:
         accumulator.append(recursiveMap(item, func))
      return type(x)(accumulator)
   
   else:
      return func(x)
   
def get_items(d):
   if isinstance(d, Mapping):
      yield from d.items()
   elif isiterable(d):
      it = iter(d)
      for x in it:
         if isinstance(x, tuple):
            yield x
            break
         else:
            print(x)
            raise TypeError(typename(x))
      yield from it
   else:
      graceful = False
      if graceful:
         yield (None, d)
      else:
         raise TypeError(typename(d))