import sys, os, json, pickle, re

from cytoolz import partial
# from numba import np
import numpy as np
P = os.path
import main
from faux.pgrid import PGrid, recursiveMap
from faux.features.ta.loadout import IndicatorBag, Indicators, bind_indicator
from tools import Struct, gets, before, after, nor
from cytoolz.dicttoolz import merge, assoc, valmap
from typing import *
import pandas as pd

class SignalGeneratorMeta(type):
   registry={}
   
   def __new__(cls, name, bases, attrs):
      # Do something when a new subclass is created
      # print(f"Creating new subclass {name} with bases {bases}")
      SignalGeneratorMeta.registry[getattr(cls, '__id__', name)] = cls
      
      return super().__new__(cls, name, bases, attrs)
   
class DataLabellerMeta(type):
   registry={}
   
   def __new__(cls, name, bases, attrs):
      # Do something when a new subclass is created
      # print(f"Creating new subclass {name} with bases {bases}")
      DataLabellerMeta.registry[getattr(cls, '__id__', name)] = cls
      
      return super().__new__(cls, name, bases, attrs)
   
class SignalGeneratorBase(metaclass=SignalGeneratorMeta):
   def __init__(self, **options):
      self.output_labels = None
      self.init_options = options.copy()
   
   def __call__(self, inputs:Any):
      #TODO
      pass
   
class 