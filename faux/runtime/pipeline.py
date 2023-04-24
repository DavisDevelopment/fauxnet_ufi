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

class PipelineBackendMeta(type):
   registry={}
   
   def __new__(cls, name, bases, attrs):
      # Do something when a new subclass is created
      # print(f"Creating new subclass {name} with bases {bases}")
      PipelineBackendMeta.registry[getattr(cls, '__id__', name)] = cls
      
      return super().__new__(cls, name, bases, attrs)
    
class PipelineBackendBase(metaclass=PipelineBackendMeta):
   def __init__(self, cfg:Any=None):
      self._raw_cfg = cfg
      self.config = self.parse_config(cfg)
      
   def parse_config(self, raw):
      return raw
   
   def load_config(self, config:Dict[str, Any]):
      """
      load the configuration settings from the dictionary onto `self`
      """
      self.config = config
      return self
      
def resolve_engine(engine_id):
   if engine_id in PipelineBackendMeta.registry:
      engine_cls = PipelineBackendMeta.registry[engine_id]
      return engine_cls
   
   else:
      raise KeyError(f'No backend named "{engine_id}" found in registry; Has the module which defines it been loaded?')
   
class PipelinePrototype:
   def __init__(self, raw_config):
      engineCls = resolve_engine(raw_config.get('engine', 'faux:trade'))
      
   
def make_pipeline(proto_config:Dict[str, Any]):
   pass