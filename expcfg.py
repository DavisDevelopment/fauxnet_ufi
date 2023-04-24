
import sys, os, json, pickle, re

from cytoolz import partial
# from numba import np
import numpy as np
P = os.path
import main
from faux.pgrid import PGrid, recursiveMap
from faux.features.ta.loadout import IndicatorBag, Indicators, bind_indicator
from tools import Struct, gets, before, after
from cytoolz.dicttoolz import merge, assoc, valmap
from typing import *
import pandas as pd

def read_config(name=None, fp=None):
   if fp is None and name is not None:
      fp = P.join(P.dirname(main.__file__), 'experiments', name, 'config.json')
   if not P.exists(fp):
      raise FileNotFoundError(fp)
   with open(fp, 'r') as f:
      raw_data = json.load(f)
      return raw_data
   
def parse_parameters(self, rawparams:Dict[str, Any]):
   consts = Struct()
   grid = PGrid()
   
   for k, v in rawparams.items():
      if isinstance(v, (type(None), bool, int, float, str)):
         #* Constants
         consts[k] = v
      elif isinstance(v, (list, dict)):
         #* Explicitly listed possible values
         if isinstance(v, dict):
            grid.add(name=k, parameters=v)
         elif isinstance(v, list):
            if len(v) == 0:
               continue
            elif len(v) == 1:
               consts[k] = v[0]
            else:
               grid.add(name=k, values=v)
         else:
            raise TypeError(f'Unexpected {type(v).__qualname__} value encountered')
   
   if len(grid.params) == 0:
      grid = None
   return consts, grid

def resolve_or_return(self, idOrValue:Any):
   if isinstance(idOrValue, str):
      m = re.search(r'\$(?!\$)([a-zA-Z_][a-zA-Z0-9_.]*)', idOrValue)
      if m is None:
         return idOrValue
      else:
         # mg = 
         # print(mg)
         (ref_id,) = m.groups()
         return self.parameters.get(ref_id, None)
   else:
      return idOrValue

def expand_feature_args(self, argdef:Dict[str, Any]):
   params = self.parameters
   
   return valmap(partial(resolve_or_return, self), argdef)
   
def parse_feature_spec(self, items:List[Any]):
   feats = []
   
   for item in items:
      if isinstance(item, str):
         feats.append(('column', item))
      
      elif isinstance(item, dict):
         # print(item)
         k, v = next(iter(item.items()))
         print(k, v)
         feats.append(('call', k, partial(expand_feature_args, self, v)))
      else:
         raise TypeError(f'Invalid feature specification: {item}')
   
   return feats

_empty_opts = (lambda : {})
FeatSpecItem = TypeVar('FeatSpecItem', Tuple[str, str], Tuple[str, str, Callable[[], Dict[str, Any]]])
def compile_feature_spec(self, feature_spec:List[FeatSpecItem]):
   feature_loaders = []
   feature_columns = []
   
   for item in feature_spec:
      item_type, item_ref = item[0], item[1]
      if item_type == 'column':
         feature_columns.append(item_ref)
         feature_loaders.append(partial(lambda col_id, df: df[col_id], item_ref))
         
      elif item_type == 'call':
         # print(item_ref)
         resolve_options = item[2] if len(item) > 2 else _empty_opts
         assert callable(resolve_options)
         
         if item_ref.startswith('ta.'):
            item_ref = after(item_ref, 'ta.')
            bound_fn = bind_indicator(item_ref)
            print(item_ref, bound_fn)
         
         fl_fn = partial(lambda df, opts=_empty_opts: bound_fn(df, options=opts()), opts=resolve_options)
         feature_loaders.append(fl_fn)
         
   def compiled_extractor(df:pd.DataFrame):
      extracted = []
      init_columns = list(df.columns)
      exclude_columns = (set(init_columns) - set(feature_columns))
      
      for lfn in feature_loaders:
         eret = lfn(df)
         feature = None
         if isinstance(eret, pd.Series):
            #TODO handle the Series' index, when it isn't identical to df.index
            extracted.append(eret)
         elif isinstance(eret, pd.DataFrame):
            eret:pd.DataFrame
            new_columns = list(set(eret.columns) - set(init_columns))
            extracted.extend([eret[c] for c in new_columns])
         elif isinstance(eret, np.ndarray):
            assert len(eret) == len(df.index)
            eret = pd.Series(data=eret, index=df.index)
            extracted.append(eret)
         
      global_offset = max(*[s.dropna().index[0] for s in extracted])
      np_features = np.asanyarray([s[global_offset:].to_numpy() for s in extracted]).T
      
      # return extracted
      return np_features
   
   return compiled_extractor

def parse_config(config=None, name=None, fp=None):
   """
   TODO: refactor function to parse a config file of the form shown below
   ```
   {
      "
   }
   ```
   """
   raw = read_config(name=name, fp=fp) if config is None else config
   assert isinstance(raw, dict)
   
   out = Struct(gets(raw, 'title', 'description', asitems=True))
   features, parameters = gets(raw, 'features', 'parameters')
   
   #* parse the parameters
   (const_parameters, parameter_space) = parse_parameters(out, parameters)
   selected_tuning = out.selected_tuning = raw.pop('$$selected_parameter_tuning', None)
   needs_tuning = out.needs_tuning = (parameter_space is not None and selected_tuning is None)
   out.parameters = const_parameters.asdict()
   out.parameter_space = parameter_space
   
   #* parse the input-features
   feature_spec = parse_feature_spec(out, features)
   out.feature_spec = feature_spec
   out.extract_features = compile_feature_spec(out, feature_spec)
   
   return out