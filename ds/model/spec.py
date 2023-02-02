
import warnings, inspect, traceback
from cytoolz import *
from cachetools import *
from functools import *
from numpy import ndarray

from .dp import ShapeDescription as ShapeDesc

# def opt_to_univar(params, df):
class Undef: pass
   
undefined = Undef()
import inspect


def instattrs(obj):
   return [a for a in dir(obj) if not a.startswith('__') and not callable(getattr(obj, a))]


def manip(obj, f):
   for k in instattrs(obj):
      newval = f(obj, k, getattr(obj, k))
      setattr(obj, k, newval)
   return obj

class Hyperparameters:
   def __init__(self, **vars):
      self.n_steps = 14
      self.n_steps_fwd = 1
      self.feature_columns = ['open', 'high', 'low', 'close']
      self.target_columns = ['open', 'high', 'low', 'close']
      self.timescale_unit = '1D'
      self.nsamples = 100000
      # self.univariate = None
      self.target_column = None
      self.feature_column = None
      
      self.parent = vars.pop('parent', None)
      self.nchannels = len(self.feature_columns)
      self.dmods = []

      myattrs = list(self.__dict__.keys())
      for k, v in vars.items():
         if k.startswith('_'): continue
         if k not in myattrs:
            traceback.print_stack()
            warnings.warn(f'assigning non-standard var {k} to NetSpec')
         self.__dict__[k] = v
         
      self._prepcache = {}
      
   def sub(self, **params):
      child:dict = self.to_kwds()
      child.update(dict(**params, parent=self))
      return Hyperparameters(**child)
   
   @property
   def feature_univariate(self):
      return self.feature_column is not None
   
   @property
   def target_univariate(self):
      return self.target_column is not None
      
   @property
   def univariate(self):
      return self.feature_univariate and self.target_univariate
   
   def extract_targets(self, y):
      if self.target_column is not None:
         ndim = y.ndim
         idx = self.target_columns.index(self.target_column)
         
         return y[..., idx]
         # code = 'y[' + (':,' * (ndim - 1)) + f'{idx}]'
         # return eval(code)
      
      return y
   
   def extract_features(self, X):
      if self.feature_column is not None:
         ndim = X.ndim
         idx = self.feature_columns.index(self.feature_column)
         
         return X[..., idx]
         # code = 'X[' + (':,' * (ndim - 1)) + f'{idx}]'
         # return eval(code)

      return X
   
   def prep_data(self, data):
      if id(data) in self._prepcache:
         return self._prepcache[id(data)]
      cachekey = id(data)
      data = data.copy()
      data.y_test = self.extract_targets(data.y_test)
      data.y_train = self.extract_targets(data.y_train)
      data.X_train = self.extract_features(data.X_train)
      data.X_test = self.extract_features(data.X_test)
      # def _f(o, k, v):
      #    if isinstance(v, ndarray):
      #       return self.extract(v)

      #    return v
      # data = manip(data, _f)
      self._prepcache[cachekey] = data
      return data
      
   def mod(self, modfunc):
      """
      adds a function to the pipeline through which all loaded DataFrame objects are passed to apply desired transformations 

      Parameters
      ----------
      modfunc : Callable[DataFrame, DataFrame]
          should modify the DataFrame inplace
      """
      self.dmods.append(modfunc)
      return modfunc
   
   def apply_mods(self, df):
      df = df.copy()
      for f in self.dmods:
         try:
            f(df)
         except:
            f(self, df)
      return df
      
   @property
   def temporal_shape(self):
      return (self.n_steps, self.n_steps_fwd)
   
   @cached_property
   def shapedesc(self):
      inpd = ShapeDesc(*self.feature_columns)
      inpd.by(self.n_steps)
      
      outd = ShapeDesc(*self.target_columns)
      outd.by(self.n_steps_fwd)
      
      return (inpd, outd)
      
   def extend(self, other):
      return Hyperparameters(**merge(self.__dict__, other))
      
   @cached
   def to_key(self):
      return ','.join(map(repr, [self.n_steps, self.n_steps_fwd, self.feature_columns, self.target_columns, self.timescale_unit]))
   
   @cached
   def __hash__(self):
      return hash(self.to_key())
   
   def __eq__(self, other):
      if not isinstance(other, Hyperparameters):
         return False
      else:
         return self.to_kwds() == other.to_kwds()
   
   def to_kwds(self):
      return self.__dict__.copy()

defspec = Hyperparameters()
