
from pymitter import EventEmitter
import numpy as np
import pandas as pd
#import modin.pandas as pd
from ds.gtools import *
from ds.model.dp import ShapeDescription, Signature
from cytoolz import *
from sklearn.preprocessing import MinMaxScaler

from typing import *
from numpy import array, asarray, asanyarray
import operator as op
from functools import *
from tools import capturablemethod

class ForecasterUnitBase(EventEmitter):
   sig:Signature
   
   def __init__(self, i, o):
      super().__init__()
      self.sig = Signature(i, o)
      self.scaler = None
      self._shapeValidation = False

   def _format_inputs(self, *inputs):
      values = list(map(tonp, inputs))
      assert isconsistent(values), 'inputs are not consistent'
      is_ = self.sig.input_shape
      inpshape = is_.shape if isinstance(is_, ShapeDescription) else is_
      if inpshape != values[0].shape:
         if compare_shapes(inpshape, values[0].shape):
            result = np.vstack(values)
            assert compare_shapes(inpshape, result.shape)
            return result
         else:
            raise ValueError(f'Expected inputs of shape {inpshape} but got {values[0].shape} instead')
      return np.hstack(values)
      
   def call(self, inputs:np.ndarray):
      'piss and shit'
      
   def __call__(self, inputs:np.ndarray):
      # call_inputs = self._format_inputs(*inputs)
      if self._shapeValidation:
         validate_shapes(inputs.shape, self.sig.input_shape.shape)
      
      return self.call(inputs)
   
   
##from tensorflow.keras import Model
from ds.model.spec import Hyperparameters

class ForecasterScalingContext:
   def __init__(self, owner, scaler):
      self.owner = owner
      self.scaler = scaler
      self.prev = None
      
   def __enter__(self):
      self.prev = self.owner.scaler
      self.owner.scaler = self.scaler
      
   def __exit__(self, type, value, traceback):
      self.owner.scaler = self.prev

#TODO refactor to remove all scaling-related code
class NNForecaster(ForecasterUnitBase):
   # model
   def __init__(self, model, hp:Hyperparameters):
      inp = ShapeDescription(hp.target_column).by(hp.n_steps).by(None)
      outp = ShapeDescription(hp.target_column)
      
      super().__init__(inp, outp)
      
      self.model = model
      self.params = hp
      
      self.inplace_scaling = False
      self.enforce_input_type = None
      self._samples = []
      
   def scalingCtx(self, scaler):
      return ForecasterScalingContext(self, scaler)
   
   def call(self, inputs):
      r = self._call(inputs)
      # if np.count_nonzero(r[r > 1.5]) > 0:
      #    raise ValueError(f'What the fuck', r, r[r > 1.5])
      self._samples.append(((inputs, {}), r))
      return r
      
   def _call(self, inputs:np.ndarray):
      #* HOW CAN THIS BE MORE EFFICIENT??
      
      if self.inplace_scaling:
         raise ValueError("there is no inplace scaling")
      
      if self.enforce_input_type is not None:
         inputs = inputs.astype(self.enforce_input_type)
      
      if True:
         try:
            out = self.model(inputs)
         except Exception as e:
            try:
               inputs = self.model.model.preprocess(inputs, None)[0]
               out = self.model(inputs)
            except:
               raise e
      
      if isinstance(out, np.ndarray):
         #? yaes
         return out
      
      elif hasattr(out, 'numpy'):
         #? most likely a Tensor object in this case
         return out.numpy()
      
      else:
         raise ValueError('Invalid', out)
   
   def __call(self, inputs:np.ndarray):
      ret = super().__call__(inputs)
      
      if self.scaler is not None:
         ret = self._scale(ret)
      
      return ret
   
   def __call__(self, inputs:np.ndarray):
      if not isinstance(inputs, np.ndarray):
         inputs = np.asarray(inputs)
         
      # if inputs.ndim > len(self.input_shape):
      #    print(inputs.shape, self.input_shape)
      if self.inplace_scaling and self.scaler is None and ((inputs > 1)|(inputs < 0)).sum() > 0:
         with self.scalingCtx(MinMaxScaler()):
            self.scaler.fit(inputs)
            return self(inputs)
      return self.__call(inputs)
   
   def _scale(self, ret, inverse=True):
      raise ValueError("absolutely the fuck not")
   
   @cached_property
   def input_shape(self):
      r = self.sig.input_shape.shape
      return r
      
   @cached_property
   def output_shape(self):
      return self.sig.output_shape.shape
   
def compare_shapes(left, right):
   if len(left) != len(right):
      return False
   
   for i in range(len(left)):
      l, r, = left[i], right[i]
      if l is None or r is None:
         continue
      elif l != r:
         return False
   return True

def validate_shapes(left, right):
   valid = compare_shapes(left, right)
   if not valid:
      raise ValueError(f'{left} != {right}')
   
ist = curry(isinstance)

def tonp(x):
   isit = ist(x)
   if isit(np.ndarray):
      return x
   elif isit((np.float, np.integer, int, float)) or np.isscalar(x):
      return x
   elif isit((list, tuple)):
      return array([tonp(o) for o in x])
   elif isit((pd.Series)):
      return x.to_numpy()
   elif isit(pd.DataFrame):
      a = x.to_numpy()
      return a.T
   elif isit((dict, MutableMapping)):
      data = {k:tonp(v) for k,v in x.items()}
      df = pd.DataFrame(data)
      return tonp(df)
   else:
      return asarray(x)

def isconsistent(array, eq=None):
   if eq is None:
      test = complement(op.eq)
   else:
      test = complement(eq)
   inp = map(tonp, array)
   try:
      init = next(inp)
   except StopIteration:
      return True
   for x in inp:
      if test(x.dtype, init.dtype) or not compare_shapes(x.shape, init.shape):
         print((x.dtype, init.dtype), (x.shape, init.shape))
         return False
   return True

def contains(small, big):
    for i in range(len(big)-len(small)+1):
        for j in range(len(small)):
            if big[i+j] != small[j]:
                break
        else:
            return i, i+len(small)
    return False
