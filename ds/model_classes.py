
import random
import numpy as np
import os
P = os.path
from numpy import *
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

from ds.common import *
from ds.gtools import ojit, Struct, mpmap, njit, jit, vprint, TupleOfLists
import ds.data as dsd
from ds.maths import *

#import tensorflow as tf
##import tensorflow.keras as keras
## keras = tf.keras
##from tensorflow.keras import *
##from tensorflow.keras import Sequential, Model
##from tensorflow.keras.layers import *
##from tensorflow.keras.models import clone_model, model_from_json
##from tensorflow.keras.optimizers import *
import warnings
import shelve, pickle
import pickletools as pt

import random
import collections
import matplotlib.pyplot as plt

from .model_train import *

#from keras.engine import base_layer
#from keras.engine import base_layer_utils
#from keras.engine import compile_utils
#from keras.engine import data_adapter
#from keras.engine import input_layer as input_layer_module
#from keras.engine import training_utils

# tf.config.optimizer.set_jit(True)

import toolz.dicttoolz as dicts
import math

# def compcall(*module_names, noself=False):
#     _d = {}
#     code = [
#         'def call(self, inputs, training=None):'
#     ]
#     if noself:
#         code[0] = code[0].replace('(self, ', '(')
#     def line(s):
#         code.append('  ' + s)
#     line('x = inputs')
#     for m in module_names:
#         line(f'x = self.{m}(x, training=training)')
#     line('return x')
#     code = '\n'.join(code)
#     exec(code, _d)
#     f = _d['call']
#     f = tf.function(f, jit_compile=True)
#     return f


# #L = keras.layers

# L.LSTM
# tf.compat.v1.enable_eager_execution()
# from functools import reduce
# class Squishifier2D(Layer):
#     def __init__(self, name=None, dtype=None, use_global_reducer=False, axis=-2, **kwds):
#         super().__init__(name=name, dtype=dtype, **kwds)
#         self.biases = None
#         self.axis = axis
#         self.nb = None
#         self.use_global_reducer = use_global_reducer
        
#     def build(self, input_shape=None):
#         super().build(input_shape)
#         print(input_shape)
        
#         # last_dim = tf.compat.dimension_value(input_shape[1])
#         # self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})

#         nb = self.nb = int(input_shape[self.axis])
#         print(nb)
#         # pre_shape = input_shape[:self.axis]
#         post_shape = input_shape[self.axis+1:]
#         ncols = post_shape[0]
#         if self.use_global_reducer:
#             self.reducer = RowReduce(name='lerp', is_global=True)
#         else:
#             self.mergers = [RowReduce(name=f'lerp_{i}_{i+1}') for i in range(nb-1)]
#         for l in self.mergers:
#             l.build(tf.TensorShape([2, ncols]))
        
#         self._squish = partial(mksquisher(nb), self)
        
#     @tf.function
#     def call(self, inputs:tf.Tensor, **kwargs):
#         if inputs.ndim == 3:
#             r = tf.map_fn(self._squish, inputs, back_prop=True)
#             return r
#         elif inputs.ndim == 4:
#             inner = lambda x: tf.map_fn(self._squish, x, back_prop=True)
#             r = tf.map_fn(inner, inputs, back_prop=True)
#             return r
#         else:
#             return inputs

# class Squishifier2DV2(Layer):
#     def __init__(self, name=None, dtype=None, axis=-2, transposed=False, **kwds):
#         super().__init__(name=name, dtype=dtype, **kwds)
#         self.biases = None
#         self.axis = axis
#         self.nb = None
#         self.transposed = transposed

#     def build(self, input_shape=None):
#         super().build(input_shape)
#         nb = self.nb = int(input_shape[self.axis])
#         post_shape = input_shape[self.axis+1:]
#         ncols = post_shape[0]
#         self.mergers = [RowReduce(name=f'lerp_{i}_{i+1}') for i in range(nb-1)]

#         for l in self.mergers:
#             l.build(tf.TensorShape([2, ncols]))
        
#         self._squish = partial(mksquisherv2(nb), self)

#     @tf.function
#     def call(self, inputs: tf.Tensor, **kwargs):
#         _reduce = partial(self._squish, **kwargs)
#         r = tf.map_fn(_reduce, inputs, back_prop=True)
        
#         if self.transposed:
#             return tf.transpose(r, perm=[0, 2, 1])
#         else:
#             return r
    
# def mksquisherv2(nrows:int):
#     lines = [
#         'def squish(self, X, *rest, **kw):'
#     ]
    
#     def put(s, lvl=1):
#         s = s.lstrip()
#         s = (' ' * (lvl * 4)) + s
#         lines.append(s)
    
#     put(f'acc = []')
#     # put
#     for i in range(nrows-1):
#         # tf.expand_dims()
#         stacked = f'tf.stack([X[{i}], X[{i+1}]])'
#         put(f'acc.append(self.mergers[{i-1}]({stacked}, **kw))')
#     put('return tf.stack(acc)')
#     code = '\n'.join(lines)
#     _d = {'tf': tf}
#     exec(code, _d)
#     f = _d['squish']
#     f = tf.function(f, jit_compile=True)
#     return f
    
# def mksquisher(nrows:int):
#     lines = [
#         'def squish(self, X, *rest, **kw):'
#     ]
#     def put(s, lvl=1):
#         s = s.lstrip()
#         s = (' ' * (lvl * 4)) + s
#         lines.append(s)
#     put(f'acc = X[0]')
#     put('if self.use_global_reducer:')
#     for i in range(nrows-1):
#         # tf.expand_dims()
#         stacked = f'tf.stack([acc, X[{i}]])'
#         put(f'acc = self.reducer({stacked}, **kw)', lvl=2)
#     put('else:')
    
#     for i in range(nrows-1):
#         stacked = f'tf.stack([acc, X[{i}]])'
#         put(f'acc = self.mergers[{i-1}]({stacked}, **kw)', lvl=2)
#     put('return acc')
#     code = '\n'.join(lines)
#     _d = {'tf': tf}
#     exec(code, _d)
#     f = _d['squish']
#     f = tf.function(f, jit_compile=True)
#     return f

# @tf.function(jit_compile=True)
# def lerp(a, b, t):
#     """Linear interpolate on the scale given by a to b, using t as the point on that scale.
#     Examples
#     --------
#         50 == lerp(0, 100, 0.5)
#         4.2 == lerp(1, 5, 0.8)
#     """
#     return (1 - t) * a + t * b

# class RowReduce(Layer):
#     def __init__(self, use_weights=False, is_global=False, **kw):
#         super().__init__(**kw)
#         # self.b = 
#         self.activation = kw.pop('activation', None)
#         if self.activation is not None and isinstance(self.activation, str):
# #            self.activation = tf.keras.activations.get(self.activation)
#             pass
#         self.use_weights = use_weights
#         self.is_global = is_global
        
#     def build(self, input_shape):
#         dtype = tf.as_dtype(self.dtype or backend.floatx())
#         input_shape = tf.TensorShape(input_shape)
#         last_dim = tf.compat.dimension_value(input_shape[-1])
#         self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
        
#         self.t = self.add_weight(
#             "t",
#             shape=[last_dim],
#             initializer=tf.constant_initializer(value=0.5),
# #            constraint=tf.keras.constraints.NonNeg(),
#             dtype=self.dtype,
#             trainable=True
#         )
        
#         if self.use_weights:
#             self.w = self.add_weight('w', 
#                 shape=[2, last_dim], 
# #                initializer=tf.keras.initializers.RandomUniform(minval=0.0, maxval=1.0),
# #                constraint=tf.keras.constraints.NonNeg(),
#                 dtype=self.dtype,
#                 trainable=True
#             )
        
#         # self.bias
#         self.built = True
        
#     @tf.function(jit_compile=True)
#     def call(self, inputs:tf.Tensor, **kw):
#         x = inputs[0]
#         y = inputs[1]
        
#         if self.use_weights:
#             xW = self.w[0]
#             yW = self.w[1]
#             ret = lerp((x * xW), (y * yW), self.t)
#         else:
#             ret = lerp(x, y, self.t)
#         if self.is_global:
#             pass
            
#         if self.activation is not None:
#             ret = self.activation(ret)
#         return ret

# def mkmerger(ncols:int):
#     i = Input(shape=(2, ncols))
#     return Sequential([
#         i,
#         Flatten(),
#         Dense(ncols, activation='relu')
#     ])
    
# class Squishifier3D(Layer):
#     def __init__(self, **kwargs):
#         super().__init__(dynamic=True, trainable=True, **kwargs)
#         self.sq = Squishifier2D(**kwargs)
        
#     def compute_output_shape(self, input_shape):
#         nsamples = input_shape[0]
#         nfilters = input_shape[-1]
#         nchannels = input_shape[-2]

#         return tf.TensorShape((nsamples, nfilters, nchannels))
    
#     # @tf.function(jit_compile=True)
#     @tf.autograph.experimental.do_not_convert
#     def call(self, inputs:tf.Tensor, **kwargs):
#         tf.print(inputs)
#         tf.print(inputs.shape)
#         n:int = inputs.shape[-1]
#         r = []
#         for j in tf.range(n):
#             x = inputs[:, j]
#             r.append(self.sq(x))
#         tr = tf.stack(r)
#         rshape = list(tr.shape)
#         # _n = rshape[1]
#         # rshape[1] = rshape[0]
#         # rshape[0] = _n
#         rshape[1],rshape[0] = rshape[0],rshape[1]
#         tr = tf.reshape(tr, rshape)
#         tf.print(tr)
#         return tr

# class RecurrentOHLCCell(Layer):
#     def __init__(self, ntimesteps=14, nchannels=1, name=None):
#         super().__init__(name=name)
        
#         self.ntimesteps = ntimesteps
#         self.nchannels = nchannels
        
#         self.a = L.LSTM(ntimesteps * 2, unroll=True, stateful=True)#, return_state=True)
#         self.out = L.Dense(nchannels, activation='sigmoid')
        
#     def reset_states(self, states=None):
#         self.a.reset_states(states)
        
#     def call(self, inputs, training=None):
#         x = inputs
#         for i in range(self.ntimesteps):
#             x = self.a(x[i, :])
#         # x = self.a(inputs)
#         # ...
#         return self.out(x)
    
# class CModel(Model):
#     def preprocess(self, x, y):
#         if x is None and y is None:
#             return x, y
#         if x is not None:
#             x = x.reshape(x.shape[0], 1, x.shape[1])
#         return x, y
    
# @tf.function(jit_compile=True)
# def callmodules(inputs, *steps, **kwargs):
#     x = inputs
#     for f in steps:
#         x = f(x, **kwargs)
#     return x
    
    
# def dcnnpred_2d(seq_len=60, n_features=82, n_filters=(8, 8, 8), droprate=0.025):
#     "2D-CNNpred model according to the paper"
#     model = Sequential([
#         Input(shape=(seq_len, n_features, 1)),
#         Conv2D(n_filters[0], kernel_size=(1, n_features), activation="relu", padding='same'),
#         MaxPool2D(pool_size=(2, 1), padding='same'),
#         Conv1D(n_filters[1], 1, activation="relu", padding='same'),
#         Flatten(),
#         Dropout(droprate),
#         Dense(50, activation='relu'),
#         Dense(400),
#         Dense(n_features)
#     ])
#     return model


# def cnnpred_2d(seq_len=10, n_features=4, n_features_out=4, n_filters=(4, 16), hidden=(1500,), droprate=0.001):
#     convkw = dict(padding='same', activation='relu')
#     model = Sequential([
#         Input(shape=(seq_len, n_features)),
#         Lambda(lambda x: tf.expand_dims(x, axis=-1)),
#         Conv2D(n_filters[0], kernel_size=2, **convkw),
#         Flatten(),
#         *[Dense(n, activation='relu') for n in hidden],
#         Dense(n_features_out)
#     ])
#     return model


# def cnnpred_2d2(seq_len=10, n_features=4, n_features_out=4, n_filters=(16, 16), hidden=(500,), droprate=0.001):
#     convkw = dict(padding='same', activation='relu')
#     model = Sequential([
#         Input(shape=(seq_len, n_features, 1)),
#         # Lambda(lambda x: tf.transpose(x)),
#         # Lambda(lambda x: tf.expand_dims(x, axis=-1)),
#         Conv2D(n_filters[0], kernel_size=4, **convkw),
#         # MaxPool2D(padding='same'),
#         # Conv2D(3, kernel_size=2, strides=(2, 1), **convkw),
#         Flatten(),
#         # Dropout(droprate),
#         # *[Dense(n, activation='relu') for n in hidden],
#         Dense(400, activation='relu'),
#         Dense(n_features_out)
#     ])
#     return model

# def compile_cnn(model):
#     _dense = lambda inputs, w, b: f'matmul({inputs}, {w}) + {b}'
    
#     raise NotImplementedError()

# def cnnpred_2d3(seq_len=10, n_features=4, n_features_out=4, n_filters=(16, 16), hidden=1500, droprate=0.001):
#     #TODO evaluate this model's performance against the previous revision
#     #! ALWAYS update the model architecture this way, so that the best performer is never lost again
#     convkw = dict(padding='same', activation='relu')
#     model = Sequential([
#         Input(shape=(seq_len, n_features, 1)),
#         Conv2D(n_filters[0], kernel_size=2, **convkw),
#         MaxPool2D(pool_size=2, padding='same'),
#         Conv2D(n_filters[1], kernel_size=2, **convkw),
#         MaxPool2D(pool_size=2, padding='same'),
#         Flatten(),
#         # Dropout(droprate),
#         # *[Dense(n, activation='relu') for n in hidden],
#         # Dense(600, activation='relu'),
#         Dense(n_features_out, activation='relu')
#     ])
#     return model

# def cnnpred_deep2d(seq_len=10, n_features=4, n_features_out=4, n_filters=(4, 4), hidden=200, droprate=0.001):
#     #TODO evaluate this model's performance against the previous revision
#     #! ALWAYS update the model architecture this way, so that the best performer is never lost again
#     convkw = dict(padding='same', activation='relu')
#     model = Sequential([
#         Input(shape=(seq_len, n_features, 1)),
#         Conv2D(n_filters[0], kernel_size=(n_features, 2), **convkw),
#         AvgPool2D(pool_size=(2, 2), padding='same'),
#         Flatten(),
#         Dense(hidden),
#         Dense(n_features_out, activation='relu')
#     ])
    
#     return model


# def cnnpred_bigdeep2d(seq_len=10, n_features=4, n_features_out=4, n_filters=(4, 4), hidden=900, droprate=0.001):
#     #TODO evaluate this model's performance against the previous revision
#     #! ALWAYS update the model architecture this way, so that the best performer is never lost again
#     convkw = dict(padding='same', activation='relu')
#     model = Sequential([
#         Input(shape=(seq_len, n_features, 1)),
#         Conv2D(n_filters[0], kernel_size=(2, 2), **convkw),
#         AvgPool2D(pool_size=(2, 2), padding='same'),
#         Flatten(),
#         Dense(hidden),
#         Dense(n_features_out, activation='relu')
#     ])

#     return model

# def deep2d2(seq_len=10, n_features=4, n_features_out=4, n_filters=(4, 4), hidden=(1500,), droprate=0.001):
#     #TODO evaluate this model's performance against the previous revision
#     #! ALWAYS update the model architecture this way, so that the best performer is never lost again
#     convkw = dict(padding='same', activation='relu')
#     # model = Sequential([
#     enter = Input(shape=(seq_len, n_features, 1))
#     l = enter
#     l = Conv2D(6, kernel_size=(2, 2), padding='same', activation='relu')(l)
#     l = tf.transpose(l, perm=[0, 3, 1, 2])
#     l = Squishifier2D()(l)
    
#     l = Flatten()(l)
    
#     l = Dense(800)(l)
#     l = Dense(n_features_out, activation='relu')(l)
#     model = Model(inputs=enter, outputs=l)
    
#     return model

# def pred_rcnn(seq_len=10, n_features=4, n_features_out=4, memsize=None, n_filters=(4, 4)):
#     if memsize is None: memsize = seq_len
    
#     unroll = True
#     RecL = LSTM
#     RecKw = dict(unroll=unroll)
#     if RecL is GRU:
#         RecKw['reset_after'] = True
    
#     RL = lambda *args, **kw: RecL(*args, **dicts.merge(RecKw, kw))
    
#     model = Sequential([
#         Input(shape=(seq_len, n_features)),
#         RL(seq_len, return_sequences=True),
#         Lambda(lambda x: tf.expand_dims(x, -1)),
#         Conv2D(n_filters[0], kernel_size=(n_features, 1), activation='relu', padding='same'),
#         AvgPool2D(pool_size=(2, 2), padding='same'),
#         Dense(200),
#         Flatten(),
#         Dense(n_features_out, activation='relu')
#     ])
    
#     return model

# def cnnpred_1d3(seq_len=10, n_features=4, n_features_out=4, n_filters=(16, 16), hidden=(1500,), droprate=0.001):
#     #TODO evaluate this model's performance against the previous revision
#     #! ALWAYS update the model architecture this way, so that the best performer is never lost again
#     convkw = dict(padding='same', activation='relu')
#     model = Sequential([
#         Input(shape=(seq_len, n_features)),
#         Conv1D(n_filters[0], kernel_size=2, **convkw),
#         MaxPool1D(pool_size=2, padding='same'),
#         Conv1D(n_filters[1], kernel_size=2, **convkw),
#         # MaxPool1D(pool_size=2, padding='same'),
#         Flatten(),
#         # Dropout(droprate),
#         # *[Dense(n, activation='relu') for n in hidden],
#         # Dense(600, activation='relu'),
#         Dense(n_features_out, activation='relu')
#     ])
#     return model


# def cnnpred_1d4(seq_len=10, n_features=4, n_features_out=4, n_filters=(16, 16), hidden=(1500,), droprate=0.001):
#     #TODO evaluate this model's performance against the previous revision
#     #! ALWAYS update the model architecture this way, so that the best performer is never lost again
#     convkw = dict(padding='same', activation='relu')
#     model = Sequential([
#         Input(shape=(seq_len, n_features)),
#         Conv1D(n_filters[0], kernel_size=2, **convkw),
#         MaxPool1D(pool_size=2, padding='same'),
#         Conv1D(n_filters[1], kernel_size=2, **convkw),
#         Flatten(),
#         *[Dense(n, activation='relu') for n in hidden],
#         Dense(n_features_out, activation='relu')
#     ])
#     return model

# def cnnpred_1d(seq_len=10, n_input_features=4):
#     model = Sequential([
#         Input(shape=(seq_len, n_input_features)),
#         Lambda(lambda x: tf.expand_dims(x, axis=-2)),
#         Conv1D(6, 1, activation='relu', padding='same'),
#         Flatten(),
#         Dense(1000, activation='relu'),
#         Dense(1)
#     ])
    
#     return model

# def pred_1d(seq_len=10, hidden=(100,100)):
#     model = Sequential([
#         Input(shape=(seq_len,)),
#         Dense(seq_len, activation='relu'),
#         *[Dense(size, activation='relu') for size in hidden],
#         Dense(1)
#     ])
#     return model

# def pred_rnn(seq_len=10, n_features=4, n_features_out=1, hidden=(100,100), memsize=10):
#     BdRL = lambda l,**kw: l
#     model = Sequential([
#         Input(shape=(seq_len, n_features)),
#         Lambda(lambda x: tf.expand_dims(x, axis=-1)),
#         ConvLSTM1D(seq_len, 1, padding='same'),
#         Flatten(),
#         Dense(10, activation='relu'),
#         Dense(1)
#     ])
    
#     return model

# def pred_rnn2(seq_len=10, n_features=4, n_features_out=4, memsize=None):
#     if memsize is None:
#         memsize = seq_len
        
#     unroll = True
#     RecL = LSTM
#     RecKw = dict(unroll=unroll)
#     if RecL is GRU:
#         RecKw['reset_after'] = True
    
#     RL = lambda *args, **kw: RecL(*args, **dicts.merge(RecKw, kw))
    
#     model = Sequential([
#         Input(shape=(seq_len, n_features)),
#         RL(seq_len, return_sequences=True),
#         RL(memsize, dropout=0.001),
#         Dense(500, activation='relu'),
#         Dense(n_features_out)
#     ])
    
#     return model


# def pred_rnn3(seq_len=10, n_features=4, n_features_out=4, memsize=10):
#     model = Sequential([
#         Input(shape=(seq_len, n_features)),
#         LSTM(seq_len, return_sequences=True, unroll=True),
#         LSTM(seq_len, return_sequences=True, unroll=True),
#         LSTM(memsize, unroll=True),
#         Dense(25, activation='relu'),
#         Dense(25, activation='relu'),
#         Dense(n_features_out)
#     ])
#     return model

# class Cell(CModel):
#     def __init__(self, ntimesteps=None, **kwds):
#         super().__init__(**kwds)
        
#         # self.pred2d = cnnpred_1d(seq_len=ntimesteps, n_features=1)
#         # pruning_params = {
#         #     'pruning_schedule': S.ConstantSparsity(
#         #         target_sparsity=0.5,
#         #         begin_step=0,
#         #         frequency=2
#         #     )
#         # }
        
#         # P = lambda l: S.prune_low_magnitude(l, sparsity_m_by_n=(2, 4))
        
#         self.ntimesteps = ntimesteps

#         self.conv1 = Conv1D(6, 1, activation='relu', padding='same')
#         self.conv2 = Conv1D(6, 1, activation='relu', padding='same')
#         self.conv3 = Conv1D(6, 1, activation='relu', padding='same')
#         self.pool = GlobalMaxPool1D(keepdims=True, trainable=True)
#         # self.pool = MaxPool1D(pool_size=1)
#         self.flat = Flatten()
#         self.d1 = Dense(1000, activation='relu')
#         self.d2 = Dense(500, activation='relu')
#         self.d3 = Dense(100, activation='relu')
#         # self.d2 = Dense(80)
#         self.out = Dense(1)
        
#         # self.conv, self.flat, self.d1, self.out = tuple([P(l) for l in (self.conv, self.flat, self.d1, self.out)])
        
#     def call(self, inputs, training=None):
#         x = inputs
#         # return self.pred2d(x, training=training)
#         conv1, conv2, conv3, pool, flat, d1, d2, d3, out = self.layers
        
#         x = conv1(x, training=training)
#         x = conv2(x, training=training)
#         x = conv3(x, training=training)
#         # x = pool(x, training=training)
#         x = flat(x, training=training)
#         x = d1(x, training=training)
#         x = d2(x, training=training)
#         x = d3(x, training=training)
#         x =  out(x, training=training)
        
#         return x
    
# class RecurrentOHLC(Model):
#     def __init__(self, ntimesteps=14, nchannels=4, name=None):
#         super().__init__(name=name)
#         self.ntimesteps = ntimesteps
#         self.nchannels = nchannels
        
#         self.rnn1 = LSTM(ntimesteps, unroll=True, return_sequences=True)
#         self.conv1 = Conv1D(4, 2, activation='relu', padding='same')
#         self.flatten = Flatten()
#         self.rnn2 = LSTM(32, unroll=True)
#         self.fc1 = Dense(1200)
#         self.out = Dense(nchannels, activation='sigmoid')
        
# RecurrentOHLC.call = compcall('rnn1', 'conv1', 'flatten', 'rnn2', 'fc1', 'out')

# class OHLCCell1D(Layer):
#     def __init__(self, ntimesteps=14, nchannels=1, name=None, dtype='float64'):
#         super().__init__(name=name, dtype=dtype)
#         self.ntimesteps = ntimesteps
#         self.conv1 = Conv1D(10, 2, strides=1, activation='relu', padding='same', dtype=dtype)
        
#         self.flat = Flatten(dtype=dtype)

#         # self.d1 = Dense(1152, dtype=dtype)
#         self.d2 = Dense(500, dtype=dtype)
#         self.out = Dense(1, activation='sigmoid', dtype=dtype)
        
#     # def call(self, inputs, training=None):
#     #     x = self.conv1(inputs)
#     #     x = self.flat(x)
#     #     x = self.
#     #     x = self.dense(x)
#     #     return self.f(x)


# setattr(OHLCCell1D, 'call', compcall('conv1', 'flat', 'd2', 'out'))

# class OHLCCell(Layer):
#     def __init__(self, ntimesteps=14, nchannels=1, name=None, **kwargs):
#         super().__init__(name=name)
#         self.ntimesteps = ntimesteps
#         self.nchannels = nchannels
        
#         self.conv1 = Conv2D(3, 2, strides=(2, 1), activation='relu', padding='same', **kwargs)
#         self.pool = MaxPool2D(pool_size=(2, 1), padding='same', **kwargs)
#         self.flat = Flatten(**kwargs)
#         self.dense = Dense(400, **kwargs)
#         self.out = Dense(1, activation='relu', **kwargs)
        
#     def build(self, input_shape):
#         pass
    
#     def get_config(self):
#         return dicts.merge(super().get_config(), dict(ntimesteps=self.ntimesteps, nchannels=self.nchannels))
    
#     @tf.function(jit_compile=True)
#     def call(self, inputs, training=None, **kw):
#         x = inputs
#         x =  self.conv1(x, training=training)
#         x =  self.pool(x, training=training)
#         x =   self.flat(x, training=training)
#         x =  self.dense(x, training=training)
#         x =  self.out(x, training=training)
#         return x
     
# # setattr(OHLCCell, 'call', compcall('conv1', 'conv2', 'flat', 'd1', 'out'))


# class DeepOHLCCell(Layer):
#     def __init__(self, ntimesteps=14, nchannels=1, name=None):
#         super().__init__(name=name)
        
#         self.ntimesteps = ntimesteps
#         self.nchannels = nchannels
#         self.fc_hidden_layer_sizes = [200, 25]
#         self.convolutions = [
#             # dict(filters=64, kernel_size=2, strides=(2, 1)),
#             # dict(filters=128, kernel_size=8),
#             # dict(filters=32, kernel_size=2, strides=(2, 1)),
#             dict(filters=32, kernel_size=2, strides=(2, 1)),
#             # dict(filters=16, kernel_size=2, strides=(2, 1)),
#             # dict(filters=16, kernel_size=1, strides=(2, 1)),
#             # dict(filters=6, kernel_size=2, strides=(1, 1)),
#             # dict(filters=4, kernel_size=1, strides=(1, 1)),
#         ]
        
#         self._modnames = []
        
#         def appendConvComponent(i, filters=128, kernel_size=4, dilation_rate=(1, 1), strides=(1, 1), groups=1):
#             conv = Conv2D(filters, kernel_size, strides=strides, dilation_rate=dilation_rate, groups=groups, activation='relu', padding='same')
#             pool = MaxPool2D(pool_size=(2, 1), padding='same')
#             setattr(self, f'conv{i}', conv)
#             setattr(self, f'pool{i}', pool)
#             self._modnames.extend([f'conv{i}', f'pool{i}'])
        
#         def appendHiddenLayer(i, size):
#             setattr(self, f'fc{i}', Dense(size))
#             self._modnames.append(f'fc{i}')
            
#         # self._modnames.append('reshape')
#         # self.reshape = Reshape((self.ntimesteps, 1, self.nchannels))
            
#         for i, spec in enumerate(self.convolutions):
#             appendConvComponent(i, **spec)
#         self.flat = Flatten()
#         self._modnames.append('flat')
#         for i, size in enumerate(self.fc_hidden_layer_sizes):
#             appendHiddenLayer(i, size)
#         # self.conv2 = Conv2D(3, 1, padding='same', activation='relu')
#         self.f = Dense(1, activation='relu')
#         self._modnames.append('f')
        
#         self._call = compcall(*self._modnames)
    
#     def call(self, inputs, training=None):
#         return self._call(self, inputs, training=training)

# #loss_tracker = keras.metrics.MeanSquaredError(name='mse')
# #mae_metric = keras.metrics.MeanAbsoluteError(name="mae")

# cell_classes = [OHLCCell, OHLCCell1D, RecurrentOHLCCell]

# class TotemicOHLC(Model):
#    def __init__(self, name='totemic_ohlc', hyperparams=None):
#       super().__init__(name='totemic_ohlc')
      
#       self.hyperparams = hyperparams
#       hp = self.hyperparams
      
#       self.n_steps = hp.n_steps
#       self.n_outs = len(hp.target_columns)
#       self.n_steps_fwd = hp.n_steps_fwd
#       self.channels = hp.feature_columns[:]
      
#       self.in_reshape = Reshape((self.n_steps, 1, len(hp.feature_columns)))
#       self.conv1 = Conv2D(16, 2, strides=1, activation='relu', padding='same')
#       self.pool1 = MaxPool2D(pool_size=(2, 1), padding='same')
#       self.flat = Flatten()
#       # self.hl1 = Dense(1152, activation='relu')
#       self.hl1 = Dense(200, activation='relu')
#       self.out = Dense((self.n_steps_fwd * self.n_outs))
#       self.out_reshape = Reshape((self.n_steps_fwd, self.n_outs))
      
#       self._call = compcall('in_reshape', 'conv1', 'pool1', 'flat', 'hl1', 'out')
      
#    def save(self, path):
#       super().save(path)
#       with open(P.join(path, 'hyperparams'), 'wb+') as f:
#          pickle.dump(self.hyperparams, f)
      
      
#    def call(self, inputs, training=None):
#       outputs = self._call(self, inputs, training=training)
#       if self.n_steps_fwd > 1:
#          outputs = self.out_reshape(outputs)
#       return outputs
   
#    def preprocess(self, x, y):
#       return x, y

# class OHLC(Model):
#    def __init__(self, name='ohlc', celltype=0, hyperparams=None, **kwargs):
#       super().__init__(name=name)
#       self.hyperparams = hyperparams
#       hp = hyperparams
#       self.ntimesteps = hp.n_steps
#       self.channels = hp.feature_columns[:]
#       self.celltype = celltype
      
#       if self.celltype < 2:
#           self.reshape = Reshape((hp.n_steps, 1, len(self.channels)), dtype='float64')
#       cellcls = cell_classes[self.celltype]
#       self.cells = [cellcls(ntimesteps=self.ntimesteps, nchannels=1, name=k, dtype='float64') for k in self.hyperparams.target_columns]
#       self.join = Concatenate(dtype='float64')
#       self.f = Dense(len(hp.target_columns), activation='sigmoid', dtype='float64')
   
#    def save(self, path):
#       super().save(path)
#       with open(P.join(path, 'hyperparams'), 'wb+') as f:
#          pickle.dump(self.hyperparams, f)
      
#    def preprocess(self, x, y):
#       return x, y
      
#    @tf.function(jit_compile=True)
#    def call(self, inputs, training=None, mask=None):
#       inputs = tf.cast(inputs, tf.float64)
#       if self.celltype < 2:
#           inputs = self.reshape(inputs)
#       outs = [cell(inputs, training=training) for cell in self.cells]
#       x = self.join(outs)
#       return self.f(x)
     
#    def piss_ass(self, data):
#       # Unpack the data. Its structure depends on your model and
#       # on what you pass to `fit()`.
#       if len(data) == 3:
#          x, y, sample_weight = data
#       else:
#          sample_weight = None
#          x, y = data

#       with tf.GradientTape() as tape:
#          y_pred = self(x, training=True)  # Forward pass
#          # Compute the loss value.
#          # The loss function is configured in `compile()`.
#          loss = self.compiled_loss(
#                y,
#                y_pred,
#                sample_weight=sample_weight,
#                regularization_losses=self.losses,
#          )

#       # Compute gradients
#       trainable_vars = self.trainable_variables
#       gradients = tape.gradient(loss, trainable_vars)

#       # Update weights
#       self.optimizer.apply_gradients(zip(gradients, trainable_vars))

#       # Update the metrics.
#       # Metrics are configured in `compile()`.
#       self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)

#       # Return a dict mapping metric names to current value.
#       # Note that it will include the loss (tracked in self.metrics).
#       return {m.name: m.result() for m in self.metrics}
  
# class OHLC2(Model):
#     def __init__(self, in_columns=None, out_columns=None, ntimesteps=14, **kwargs):
#         super().__init__(**kwargs)
#         self.in_columns = in_columns
#         self.out_columns = out_columns
#         self.ntimesteps = ntimesteps
        
#         assert (None not in [self.in_columns, self.out_columns])
        
#         self.cells = []
#         for name in self.in_columns:
#             cell = OHLCCell1D(self.ntimesteps)
#             setattr(self, name, cell)
#             # cell.build()
#             self.cells.append(cell)
            
#         self.join = Concatenate()
#         self._preprocess = None
        
#     # def build(self, input_shape):
#     #     print(f"Building model with input_shape={input_shape}")
#     #     super().build(input_shape)
        
#     def preprocess(self, x, y):
#         if callable(self._preprocess):
#             return self._preprocess(x, y)
#         return x, y
        
#     def call(self, inputs, training=None):
#         inputs = tf.cast(inputs, 'float32')
#         outputs = []
        
#         for i, cell in enumerate(self.cells):
#             cell_name = self.in_columns[i]
#             cell_input = inputs[i]
#             # print(cell_input)
#             # cell_input = tf.reshape(cell_input, (cell_input.shape[0], 1, cell_input.shape[1]))
#             # print(cell_name, cell_input)
#             cell_output = cell(cell_input, training=training)
#             outputs.append(cell_output)
            
#         out = self.join(outputs)
#         return out

# class FlatOHLC(Model):
#     def __init__(self, ntimesteps=14, channels=None, name=None):
#         super().__init__(name=name)
#         self.ntimesteps = ntimesteps
#         nchannels = self.nchannels = len(channels)
#         self.channels = channels
        
#         self.reshape = Reshape((ntimesteps, 1, nchannels))
#         self.conv1 = Conv2D(32, 2, strides=(2, 1), activation='relu', padding='same')
#         self.pool = MaxPool2D(pool_size=(2, 1), padding='same')
#         self.conv2 = Conv2D(16, 2, strides=(2, 1), activation='relu', padding='same')
#         self.pool2 = MaxPool2D(pool_size=(2, 1), padding='same')
#         # self.conv2 = Conv2D(3, 1, padding='same', activation='relu')
#         # self.conv3 = Conv2D(8, 2, strides=(2, 1), activation='relu', padding='same')
#         self.flat = Flatten()
#         self.dense = Dense(900)
#         self.f = Dense(nchannels, activation='relu')
        
        
#     def preprocess(self, x, y):
#         return x, y

#     def call(self, inputs, training=None, **kw):
#         x = inputs
#         x = self.reshape(x)
#         x = self.conv1(x, training=training)
#         x = self.pool(x, training=training)
#         x = self.conv2(x, training=training)
#         x = self.pool2(x, training=training)
#         x = self.flat(x, training=training)
#         x = self.dense(x, training=training)
#         return self.f(x, training=training)
  
# classes = [FlatOHLC, OHLC, OHLCCell, OHLCCell1D, Cell]

# custobjs = {c.__name__:c for c in classes}
