import math
import pickle
import shelve
import toolz.dicttoolz as dicts
#from keras.engine import training_utils
#from keras.engine import input_layer as input_layer_module
#from keras.engine import data_adapter
#from keras.engine import compile_utils
#from keras.engine import base_layer_utils
#from keras.engine import base_layer
from .model_train import *
import matplotlib.pyplot as plt
import collections
import pickletools as pt
import warnings
##from tensorflow.keras.optimizers import *
##from tensorflow.keras.models import clone_model, model_from_json
##from tensorflow.keras.layers import *
##from tensorflow.keras import Sequential, Model
##from tensorflow.keras import *
##import tensorflow.keras as keras
#import tensorflow as tf
from ds.maths import *
import ds.data as dsd
from ds.gtools import ojit, Struct, mpmap, njit, jit, vprint, TupleOfLists
from ds.common import *
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from numpy import *
import random
import numpy as np
import os
P = os.path
from functools import partial

## keras = tf.keras


tf.config.optimizer.set_jit(True)

# def napply(nn, steps:int=4, inputs:Tensor):
#    # import 
#    pass


def conv_ae(seq_len=10, n_features=4, n_features_out=4, n_filters=4, hidden=400, droprate=None, kernel=None, **kwargs):
   convkw = dict(padding='same', activation='relu')
   hidden_expansion_factor = kwargs.pop('hidden_expansion_factor', None)
   crossways = kwargs.pop('crossways', False)
   
   if kernel is None:
      kernel = lambda x: 2
   
   elayers = [
      
   ]
   if isinstance(n_filters, tuple):
      if crossways:
         n_filters = iter(n_filters)
         elayers.append(Conv2D(next(n_filters), kernel_size=(n_features, 2), **convkw))
         for filter_size in n_filters:
            elayers.append(Conv2D(filter_size, kernel_size=2, **convkw))
      else:
         for filter_size in n_filters:
            elayers.append(Conv2D(filter_size, kernel_size=2, **convkw))
   else:
      conv = Conv2D(n_filters, kernel_size=(2 if not crossways else (n_features, 2)), **convkw)
      elayers.append(conv)
   
   elayers.extend([
      AvgPool2D(pool_size=(2, 2), padding='same'),
      Flatten()
   ])
   
   if droprate is not None:
      elayers.insert(2, Dropout(droprate))
   
   encode = Sequential(elayers)
   
   dlayers = []
   
   dexpand = lambda d, n: tuple(n//d for x in range(d))
   dexpand = partial(dexpand, hidden_expansion_factor)
   expand_dense = (hidden_expansion_factor is not None)
   
   if isinstance(hidden, tuple):
      for size in hidden:
         if expand_dense:
            dlayers.extend([Dense(n) for n in dexpand(size)])
         else:
            dlayers.append(Dense(size))
   else:
      if expand_dense:
         dlayers.extend([Dense(n) for n in dexpand(hidden)])
      else:
         dlayers.append(Dense(hidden))
   dlayers.append(Dense(n_features_out, activation='relu'))
   
   decode = Sequential(dlayers)
   
   inputs = Input(shape=(seq_len, n_features, 1))
   outputs = decode(encode(inputs))
   
   model = Model(inputs=[inputs], outputs=[outputs])
   
   # model = Sequential([encode, decode])
   
   return model


def cnnpred_deep2d(seq_len=10, n_features=4, n_features_out=4, n_filters=(4, 4), hidden=200, droprate=0.001):
    #TODO evaluate this model's performance against the previous revision
    #! ALWAYS update the model architecture this way, so that the best performer is never lost again
    convkw = dict(padding='same', activation='relu')
    model = Sequential([
        Input(shape=(seq_len, n_features, 1)),
        Conv2D(n_filters[0], kernel_size=(n_features, 2), **convkw),
        AvgPool2D(pool_size=(2, 2), padding='same'),
        Flatten(),
        Dense(hidden),
        Dense(n_features_out, activation='relu')
    ])

    return model


def cnnpred_bigdeep2d(seq_len=10, n_features=4, n_features_out=4, n_filters=(4, 4), hidden=900, droprate=0.001):
    #TODO evaluate this model's performance against the previous revision
    #! ALWAYS update the model architecture this way, so that the best performer is never lost again
    convkw = dict(padding='same', activation='relu')
    model = Sequential([
        Input(shape=(seq_len, n_features, 1)),
        Conv2D(n_filters[0], kernel_size=(2, 2), **convkw),
        AvgPool2D(pool_size=(2, 2), padding='same'),
        Flatten(),
        Dense(hidden),
        Dense(n_features_out, activation='relu')
    ])

    return model
