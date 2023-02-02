
import numpy as np
from numpy import *
import pandas as pd
#import modin.pandas as pd

# from sklearn.base import BaseEstimator, TransformerMixin, _OneToOneFeatureMixin
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import MinMaxScaler as MMS
# from sklearn.preprocessing import QuantileTransformer as Quant
from sklearn.utils.validation import (
    check_is_fitted,
    check_random_state,
    _check_sample_weight,
    FLOAT_DTYPES,
)

from itertools import chain, repeat
from functools import *
from collections import deque

from numba import *
from numba import types
from ds.gtools import ojit, Struct
import pandas as pd
#import modin.pandas as pd
from tools import *

BOUNDS_THRESHOLD = 1e-7

def clean_dataframe(df:pd.DataFrame, columns=None):
    if 'date' not in df.columns:
        if 'time' in df.columns:
            df.rename(columns={'time': 'date'}, inplace=True)
            # if isinstance(df.index, pd.DatetimeIndex):
        df['date'] = df.index
            
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.dropna(inplace=True)
    
    if columns is not None:
        curcols = set(df.columns.tolist())
        df.drop(list(curcols - set(columns)), axis=1, inplace=True)
    
    return df

def future_dataframe(df:pd.DataFrame, target_columns, lookup_step=1):
    for c in target_columns:
        df[f'future_{c}'] = df[c].shift(-lookup_step)
    # df.drop(index=df.index[0], axis=0, inplace=True)
    df.drop(df.tail(1).index, inplace=True)
    return df


#? got 'em! runs substantially (like, REALLY substantially) faster than the legacy sample-computation function
@jit(cache=True)
def _seqfrom(seqs:float32[:, :], targets:float32[:, :], nsteps:int):
    rows:int = seqs.shape[0]
    cols:int = seqs.shape[1]
    
    X = np.zeros((rows-nsteps, nsteps, cols))
    Y = np.zeros((rows-nsteps, targets.shape[1]))
    
    for i in prange(nsteps, rows):
        x = seqs[i-nsteps:i, :]
        y = targets[i+1, :]
        
        X[i-nsteps, :, :] = x
        Y[i-nsteps, :] = y
        
    return X, Y

def dffunc(func):
    jitted_func = jit(cache=True, nopython=True)(func)
    @wraps(jitted_func)
    def wrapper(df:pd.DataFrame, *args, **kwargs):
        ndf:ndarray = df if isinstance(df, ndarray) else df.values
        return jitted_func(*(ndf, *args), **kwargs)
    return wrapper

# @jit(cache=True)
@dffunc
def feature_seqs(df:float32[:, :], n_steps:i4):
    rows:i4 = df.shape[0]
    cols:i4 = df.shape[1]
    
    X:float32[:,:,:] = np.zeros((rows-n_steps, n_steps, cols))
    for i in prange(n_steps, rows):
        X[i-n_steps, :, :] = df[i-n_steps:i]
    return (n_steps, 1), X

# @jit(cache=True)
@dffunc
def target_seqs(df:float32[:, :], n_steps:i4):
    rows:i4 = df.shape[0]
    cols:i4 = df.shape[1]
    y:float32[:, :] = np.zeros((rows-n_steps, cols))
    for i in prange(n_steps, rows):
        y[i-n_steps, :] = df[i, :]
    return (n_steps, 0), y

def sequences_from(df:pd.DataFrame, feature_columns, target_columns, n_steps, lookup_step=1):
    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs

    future_columns = ['future_{}'.format(c) for c in target_columns]
    
    #? our sample pairs
    left = df[feature_columns].values
    right = df[future_columns].values

    return _seqfrom(left, right, n_steps)

def df2seqs(df, xcols=None, ycols=None, n_steps=14):
    xcols = xcols if xcols is not None else 'open high low close'.split(' ')
    ycols = ycols if ycols is not None else xcols
    zcols = ['future_{}'.format(c) for c in ycols]#+["date"]
    df = df.copy()
    clean_dataframe(df)
    last = df[xcols].tail(n_steps).values
    future_dataframe(df, ycols)
    clean_dataframe(df, list(set(xcols+ycols+zcols)))
    
    return sequences_from(df, xcols, ycols, n_steps)


def scale_dataframe(df:pd.DataFrame, columns=None, scaler=None):
    grouped = False
    scalers = {}
    scaler = scaler if scaler is not None else QuantileTransformer
    for k in columns:
        if isiterable(k) and not isinstance(k, str):
            grouped = True
            break
    
    for c in columns:
        if isinstance(c, str):
            sc = scalers[c] = scaler()
            col = df[c].values
            scaled_val = sc.fit_transform(col.reshape(-1, 1))[:, 0]
            df[c] = scaled_val
            
        elif isiterable(c):
            sc = scaler()
            cols = list(iter(c))
            vals = df[cols].values.T
            scaled_vals = sc.fit_transform(vals)
            for i,_c in enumerate(c):
                df[_c] = scaled_vals[i,:]
            for k in c: 
                scalers[k] = sc
    return scalers

import inspect


                
def compile_scalers(scalers, columns):
    routines = []
    def part(i, s, X:ndarray, inverse=False):
        col = X[i]
        col = (s.transform if not inverse else s.inverse_transform)(
            col.reshape(-1, 1))[:, 0]
        X[i] = col
    
    for i, c in enumerate(columns):
        f = partial(part, i, scalers[c])
        print(inspect.signature(f))
        routines.append(f)
    
    def cscale(subroutines, X:ndarray, inverse=False):
        X = X.copy()
        for f in subroutines:
            f(X, inverse=inverse)
        return X
    return partial(cscale, routines)
