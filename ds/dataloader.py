from ds.common import *
from ds.datasets import load_dataset, all_available_symbols, Stock, Crypto
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from sklearn.model_selection import train_test_split
from numpy import *
from .datatools import *
from ds.gtools import Struct, njit, ojit, vprint, once, TupleOfLists, mpmap
from ds.maths import *
from ds import t
# from ds.model.spec import Hyperparameters
# from ds.model.dp import ShapeDescription as ShapeDesc

import os
P = os.path
import shelve
# import .fd as fd
# from .fd import FileDict
from functools import lru_cache, partial, singledispatch, singledispatchmethod
from typing import *
from datetime import datetime, timedelta

from pymitter import EventEmitter
from cytoolz import *
# from engine.data.scaling_base import *
from tools import unzip

def identity(*x):
   return tuple(x) if len(x) > 1 else x[0]


@persistent_memoize
def memoized_load(spec, name, *args, **kwargs):
   l = DataLoader(hparams=spec, **kwargs)
   dlr = l.load(name)
   return dlr

class DataLoaderParams:
   def __init__(self, **kwargs):
      # super().__init__(**kwargs)
      self.params = kwargs
      # self.prev_params = None
      
   def apply(self, dl):
      ctx = DataLoaderContext(self, dl)
      return ctx
   
   def write(self, dl):
      for k, v in self.params.items():
         setattr(dl, k, v)
      
      
class DataLoaderContext:
   def __init__(self, params, dl):
      self.params = params
      self.prev_params = {}
      self.dl = dl
         
   def __enter__(self):
      self.prev_params = {}
      for k, v in self.params.params.items():
         self.prev_params[k] = getattr(self.dl, k)
         setattr(self.dl, k, v)
         
   def __exit__(self, exc_type, exc_val, exc_tb):
      for k, v in self.prev_params.items():
         setattr(self.dl, k, v)
         
class DataLoaderResult:
   def __init__(self, **kwds):
      self.ticker = None
      self.df = None
      self.test_df = None
      self.X_train = None
      self.y_train = None
      self.X_test = None
      self.y_test = None
      self.column_scalers = None
      self.last_sequence = None
      self.split_index = None
      
      self.__dict__.update(kwds)
      
   def copy(self):
      return DataLoaderResult(**self.__dict__)
   
   def extend(self, ext):
      p = self.__dict__.copy()
      p.update(ext)
      return DataLoaderResult(**p)
      
   def eget(self, *keys):
      return tuple(getattr(self, k) for k in keys)
      

class DataLoader(EventEmitter):
   def __init__(self, hparams=None, **params):
      super().__init__()
      
      if hparams is not None:
         # assert isinstance(hparams, Hyperparameters)
         params = merge(hparams.to_kwds(), params)
         
         # self.on('df', hparams.apply_mods)
      
      self.n_steps = 50
      self.n_steps_fwd = 1
      self.lookup_step = 1 
      self.timescale_unit = '1D'
      self.feature_columns = ['open', 'low', 'high', 'close']
      self.target_columns = self.feature_columns[:]
      
      self.scale = True 
      self.shuffle = True 
      self.split_by_date = True
      self.strict = True
      self.test_size = 0.2
      self.length_threshold = 1000
      self.nan_pct_limit = 0.8
      self.Scaler = MinMaxScaler
      self._scale_columns_individually = True
      self._scaling_columns = None #? when set, this would be the 'columns' argument passed to t.scale_dataframe
      self.batch_scaling = False
      self.batch_size = 5 #? this is actually the coefficient by which `n_steps` is multiplied to compute the actual batch size
      self.fex = None
      self.delta_mode = False
      # DataLoaderParams(**params).write(self)
      
      # self.df = None
      # self.sdf = None
      # self._scaled = False
      # self.Scaler = MinMaxScaler
      self._on_column = identity
      self._on_before_sample_gen = identity
      self._on_sample = identity
      # self._pre_df = identity
      self._dfmaps = []
      if hparams is not None:
         self._dfmaps.append(hparams.apply_mods)
      self._postprocess = identity
      
      self.current_symbol = None
      
      attrs = list(self.__dict__.keys())
      for k, v in params.items():
         if k in attrs:
            setattr(self, k, v)
         
         elif callable(v) and f'_{k}' in attrs:
            
            getattr(self, f'{k}')(v)
            
      if self.fex is not None:
         if self.delta_mode:
            self.fex = self.fex.extend(head=self._diff, tail=self._patch)
         
   def _diff(self, df:pd.DataFrame):
      if not self.delta_mode:
         raise ValueError('should not even be called')
      
      _df = df
      df = df.copy(deep=True)
      for c in self.target_columns:
         df[c] = (df[c] - df[c].shift())
      df = df.dropna()
      tR = {}
      self._scaledf(df, tR)
      self._scalers = tR["column_scaler"]
      # print(df)
      return df
   
   def _patch(self, rdf:pd.DataFrame):
      # if self._scalers is not None:
      #    rdf['_scalers'] = [self._scalers]*len(rdf)
      return rdf
      
   def scaler(self, Scaler):
      self.Scaler = Scaler
      return self
   
   def x(self, *columns):
      self.feature_columns = list(columns)
      return self
   
   def y(self, *columns):
      self.target_columns = list(columns)
      return self
   
   def context(self, **params):
      params = DataLoaderParams(**params)
      return params.apply(self)
   
   def on_column(self, func):
      self._on_column = func
      return self
   
   def on_sample(self, func):
      self._on_sample = func
      return self
   
   def moddf(self, func):
      self._dfmaps.append(func)
      return self
   
   def _pre_df(self, df:pd.DataFrame):
      for f in self._dfmaps:
         r = f(df)
         if isinstance(r, pd.DataFrame):
            df = r
      # self._pre_df = func
      df = df.dropna()
      return df
   
   def postprocess(self, func):
      self._postprocess = func
      return self
   
   def _scaledf(self, df:pd.DataFrame, result):
      # types = df.dtypes.to_dict()
      if self.scale:
         allcols = set(self.feature_columns + self.target_columns)
         allcols &= set(df.dtypes[df.dtypes == np.float64].index)
         allcols = list(allcols)
         if not self._scale_columns_individually:
            column_scaler = {}
            
            common_scaler = self.Scaler()
            # scale the data (prices) from 0 to 1
            # scale_columns = np.unique(feature_columns[:] + target_columns[:]).tolist()
            # scale_columns = [k for k in types.keys() if types[k] == np.float64]
            values = df[allcols].values
            scaled_values = common_scaler.fit_transform(values).T
            
            for i,col in enumerate(allcols):
               df[col] = scaled_values[i, :]
               column_scaler[col] = common_scaler

            for column in allcols:
               scaler = common_scaler
               # df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
               df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))[:, 0]
               column_scaler[column] = scaler

            # add the MinMaxScaler instances to the result returned
            result["column_scaler"] = column_scaler
         else:
            scalers = t.scale_dataframe(df, 
               self._scaling_columns if self._scaling_columns is not None else allcols, 
               scaler=self.Scaler
            )
            result["column_scaler"] = scalers

         # schweeeeet
         self._scalers = result.get('column_scaler', None)
         
   def transform_column(self, df:pd.DataFrame, column:str, column_data:pd.Series):
      return None
         
   def import_df(self, df:pd.DataFrame, wholestate=False):
      R = {}
      
      self.emit('df', df)

      feature_columns = list(self.feature_columns)
      target_columns = list(self.target_columns)
      n_steps = self.n_steps
      n_forward_steps = self.n_steps_fwd
      test_size = self.test_size

      # we will also return the original dataframe itself
      R['df'] = df
      df = df.copy()
      
      # df = df.fillna(method='ffill')

      # add date as a column
      if "date" not in df.columns:
         df["date"] = df.index
      df = df.reset_index()
      df.dropna(inplace=True)

      df = df.reset_index(drop=True)
      df = df.fillna(method='ffill').fillna(method='bfill')
      df = df.dropna()
      df.index = df.date
      df['date'] = df.index

      zeroshift = False
      for c in self.target_columns:
         if df[c].values[0] == 0:
            zeroshift = True
            break
      if zeroshift:
         df = df.iloc[1:].copy(deep=True)
         R['df'] = df

      for c in self.target_columns:
         column = df[c].values
         column = self._on_column(column)
         #TODO write custom delta function which works around the division-by-zero problem
         # column_delta = divdelta(column)
         # df[c] = column_delta
         df[c] = column

      # drop the first row of the DataFrame
      df = df.iloc[1:].copy(deep=True)

      df = self._pre_df(df)

      # make sure that the passed feature_columns exist in the dataframe
      for col in feature_columns:
         df[col] = self._on_column(df[col].values)
         assert col in df.columns, f"'{col}' does not exist in the dataframe."

      # apply scaling to the DataFrame
      self._scaledf(df, R)

      for c in target_columns:
         df[f'future_{c}'] = df[c].shift(-self.lookup_step)
      df = df.iloc[:-1].copy()
      if df.isna().sum().sum() > 0:
         print(df.isna())

      R['df'] = df
      # last `lookup_step` columns contains NaN in future column
      # get them before droping NaNs
      last_sequence = np.array(df[feature_columns].tail(self.lookup_step))

      #TODO loops A & B by converting sequence_data to TupleOfLists
      

      future = None
      trim_steps = 0

      if n_forward_steps > 1:
         future = []

         for i in range(len(df)):
            fut = df[future_columns].iloc[i:i+n_forward_steps].values

            if output_shape is None:
               output_shape = fut.shape

            if len(fut) < n_forward_steps:
               trim_steps = (len(df) - i)
               break

            future.append(fut)
            
      df = self._on_before_sample_gen(df)
      if wholestate:
         return locals()
      return df

   def samples_from(self, state):
      #!============
      #TODO optimize the same way that .load() has been optimized
      #!============
      
      sequence_data = []
      sequences = deque(maxlen=self.n_steps)
      test_size = self.test_size
      shuffle = self.shuffle

      future_columns = ['future_{}'.format(c) for c in self.target_columns]
      input_shape, output_shape = None, None
      
      R, df, trim_steps, feature_columns, last_sequence = dunpack(state, 'R', 'df', 'trim_steps', 'feature_columns', 'last_sequence')
      
      def left(a): return (a[:-trim_steps] if trim_steps > 0 else a)

      #? our sample pairs
      et = zip(
         left(df[feature_columns + ["date"]].values),
         df[future_columns].values
      )

      for entry, target in et:
         sequences.append(entry)
         if len(sequences) == self.n_steps:
            sequence_data.append([
                  array(sequences),
                  target
            ])

      # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
      # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
      # this last_sequence will be used to predict future stock prices that are not available in the dataset
      last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
      last_sequence = np.array(last_sequence).astype(np.float64)
      R['last_sequence'] = last_sequence

      # construct the X's and y's
      X, y = [], []
      for seq, target in sequence_data:
         seq, target = self._on_sample(seq, target)
         if len(seq[seq == np.nan]) > 0 or len(target[target == np.nan]) > 0:
            raise ValueError('nan encountered in X,y set; u stoopid')

         X.append(seq)
         y.append(target)

      # convert to numpy arrays
      X, y = np.array(X), np.array(y)

      result = R
      if self.split_by_date:
         # split the dataset into training & testing sets by date (not randomly splitting)
         train_samples = int((1 - test_size) * len(X))
         result['split_index'] = train_samples
         result["X_train"] = X[:train_samples]
         result["y_train"] = y[:train_samples]
         result["X_test"] = X[train_samples:]
         result["y_test"] = y[train_samples:]
         if self.shuffle:
            # shuffle the datasets for training (if shuffle parameter is set)
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
      else:
         # split the dataset randomly
         result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(
               X, y, test_size=test_size, shuffle=shuffle)

      dates = result["X_test"][:, -1, -1]

      # retrieve test features from the original dataframe
      result["test_df"] = result["df"].loc[dates]

      # remove duplicated dates in the testing dataframe
      result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(
            keep='first')]

      # remove dates from the training/testing sets & convert to float32
      preshape = result['X_train'].shape
      result["X_train"] = result["X_train"][:, :,
                                             :len(feature_columns)].astype(np.float32)
      result["X_test"] = result["X_test"][:, :,
                                          :len(feature_columns)].astype(np.float32)

      assert result['X_train'].shape[-1] == len(feature_columns)

      # result['ticker'] = ticker
      if self._postprocess is not None:
         result = self._postprocess(result)

      r = DataLoaderResult(**R)
      self.emit('result', r)
      return r
         
   def load(self, ticker):
      R = {}
      # see if ticker is already a loaded stock from yahoo finance
      if isinstance(ticker, str):
         # load it from yahoo_fin library
         df = load_dataset(ticker, resampling=self.timescale_unit)
         R['ticker'] = ticker
         self.current_symbol = ticker

      elif isinstance(ticker, pd.DataFrame):
         # already loaded, use it directly
         df = ticker

      else:
         raise TypeError(f"ticker can be either a str or a `pd.DataFrame` instances, got {type(ticker)} instead")

      if len(df) < self.length_threshold:
         return None
      
      self.emit('df', df)
      feature_columns = list(self.feature_columns)
      target_columns = list(self.target_columns)
      n_steps = self.n_steps
      n_forward_steps = self.n_steps_fwd
      test_size = self.test_size
      
      if self.strict:
         for col in feature_columns:
            #? reject if more than 15% of any one column is filled with null values
            pct_null = (df[col].isnull().sum()/len(df))
            if pct_null > self.nan_pct_limit:
               return None

      # we will also return the original dataframe itself
      R['df'] = df
      # df = df.fillna(method='ffill')

      t.clean_dataframe(df, list(set(feature_columns + target_columns + ['date'])))
      
      # get most recent date
      mr = df['date'].max()
      sincelastsample = datetime.now() - mr
      if sincelastsample >= timedelta(days=(365 * 5)):
         return None

      for c in set(target_columns + feature_columns):
         r = self.transform_column(df, c, df[c])
         if r is None:
            continue
         elif isinstance(r, (ndarray, pd.Series)):
            df[c] = r

      # if self._pre_df is not None:
      df = self._pre_df(df)
      
      # make sure that the passed feature_columns exist in the dataframe
      for col in feature_columns:
         assert col in df.columns, f"'{col}' does not exist in the dataframe."

      # apply scaling to the DataFrame
      if self.batch_scaling:
         
         cols = list(set(feature_columns + target_columns))
         ndoc = df[cols].to_numpy()
         batch_size = (self.n_steps * self.batch_size)
         # batches = group_list(ndoc, batch_size)
         # sbatches = []
         # batch_ranges = []
         
         # for g in batches:
         #    sr, sg = rowRollingScale(g, target_range=(0, 1))
         #    sbatches.append(sg)
         #    batch_ranges.append(sr)
         # scaled_ndoc:np.ndarray = np.concatenate(sbatches, axis=0)
         sample_ranges, scaled_ndoc = batch_scale(ndoc, batch_size)
         print('batch-scaled the dataFrame')
         # print(scaled_ndoc)
         scaled_cols = scaled_ndoc.T
         scaled_df:pd.DataFrame = df.head(len(scaled_ndoc)).copy(deep=True)
         for i,name in enumerate(cols):
            scaled_df[name] = scaled_cols[i]
         R['usdf'] = df
         df = R['df'] = scaled_df
         # print(scaled_df)
      else:
         if len(df) <= self.length_threshold:
            return None
         #? pussy
         self._scaledf(df, R)
      
      if self.fex is None:
         self._scaledf(df, R)
         t.future_dataframe(df, target_columns)
         
         if df.isna().sum().sum() > 0:
            print(df.isna())
         
         R['df'] = df
         # last `lookup_step` columns contains NaN in future column
         # get them before droping NaNs
         X, y = t.sequences_from(df, feature_columns, target_columns, n_steps)
         # rX, X = unzip(map(rowRollingScale, X))
         # rX, X = list(rX), list(X)
         # ry, y = unzip(map(rowRollingScale, y))
         # ry, y = list(ry), list(y)
         # X, y = np.asanyarray(X), np.asanyarray(y)
         # print(y)
         
         if True:
            # split the dataset into training & testing sets by date (not randomly splitting)
            train_samples = int((1 - test_size) * len(X))
            
            dates = df['date'].iloc[train_samples:]
            R['split_index'] = train_samples
            R["X_train"] = X[:train_samples]
            R["y_train"] = y[:train_samples]
            R["X_test"] = X[train_samples:]
            R["y_test"] = y[train_samples:]
            
            if self.shuffle:
               # shuffle the datasets for training (if shuffle parameter is set)
               shuffle_in_unison(R["X_train"], R["y_train"])
               shuffle_in_unison(R["X_test"], R["y_test"])
         else:
            # split the dataset randomly
            R["X_train"], R["X_test"], R["y_train"], R["y_test"] = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
            dates = R["X_test"][:, -1, -1]

         # retrieve test features from the original dataframe
         R["test_df"] = R["df"].loc[dates]

         # remove duplicated dates in the testing dataframe
         R["test_df"] = R["test_df"][~R["test_df"].index.duplicated(keep='first')]

         # remove dates from the training/testing sets & convert to float32
         preshape = R['X_train'].shape
         R["X_train"] = R["X_train"][:, :, :len(feature_columns)].astype(np.float32)
         R["X_test"] = R["X_test"][:, :, :len(feature_columns)].astype(np.float32)

         assert R['X_train'].shape[-1] == len(feature_columns)

         R['ticker'] = ticker
         if self._postprocess is not None:
            R = self._postprocess(R)
      
      else:
         R['df'] = df
         try:
            rdf:pd.DataFrame = self.fex(df)
            
            R['column_scaler'] = self._scalers
            self._scalers = None
            
            R['rdf'] = rdf
         
         except Exception as error:
            import traceback as tb
            print('fex failed on:\n', df)
            raise error
         
         npify = F(lambda c: c.tolist()) >> np.asanyarray
         rdf = rdf.dropna()
         X, y = tuple(map(npify, [rdf.X, rdf.y]))
         # print(X)
         # print(y)
         train_samples = int((1 - test_size) * len(X))
         res = (lambda k, v: R.__setitem__(k, v))
         # raise ValueError('my balls are full')
         dates = pd.Series(rdf.dropna().index).iloc[train_samples:]
         res('date', dates)
         x_train = R["X_train"] = X[:train_samples]
         y_train = R["y_train"] = y[:train_samples]
         x_test = R["X_test"] = X[train_samples:]
         y_test = R["y_test"] = y[train_samples:]
         if self.shuffle:
            shuffle_in_unison(x_train, y_train)
            shuffle_in_unison(x_test, y_test)
         R['ticker'] = ticker
      
      r = DataLoaderResult(**R)
      self.emit('result', r)
      self.current_symbol = None
      return r
   
from engine.data.scaling_base import rowRollingScale

@jit(cache=True)
def batch_scale(usarr:np.ndarray, batch_size:int)->np.ndarray:
   batches = list(group_list(usarr, batch_size))
   sbatches = np.empty((len(batches), batch_size, *usarr.shape[1:]))
   batch_ranges = np.empty((len(batches), 2))

   for i, g in enumerate(batches):
      sr, sg = rowRollingScale(g, target_range=(0, 1))
      batch_ranges[i] = sr
      sbatches[i, ...] = sg
   
   scaled_ndoc:np.ndarray = np.concatenate(sbatches, axis=0)
   
   return batch_ranges, scaled_ndoc
   
@jit(cache=True)
def group_list(a:np.ndarray, group_size:int):
   """
   :param l:           list
   :param group_size:  size of each group
   :return:            Yields successive group-sized lists from l.
   """
   for i in range(0, len(a), group_size):
      if  (i + group_size) > len(a):
         break
      yield a[i:i+group_size]

   
chunk = lambda lst, n: [lst[i:i + n] for i in range(0, len(lst), n)]
kwp = lambda kw,k,dv=None: kw.pop(k) if k in kw.keys() else dv
def dunpack(d:dict, *keys):
   return tuple(d[k] for k in keys)

def popparams(kwds, *paramnames, **defaults):
   p = {}
   p.update(defaults)
   for k in paramnames: p[k] = None
   pop = partial(kwp, kwds)
   for k,dv in p.items():
      p[k] = pop(k, dv)
   return p

class DatasetBuilder(EventEmitter):
   def __init__(self, hparams, **kwds):
      super().__init__()
      
      self.hyperparams = hparams
      self.symbols = all_available_symbols()
      self.results = {}
      self.parallel = True
      self._onloader = identity
      self._overridefilter = None
      
      self.blacklist = []#cache.get('blacklist', [])
      self._prioritized = None
      self.fex = None
      self.kwds = kwds
      
      self.__dict__.update(kwds)
      
   def onloader(self, f):
      # self._onloader = f
      self.on('loader.construct', f)
      return self
   
   def prioritize(self, names):
      self._prioritized = list(names)
      return self
   
   def _fwd_events(self, loader:DataLoader):
      loader.on('result', partial(self.emit, 'loader.result', loader))
      
   def _load_step(self, params, symbol):
      return memoized_load(self.hyperparams, symbol)
   
   def dropListeners(self):
      self.off_all()
   
   def loadstep(self, **params):
      symbols = list(set(self.symbols) - set(self.blacklist))
      random.shuffle(symbols)
      
      if self._prioritized is not None:
         symbols = self._prioritized + symbols
      
      if self.parallel:
         symchunks = chunk(symbols, 10)
         for symbols in symchunks:
            
            results = mpmap(self._load_step, symbols, partialargs=(params,))
            res = {symbols[i]: v for i, v in enumerate(results)}
            self.results.update(res)
         
            for k,v in res.items():
               yield k, v
      
      else:
         from tqdm import tqdm
         # prog_symbols = tqdm(symbols)
         loader = self.loader = DataLoader(hparams=self.hyperparams, **self.kwds)
         for sym in symbols:
            # result = memoized_load(self.hyperparams, sym)
            result = loader.load(sym)
            
            self.results[sym] = result
            yield sym, result
               
   def loadsymbol(self, symbol, **params):
      loader = DataLoader(self.hyperparams)
      self._fwd_events(loader)
      self.emit('loader.construct', loader)
      
      with loader.context(self.hyperparams.to_kwds()):
         return loader.load(symbol)

   def filter(self, symbol, result:DataLoaderResult):
      if self._overridefilter is not None:
         return self._overridefilter(result)
      return True
               
               
   def load(self, **params):
      myparams = popparams(params, nsamples=10000, parallel=True, prioritize=None)
      dnsamples, parallel, prior = dunpack(myparams, 'nsamples', 'parallel', 'prioritize')
      exsymbols = params.pop('symbols', None)
      if exsymbols is not None:
         self.symbols = exsymbols
         
      self.parallel = parallel if parallel is not None else self.parallel
      self._prioritized = prior
      
      step = self.loadstep(**params)
      
      nsamples = 0
      results = []
      # acc = TupleOfLists((TupleOfLists(2), TupleOfLists(2)))
      x = TupleOfLists(2)
      y = TupleOfLists(2)
      
      blacklist = self.blacklist
      
      for symbol, result in step:
         if result is None or not self.filter(symbol, result):
            del self.results[symbol]
            blacklist.append(symbol)
            continue
         
         nsamples += len(result.X_train)
         results.append(result)
         x.append(result.X_train, result.X_test)
         y.append(result.y_train, result.y_test)
         
         if nsamples >= dnsamples:
            break
         
      step = None
      # x, y = acc
      x_train, x_test = x.map(flat)
      y_train, y_test = y.map(flat)
      
      return Dataset(
         loader=self,
         parameters=params,
         results=results,
         nsamples=nsamples,
         X_train=x_train,
         y_train=y_train,
         X_test=x_test,
         y_test=y_test
      )
      
   def __getstate__(self):
      self.dropListeners()
      return self.__dict__
      
class Dataset:
   results:List[DataLoaderResult] = None
   modules:Dict[str, DataLoaderResult] = None
   X_train:ndarray = None
   y_train:ndarray = None
   X_test: ndarray = None
   y_test: ndarray = None
   nsamples:int = None

   def __init__(self, **kwds):
      self.results = None
      self.X_train = None
      self.y_train = None
      self.X_test = None
      self.y_test = None
      self.nsamples = None
      
      self.__dict__.update(kwds)
      self.modules = {e.ticker:e for e in self.results}
      
   def copy(self):
      c = Dataset(**self.__dict__)
      c.X_test = np.copy(c.X_test)
      c.X_train = np.copy(c.X_train)
      c.y_test = np.copy(c.y_test)
      c.y_train = np.copy(c.y_train)
      cm = {}
      for k, v in c.modules.items():
         cv = cm[id(v)] = v.copy()
         c.modules[k] = cv
      for i, v in enumerate(c.results):
         c.results[i] = cm[id(v)]
      return c

   def extend(self, ext):
      p = self.__dict__.copy()
      p.update(ext)
      return Dataset(**p)

   def eget(self, *keys):
      return tuple(getattr(self, k) for k in keys)
   
   def xy(self, module=None, set='train'):
      assert set == 'train' or set == 'test'
      if module is None:
         x, y = self.eget(f'X_{set}', f'y_{set}')
      else:
         x, y = self.modules[module].eget(f'X_{set}', f'y_{set}')
      assert len(x) == len(y)

      for i in range(0, len(y)):
         yield i, (x[i], y[i])
      
      
def flat(lol):
   fl = []
   for l in lol:
      fl.extend(l)
   return array(fl)

from fn import F

class DatasetBuffer:
   def __init__(self):
      self.X = []
      self.y = []
      self.split_size = 0.2
      self.dtype = 'float32'
   
   def add(self, *sample):
      if len(sample) == 4:
         features = np.concatenate([sample[0], sample[2]])
         target = np.concatenate([sample[1], sample[3]])
      elif len(sample) == 2:
         features, target = sample
      elif len(sample) == 1:
         features, target = np_sample(sample)
      else:
         raise ValueError('invalid DatasetBuffer')
      # (features, target) = np_sample(sample)
      assert len(features) == len(target)
      
      features, target = features.astype(self.dtype), target.astype(self.dtype)
      
      self.X.append(features)
      self.y.append(target)
      
      return self
   
   def pack(self, shuffle=False):
      npX = np.concatenate(tuple(self.X))
      npy = np.concatenate(tuple(self.y))
      
      if shuffle:
         shuffle_in_unison(npX, npy)
      
      # split the dataset into training & testing sets by date (not randomly splitting)
      train_samples = int((1 - self.split_size) * len(npX))

      X_train = npX[:train_samples]
      y_train = npy[:train_samples]
      X_test  = npX[train_samples:]
      y_test  = npy[train_samples:]
      
      if shuffle:
         shuffle_in_unison(X_train, y_train)
         shuffle_in_unison(X_test, y_test)
      
      return X_train, y_train, X_test, y_test
   

def _tonp(x):
   return np.asarray(x)
np_sample = F() >> (map, _tonp) >> tuple

def quickload(name, test_size=0.1, shuffle=False, **kwargs):
   from ds.model.spec import Hyperparameters
   from engine.data.prep import compileFromHyperparams as extractor
   
   # test_size = kwargs.pop('test_size')
   spec = Hyperparameters(**kwargs)
   fex = extractor(spec)
   # dsl = DatasetBuilder(spec, fex=fex)
   dl = DataLoader(hparams=spec, fex=fex)
   dl.length_threshold = (spec.n_steps + 1)
   
   if isinstance(name, str):
      r = dl.load(name)
      return r.X_train, r.y_train, r.X_test, r.y_test
   else:
      b = DatasetBuffer()
      b.shuffle = shuffle
      if test_size is not None:
         b.split_size = test_size
      for s in name:
         r = dl.load(s)
         if r is None: continue
         sample_batch = (r.X_train, r.y_train, r.X_test, r.y_test)
         b.add(*sample_batch)
      return b.pack(shuffle=True)
