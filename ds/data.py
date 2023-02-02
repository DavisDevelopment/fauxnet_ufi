from ds.common import *
from ds.datasets import load_dataset, all_available_symbols, Stock, Crypto
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from numpy import *
from .datatools import *
from ds.gtools import Struct, njit, ojit, vprint, once
from ds.maths import *

from .dataloader import *

@persistent_memoize
def load_data(ticker, n_steps=50, n_forward_steps=1, scale=True, shuffle=True, lookup_step=1, split_by_date=True,
              test_size=0.2, feature_columns=['close', 'volume', 'open', 'high', 'low'], target_column='close'):
   """
   -
   Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.
   Params:
      ticker (str/pd.DataFrame): the ticker you want to load, examples include AAPL, TESL, etc.
      n_steps (int): the historical sequence length (i.e window size) used to predict, default is 50
      scale (bool): whether to scale prices from 0 to 1, default is True
      shuffle (bool): whether to shuffle the dataset (both training & testing), default is True
      lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
      split_by_date (bool): whether we split the dataset into training/testing by date, setting it 
            to False will split datasets in a random way
      test_size (float): ratio for test data, default is 0.2 (20% testing data)
      feature_columns (list): the list of features to use to feed into the model, default is everything grabbed from yahoo_fin
   """
   # see if ticker is already a loaded stock from yahoo finance
   if isinstance(ticker, str):
      # load it from yahoo_fin library
      df = load_dataset(ticker)

   elif isinstance(ticker, pd.DataFrame):
      # already loaded, use it directly
      df = ticker

   else:
      raise TypeError(
          "ticker can be either a str or a `pd.DataFrame` instances")

   if len(df) < 500:
      return None

   feature_columns = list(feature_columns)
   target_columns = target_column[:] if isinstance(target_column, list) else [target_column]
   
   # this will contain all the elements we want to return from this function
   result = {}
   # we will also return the original dataframe itself
   result['df'] = df.copy()
   # make sure that the passed feature_columns exist in the dataframe
   for col in feature_columns:
      assert col in df.columns, f"'{col}' does not exist in the dataframe."
   
   # add date as a column
   if "date" not in df.columns:
      df["date"] = df.index
   
   if scale:
      column_scaler = {}
      # scale the data (prices) from 0 to 1
      scale_columns = np.unique(feature_columns[:] + target_columns[:]).tolist()
      for column in scale_columns:
         scaler = MinMaxScaler()
         df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
         column_scaler[column] = scaler
      
      # add the MinMaxScaler instances to the result returned
      result["column_scaler"] = column_scaler

   # add the target column (label) by shifting by `lookup_step`
   for tc in target_columns:
      df['future_'+tc] = df[tc].shift(-lookup_step)
   # last `lookup_step` columns contains NaN in future column
   # get them before droping NaNs
   last_sequence = np.array(df[feature_columns].tail(lookup_step))
   
   # drop NaNs
   df.dropna(inplace=True) #! this is problematic for the purity of the samples later on...
   
   sequence_data = []
   sequences = deque(maxlen=n_steps)

   future_columns = ['future_{}'.format(c) for c in target_columns]
   input_shape, output_shape = None, None
   future = None

   trim_steps = 0
   if n_forward_steps > 1 or len(target_columns) > 1:
      future = []
      for i in range(len(df)):
         fut = df[future_columns].iloc[i:i+n_forward_steps].values
         if output_shape is None:
            output_shape = fut.shape

         if len(fut) < n_forward_steps:
            trim_steps = (len(df) - i)
            break

         future.append(fut)

   def left(a): return (a[:-trim_steps] if trim_steps > 0 else a)

   et = zip(
       left(df[feature_columns + ["date"]].values),
       future if future is not None else df[future_columns].values
   )

   for entry, target in et:
      sequences.append(entry)
      if len(sequences) == n_steps:
         sequence_data.append([
             np.array(sequences),
             target
         ])

   # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
   # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
   # this last_sequence will be used to predict future stock prices that are not available in the dataset
   last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
   last_sequence = np.array(last_sequence).astype(np.float32)
   # add to result
   result['last_sequence'] = last_sequence
   
   # construct the X's and y's
   X, y = [], []
   for seq, target in sequence_data:
      X.append(seq)
      y.append(target)
   # convert to numpy arrays
   X = np.array(X)
   y = np.array(y)
   
   if split_by_date:
      # split the dataset into training & testing sets by date (not randomly splitting)
      train_samples = int((1 - test_size) * len(X))
      result['split_index'] = train_samples
      result["X_train"] = X[:train_samples]
      result["y_train"] = y[:train_samples]
      result["X_test"] = X[train_samples:]
      result["y_test"] = y[train_samples:]
      if shuffle:
         # shuffle the datasets for training (if shuffle parameter is set)
         shuffle_in_unison(result["X_train"], result["y_train"])
         shuffle_in_unison(result["X_test"], result["y_test"])
   else:
      # split the dataset randomly
      result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
   # get the list of test set dates
   try:
      dates = result["X_test"][:, -1, -1]
   except IndexError as e:
      print(result['X_test'])
      raise e
   
   # retrieve test features from the original dataframe
   result["test_df"] = result["df"].loc[dates]
   
   # remove duplicated dates in the testing dataframe
   result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
   
   # remove dates from the training/testing sets & convert to float32
   result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
   result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)

   return result


def partial_load_data(ticker, n_steps=50, n_forward_steps=1, scale=True, shuffle=True, lookup_step=1, split_by_date=True,
              test_size=0.2, feature_columns=['close', 'volume', 'open', 'high', 'low']):
   
   # see if ticker is already a loaded stock from yahoo finance
   if isinstance(ticker, str):
      # load it from yahoo_fin library
      df = load_dataset(ticker)

   elif isinstance(ticker, pd.DataFrame):
      # already loaded, use it directly
      df = ticker

   else:
      raise TypeError(
          "ticker can be either a str or a `pd.DataFrame` instances")

   if len(df) < 500:
      return None

   feature_columns = list(feature_columns)
   # target_columns = target_column[:] if isinstance(target_column, list) else [target_column]

   # this will contain all the elements we want to return from this function
   proto_result = {}
   # we will also return the original dataframe itself
   proto_result['df'] = df
   
   # make sure that the passed feature_columns exist in the dataframe
   for col in feature_columns:
      assert col in df.columns, f"'{col}' does not exist in the dataframe."

   # add date as a column
   if "date" not in df.columns:
      df["date"] = df.index
   
   types = df.dtypes.to_dict()
   if scale:
      column_scaler = {}
      # scale the data (prices) from 0 to 1
      # scale_columns = np.unique(feature_columns[:] + target_columns[:]).tolist()
      scale_columns = [k for k in types.keys() if types[k] == np.float64]
      
      for column in scale_columns:
         scaler = MinMaxScaler()
         df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
         column_scaler[column] = scaler

      # add the MinMaxScaler instances to the result returned
      proto_result["column_scaler"] = column_scaler

   R = proto_result
   def load_tail(target_column):
      result = R.copy()
      # add the target column (label) by shifting by `lookup_step`
      df = R['df'].copy()
      # print(df)
      df[f'future_{target_column}'] = df[target_column].shift(-lookup_step)
      df = df.dropna()
      df['date'] = df.index
      # last `lookup_step` columns contains NaN in future column
      # get them before droping NaNs
      last_sequence = np.array(df[feature_columns].tail(lookup_step))

      sequence_data = []
      sequences = deque(maxlen=n_steps)

      # future_columns = ['future_{}'.format(c) for c in target_columns]
      input_shape, output_shape = None, None
      # future = None

      et = zip(
         (df[feature_columns + ["date"]].values),
         df[f'future_{target_column}'].values
      )

      for entry, target in et:
         sequences.append(entry)
         if len(sequences) == n_steps:
            sequence_data.append([
               np.array(sequences),
               target
            ])

      # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
      # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
      # this last_sequence will be used to predict future stock prices that are not available in the dataset
      last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
      last_sequence = np.array(last_sequence).astype(np.float32)
      result['last_sequence'] = last_sequence

      # construct the X's and y's
      X, y = [], []
      for seq, target in sequence_data:
         X.append(seq)
         y.append(target)
      # convert to numpy arrays
      X,y = np.array(X), np.array(y)

      if split_by_date:
         # split the dataset into training & testing sets by date (not randomly splitting)
         train_samples = int((1 - test_size) * len(X))
         result['split_index'] = train_samples
         result["X_train"] = X[:train_samples]
         result["y_train"] = y[:train_samples]
         result["X_test"] = X[train_samples:]
         result["y_test"] = y[train_samples:]
         if shuffle:
            # shuffle the datasets for training (if shuffle parameter is set)
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
      else:
         # split the dataset randomly
         result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(
            X, y, test_size=test_size, shuffle=shuffle)
      
      # get the list of test set dates
      try:
         dates = result["X_test"][:, -1, -1]
      except IndexError as e:
         print(result['X_test'])
         raise e

      # retrieve test features from the original dataframe
      result["test_df"] = result["df"].loc[dates]

      # remove duplicated dates in the testing dataframe
      result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(
         keep='first')]

      # remove dates from the training/testing sets & convert to float32
      result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
      result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)

      return result

   return load_tail

@persistent_memoize
def build_full_dataset(n_steps=50, n_forward_steps=1, scale=True, shuffle=True, lookup_step=1, split_by_date=True,
                       test_size=0.2, feature_columns=['close', 'volume', 'open', 'high', 'low'], target_column='close', nb_samples=80000):
   
   #TODO shit in a hotdog bun
   kwargs = dict(
       n_steps=n_steps,
       n_forward_steps=n_forward_steps,
       scale=scale,
       shuffle=shuffle,
       lookup_step=lookup_step,
       split_by_date=split_by_date,
       test_size=test_size,
       feature_columns=feature_columns,
       target_column=target_column
   )
   
   symbols = all_available_symbols()
   random.shuffle(symbols)
   
   loadresults = []
   nsamples = 0
   for sym in symbols:
      
      data = load_data(sym, **kwargs)
      if data is None:
         continue
      data = Struct(**data)
      print(data.y_train.shape)
      loadresults.append((sym, data))
      nsamples += len(data.X_train)
      print('nbsamples: ', nsamples)
      if nsamples >= nb_samples:
         break

   data = dict(X_train=[], y_train=[], sample_trace=[])

   test = loadresults.pop()

   for ticker, res in loadresults:
      start = len(data['X_train'])-1
      data['X_train'].extend(res.X_train.tolist())
      data['y_train'].extend(res.y_train.tolist())
      end = len(data['X_train'])-1
      data['sample_trace'].append(dict(ticker=ticker,r=res,pos=(start,end)))
   
   #TODO provide option to specify the test symbol(s) explicitly
   data['X_test'] = test[1].X_test
   data['y_test'] = test[1].y_test
   data['column_scaler'] = test[1].column_scaler

   data['X_train'] = array(data['X_train'])
   data['y_train'] = array(data['y_train'])

   return data

def dmodf(d:dict, *mods):
   r = d.copy()
   for f in mods:
      f(r)
   return r

def dapply(a:dict, b:dict):
   a = a.copy()
   a.update(b)
   return a

def kws(d:dict, *ext:dict):
   r = []
   kv_mode = False
   if isinstance(ext[0], tuple):
      r = {}
      kv_mode = True
   
   if not kv_mode:
      for n in ext:
         r.append(dapply(d, n))
   else:
      for name,m in ext:
         r[name] = dapply(d, m)
   return r

def build_multivariate_dataset(hook, **kw):
   init = kw.copy()
   
   init.pop('n_steps')
   init.pop('target_columns')
   
   configurations = hook(**init)
   
   if type(configurations) is list:
      results = [None for x in range(len(configurations))]
      for i, kwds in enumerate(configurations):
         results[i] = build_full_dataset(**kwds)
      return results
   
   elif type(configurations) is dict:
      results = {}
      for name, kwds in configurations.items():
         r = build_full_dataset(**kwds)
         results[name] = r
      return results
   raise ValueError('mistakes done been made, nigga')


PLD2_length_threshold = 1000
PLD2_null_limit = 0.85
_Scaler = QuantileTransformer #MinMaxScaler

#TODO convert to a class structure of some sort, this is getting unmanageable
def partial_load_data2(ticker, n_steps=50, n_forward_steps=1, scale=True, shuffle=True, lookup_step=1, split_by_date=True,
              test_size=0.2, feature_columns=['close', 'volume', 'open', 'high', 'low'], strict=True):
   # this will contain all the elements we want to return from this function
   proto_result = {}
   
   # see if ticker is already a loaded stock from yahoo finance
   if isinstance(ticker, str):
      # load it from yahoo_fin library
      df = load_dataset(ticker)
      proto_result['ticker'] = ticker

   elif isinstance(ticker, pd.DataFrame):
      # already loaded, use it directly
      df = ticker

   else:
      raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")

   if len(df) < PLD2_length_threshold:
      return None

   feature_columns = list(feature_columns)
   # target_columns = target_column[:] if isinstance(target_column, list) else [target_column]
   if strict:
      for col in feature_columns:
         #? reject if more than 15% of any one column is filled with null values
         pct_null = (df[col].isnull().sum()/len(df))
         if pct_null > PLD2_null_limit:
            return None
   
   # we will also return the original dataframe itself
   proto_result['df'] = df
   # df = df.fillna(method='ffill')
   
   # make sure that the passed feature_columns exist in the dataframe
   for col in feature_columns:
      assert col in df.columns, f"'{col}' does not exist in the dataframe."

   # add date as a column
   if "date" not in df.columns:
      df["date"] = df.index
   df = df.reset_index()
   sl = len(df)
   df.dropna(inplace=True)
   
   @once
   def _scaling(df:pd.DataFrame, result):
      types = df.dtypes.to_dict()
      if scale:
         column_scaler = {}
         # scale the data (prices) from 0 to 1
         # scale_columns = np.unique(feature_columns[:] + target_columns[:]).tolist()
         scale_columns = [k for k in types.keys() if types[k] == np.float64]
         
         for column in scale_columns:
            scaler = _Scaler()
            # df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))[:, 0]
            column_scaler[column] = scaler

         # add the MinMaxScaler instances to the result returned
         result["column_scaler"] = column_scaler

   R = proto_result
   passthru = lambda *n: tuple(n)
   
   # @ojit
   def tail(target_columns, hook=passthru, pre=None, post=None):
      # target_column = target_columns[0]
      result = R.copy()
      
      # add the target column (label) by shifting by `lookup_step`
      result['df'] = df = R['df'].copy()
      
      df = df.reset_index(drop=True)
      df = df.fillna(method='ffill').fillna(method='bfill')
      df = df.dropna()
      df.index = df.date
      
      for c in target_columns:
         column = df[c].values
         #TODO write custom delta function which works around the division-by-zero problem
         column_delta = divdelta(column)
         df[c] = column_delta
      
      # drop the first row of the DataFrame 
      df = df.iloc[1:].copy()
      
      # apply scaling to the DataFrame
      _scaling(df, result)

      if 'date' not in df.columns:
         df['date'] = df.index
      for c in target_columns:
         df[f'future_{c}'] = df[c].shift(-lookup_step)
      # df = df.iloc[2:]
      
      # last `lookup_step` columns contains NaN in future column
      # get them before droping NaNs
      last_sequence = np.array(df[feature_columns].tail(lookup_step))

      sequence_data = []
      sequences = deque(maxlen=n_steps)

      future_columns = ['future_{}'.format(c) for c in target_columns]
      input_shape, output_shape = None, None
      # future = None
      
      if pre is not None:
         df = pre(df)

      future = None
      trim_steps = 0
      if n_forward_steps > 1 or len(target_columns) > 1:
         future = []
         
         for i in range(len(df)):
            fut = df[future_columns].iloc[i:i+n_forward_steps].values
            
            if output_shape is None:
               output_shape = fut.shape

            if len(fut) < n_forward_steps:
               trim_steps = (len(df) - i)
               break

            future.append(fut)

      def left(a): 
         return (a[:-trim_steps] if trim_steps > 0 else a)

      et = zip(
         left(df[feature_columns + ["date"]].values),
         # future if future is not None else df[future_columns].values
         df[future_columns].values
      )

      for entry, target in et:
         sequences.append(entry)
         if len(sequences) == n_steps:
            sequence_data.append([
               np.array(sequences),
               target
            ])

      # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
      # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
      # this last_sequence will be used to predict future stock prices that are not available in the dataset
      last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
      last_sequence = np.array(last_sequence).astype(np.float32)
      result['last_sequence'] = last_sequence

      # construct the X's and y's
      X, y = [], []
      for seq, target in sequence_data:
         seq, target = hook(seq, target)
         if len(seq[seq == np.nan]) > 0 or len(target[target == np.nan]) > 0:
            print('ur still stupid')
            continue
         X.append(seq)
         y.append(target)
      # convert to numpy arrays
      X,y = np.array(X), np.array(y)

      if split_by_date:
         # split the dataset into training & testing sets by date (not randomly splitting)
         train_samples = int((1 - test_size) * len(X))
         result['split_index'] = train_samples
         result["X_train"] = X[:train_samples]
         result["y_train"] = y[:train_samples]
         result["X_test"] = X[train_samples:]
         result["y_test"] = y[train_samples:]
         if shuffle:
            # shuffle the datasets for training (if shuffle parameter is set)
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
      else:
         # split the dataset randomly
         result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(
            X, y, test_size=test_size, shuffle=shuffle)
      
      # get the list of test set dates
      try:
         dates = result["X_test"][:, -1, -1]
      except IndexError as e:
         print(result['X_test'])
         raise e

      # retrieve test features from the original dataframe
      result["test_df"] = result["df"].loc[dates]

      # remove duplicated dates in the testing dataframe
      result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]

      # remove dates from the training/testing sets & convert to float32
      preshape = result['X_train'].shape
      result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
      result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)
      
      assert result['X_train'].shape[-1] == len(feature_columns)
      
      if post is not None:
         result = post(result)
      result['ticker'] = ticker
      return result

   return tail


@persistent_memoize
# @ojit
def multivariate_dataset(**kw):
   targets = kw.pop('target_columns')
   
   symbols = all_available_symbols(kind=Stock)
   random.shuffle(symbols)
   
   kwp = lambda k,dv=None: kw.pop(k) if k in kw.keys() else dv
   pre, post, hook = kwp('pre'), kwp('post'), kwp('hook')
   nloaders = kwp('nloaders') or 100
   dnsamples = kwp('nsamples')

   nsamples = 0
   loaders = []
   for sym in symbols[0:nloaders]:
      f = partial_load_data2(sym, **kw)
      if f is None:
         vprint(f'{sym.upper()} rejected from dataset')
         continue
      loaders.append((sym, f))
      print(len(loaders))
   
   # pre, post = kw.get('pre',None),kw.get('post',None)
   
   #TODO flesh out with more details
   X_train, y_train = [], []
   samples = []
   len_total = 0
   blacklist = []
   for sym, f in loaders:
      r = f(target_columns=targets, pre=pre, post=post)
      # X_train.extend(r['X_train'])
      # y_train.extend(r['y_train'])
      if not 'ticker' in r.keys():
         r['ticker'] = sym
      samples.append(r)
      len_total += len(r['X_train'])
      if dnsamples is not None and len_total >= dnsamples:
         break
   
   print(repr(blacklist))
   print(f'nb samples: ~{len_total}')
   # sort the list of results by the number of samples each contains   
   samples = [(len(r['X_train']), r) for r in samples]
   samples = sorted(samples, key=lambda x: x[0])
   print([f'({n}, ...)' for (n, data) in samples])
   
   #TODO...

   return [r for (n,r) in samples]

def default_cfg(m=None):
   defaults = dict(
      scale=True, 
      shuffle=True, 
      lookup_step=1,
      split_by_date=True, 
      test_size=0.2
   )
   if m is not None:
      defaults.update(m)
   return defaults

         
def _test():
   defaults = dict(n_steps=50, n_forward_steps=1, scale=True, shuffle=True, lookup_step=1, split_by_date=True, test_size=0.2, feature_columns=['close', 'volume', 'open', 'high', 'low'])
   r = multivariate_dataset(target_columns=['open', 'high', 'low', 'close'], **defaults)
   # for column, (X,y) in r.items():
   #    print(f'({column}) -> {len(X)} samples')
   X,Y = [],[]
   for (x,y) in map(lambda a: (a['X_train'],a['y_train']), r):
      X.extend(x)
      Y.extend(y)
   X,Y = array(X),array(Y)
   print(X)
   print(Y)


if __name__ == '__main__': _test()