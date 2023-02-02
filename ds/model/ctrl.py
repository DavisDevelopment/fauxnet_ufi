from ds.gtools import *

##from tensorflow.keras import *
##from tensorflow.keras import Model
##from tensorflow.keras.layers import *

from ds.model.spec import Hyperparameters
# from ds
# from ds.data
from ds.model_classes import *
from ds.model_train import *
from copy import deepcopy
from functools import *
from fn import F, _
from operator import methodcaller
from builtins import round
from tools import profile

DataFrame = pd.DataFrame

def meanwooutliers(a: pd.Series, zthresh=3):
   return a.where(zscore(a.to_numpy()) < zthresh).mean()

@jit(cache=True)
def meanstar(a: np.ndarray):
   assert a.ndim == 2
   acc = np.zeros(a.shape[0])
   for i in range(a.shape[0]):
      acc[i] = a[i, :].mean()
   return acc.mean()

def getcols(pattern:str, df:pd.DataFrame, tonp=True):
   import fnmatch, re
   names = df.columns.tolist() if hasattr(df, 'columns') else df.index
   mcols = fnmatch.filter(names, pattern)
   res = [df[c] for c in mcols]
   if tonp:
      res = np.asanyarray([s.values for s in res])
   return res

mean = F(meanstar) << (lambda df, suffix: getcols(f'*.{suffix}', df))

def drename(d, name:str, newname:str):
   d[newname] = d.get(name, None)
   del d[name]
   return d

def merge_dfs_from_list(df_list):
   return pd.concat(df_list, ignore_index=True)

@njit
def delta2d(arr: ndarray):
   for i in range(arr.shape[0]):
      row = arr[i, :]
      arr[i, :] = delta(row)
   return arr

class MdlCtrl:
   def __init__(self, params:Hyperparameters=None, data=None, model=None, debug=False, autoload=True):
      self.nn = None
      self.nn_path = None
      self.params = params
      self.data = data
      self.debug = debug

      self.report = None      
      self._is_trained = False
      self._is_evaluated = False
      self._convergence_threshold = 0.01
      self._convergence_reached = False
      self._novel_data = None
      self._summary = None
      
      if model is not None:
         if isinstance(model, str):
            try:
               self.nn_path = model
               if autoload:
#                  model = keras.models.load_model(model, custom_objects=custobjs)
                  self._is_trained = True
#                  assert isinstance(model, keras.models.Model)
                  self.set_model(model)
            except Exception as error:
               print(error)
      
   @property
   def model(self):
      return self.nn
   
   @model.setter
   def model(self, nn):
      self.nn = nn
      self.nn.trainable = True
      setattr(nn, 'controller', self)

      if self.debug:
         # perform an initial "bootup" call of the model
         warmup = self.params.prep_data(self.data).X_train
         warmup,_ = self.preprocess(warmup, None)
         
         try:
            r = self.nn(warmup)
         except Exception as error:
            raise error
         self._summary = self.nn.summary()
         print(self._summary)
         
   def summary(self):
      nn:Model = self.nn
      if self._summary is None:
         try:
            self._summary = nn.summary(expand_nested=True)
         
         except Exception as e:
            # print(e)
            pass
      
      if self._summary is None:
         # perform an initial "bootup" call of the model
         warmup = self.params.prep_data(self.data).X_train
         warmup, _ = self.preprocess(warmup, None)

         try:
            r = self.nn(warmup)
         except Exception as error:
            raise error
         
         self._summary = self.nn.summary(expand_nested=True)
      
      
      return self._summary
      
   def set_model(self, nn):
      MdlCtrl.model.__set__(self, nn)      
      
   def _deltafy(self, data):
      data = data.copy()
      scaler = data.column_scaler[self.params.target_column]
      
      for c in ['X_train', 'X_test', 'y_train', 'y_test']:
         arr = getattr(data, c)
         arr = scaler.inverse_transform(arr.reshape(-1, 1))[:, 0]
         arr = (delta if arr.ndim == 1 else delta2d)(arr)
         if arr.ndim == 1:
            arr = arr[1:]
         else:
            arr = arr[:, 1:]
         
         setattr(data, c, arr)
      
   def preprocess(self, x, y):
      """
      stub
      """
      if self.params.feature_univariate and x is not None:
         x = x.reshape(x.shape[0], x.shape[1], 1)
         pass
         
      return x, y
   
   def _novel_dataset(self, nbnovel=8):
      if self._novel_data is not None:
         return self._novel_data
      
      from ds.data import all_available_symbols as allsyms
      from engine.data.prep import compileFromHyperparams as extractor
      
      allsyms = allsyms()
      loadedsyms = list(self.data.modules.keys())
      notloaded = list(set(allsyms) - set(loadedsyms))
      
      sel = random.sample(notloaded, nbnovel)
      sel.append('btc')
      loader = dsd.DatasetBuilder(self.params, fex=extractor(self.params))
      data = loader.load(symbols=sel, parallel=False)
      self._novel_data = data
      return data
   
   def _thorough_evaluate_inner(self, data, mode='test', sub=None):
      from tqdm import tqdm
      
      evalentries = []
      modules = list(data.modules.items())
      if sub is not None and isinstance(sub, int) and sub < len(modules):
         modules = random.sample(modules, sub)
      
      entries = tqdm(modules)
      for name, data in entries:
         entries.set_description(name)
         y, ypred = applymodel(self, data, mode=mode, scale=True)
         print(f'y(pred)=', ypred[-1])
         print(f'y=', y[-1])
         evres = dflatten(polyscore(y, ypred, columns=self.params.target_columns), join=lambda x,y: f'{x}.{y}')
         evres['name'] = name
         # evres['scaler'] = data.column_scaler
         evres['y'] = y
         evres['ypred'] = ypred
         evalentries.append(evres)
      
      report = pd.DataFrame.from_records(evalentries)
      report.set_index('name', inplace=True, drop=False)
      report['mode'] = mode
      newcolorder = ['name', 'mode']#, 'mean_error', 'directional_accuracy', 'mean_offset', 'correlation', 'mode']
      for c in self.params.target_columns:
         newcolorder.extend([f'{c}.{n}' for n in ['mean_error', 'directional_accuracy', 'mean_offset', 'correlation']])
      newcolorder += list(set(report.columns) - set(newcolorder))
      report = report[newcolorder]
      
      summary = (
         mean(report, 'mean_error'),
         mean(report, 'correlation'),
         mean(report, 'directional_accuracy')
      )
      
      return report, summary
   
   def _thorough_evaluate(self):
      evaluate = lambda a, **kw: self._thorough_evaluate_inner(a, **kw)[0]
      
      # train = evaluate(self.data, mode='train')
      test = evaluate(self.data, mode='test', sub=20)
      
      for x in (test,):
         x['novel'] = False
      
      novel_data = self._novel_dataset()
      novel_train = evaluate(novel_data, mode='train')
      # novel_test = evaluate(novel_data, mode='test')
      for x in (novel_train,):
         x['novel'] = True
      
      dfs = [test, novel_train]
      keys = ['name', 'novel', 'mode', ]
      
      report = merge_dfs_from_list(dfs)
      report.set_index('name', inplace=True, drop=False)
      
      report['mean_error'] = None
      report['directional_accuracy'] = None
      report['correlation'] = None
      
      rowattr = F(lambda k, row: getcols(f'*.{k}', row, tonp=False)) >> np.asarray >> methodcaller('mean')
      def raggr(row, idx, name):
         report.loc[idx, name] = rowattr(name, row)
      
      for idx, row in report.iterrows():
         raggr(row, idx, 'mean_error')
         raggr(row, idx, 'directional_accuracy')
         raggr(row, idx, 'correlation')

      trunc_columns = ['name', 'novel', 'mode', 'mean_error', 'directional_accuracy', 'correlation']
      trunc_columns += list(set(report.columns) - set(trunc_columns))

      report = report[trunc_columns]
      
      return report
   
   def _gen_report(self):
      report = self._thorough_evaluate()
      pctcols = ['mean_error', 'directional_accuracy', 'correlation']
      report = report.dropna().copy(deep=True)
      for c in pctcols:
         s:pd.Series = report[c]
         report[c] = (s * 100)

      summarycols = pctcols[:]

      summary = [report[c] for c in summarycols]
      moe, dacc, corr = summary
      
      self._is_evaluated = True
      self.report = report
      return report
      
   @profile
   def stats(self, save_path=None, force=False):
      if not force and self.report is not None:
         report = self.report
      else:
         report = self._gen_report()
         # return report
      
      me:pd.Series = report.mean_error
      x = me.mean(), me.min(), me.max(), me.std()
      
      if save_path is not None:
         if not save_path.endswith('.csv'):
            save_path = save_path + '.csv'
         report.to_csv(save_path)
      
      return report
   
   def _convergence_test(self, report:DataFrame):
      return True
   
   def train(self, force=False, **kwds):
      if force or not self._is_trained:
         if self.params.univariate:
            data = self.params.prep_data(self.data)
         else:
            data = self.data
         
         print(self.summary())
         epochs = 1
         
         if 'fit_parameters' in kwds and 'epochs' in kwds['fit_parameters']: 
            epochs = kwds['fit_parameters']['epochs']
         
         keepbest = kwds.pop('keep_best', True)   
         
         def _bound_score():
            #* this overrides the default scoring function defined in train_model, giving us a much better estimate of the real-world performance when we score our network
            rep = self.stats(force=True)
            moe, cor, dac = rep.mean_error.mean(), rep.correlation.mean(), rep.directional_accuracy.mean()
            return moe, cor, dac
         
         converged = False
         step = train_model(self, data, spec=self.params, incremental=True, **kwds)
         while not converged:
            step(epochs=epochs, score_fn=_bound_score)
            converged = self._convergence_test(self.stats())
         step(done=converged, keepbest=keepbest)
         
      if self.debug:
         if not self._is_evaluated or self.report is None:
            self._gen_report()
         print(self._summary)
      
   def __call__(self, inputs):
      return self.model(inputs)
      
   @cached_property
   def forecaster(self):
      from ds.forecaster import NNForecaster
      
      return NNForecaster(self, self.params)
   
   def fit(self, *args, **kwargs):
      return self.model.fit(*args, **kwargs)
   
   def predict(self, *args, **kwargs):
      return self.model.predict(*args, **kwargs)
   
   def save(self, *args, **kwargs):
      return self.model.save(*args, **kwargs)
   
   def load(self):
      if self.nn_path is not None:
#         nn = keras.models.load_model(self.nn_path, custom_objects=custobjs)
         self.model = nn
      else:
         raise ValueError('wyd baw')
   
class MultiCtrl:
   def __init__(self, units:List[MdlCtrl]):
      self.units = units
      
