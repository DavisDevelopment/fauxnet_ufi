import random
import numpy as np
import os, math
import builtins as g
from numpy import *
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

from ds.common import *
from ds.gtools import ojit, Struct, mpmap, njit, jit, vprint, TupleOfLists
from ds.datatools import *
import ds.data as dsd
from ds.maths import *

import warnings, shelve, random
import collections
import matplotlib.pyplot as plt
from cytoolz import *
#import tensorflow as tf
##import tensorflow.keras as keras
##from tensorflow.keras import *
##from tensorflow.keras import Sequential, Model
##from tensorflow.keras.layers import *
##from tensorflow.keras.models import clone_model, model_from_json
from ..model_classes import *
from ..model_train import *
from ds.model.ctrl import MdlCtrl, MultiCtrl
from ds.forecaster import *

from numba import stencil, njit
from tools import dotset, dotget

tf.config.optimizer.set_jit(True)

SEED = 123456789
IS_TUNING = False

def tuningmode(flag:bool):
    g=globals()
    g['IS_TUNING'] = flag

#Function to initialize seeds for all libraries which might have stochastic behavior
def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
#   tf.random.set_seed(seed)
    np.random.seed(seed)

## Activate Tensorflow deterministic behavior
def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

#   tf.config.threading.set_inter_op_parallelism_threads(1)
#   tf.config.threading.set_intra_op_parallelism_threads(1)

def passthru(x: ndarray):
    return x

def nd_rows(x: ndarray):
    dims = x.shape[:-1]
    ranges = [range(n) for n in dims]
    return product(*ranges)

def nancount(a):
    return np.count_nonzero(np.isnan(a))

def nonan(**al):
    for name, a in al.items():
        nn = nancount(a)
        if nn != 0:
            raise AssertionError(f'{nn} nans in {name}')
        

@njit
def diffthatgayshit(x):
    for i in range(0, len(x)):
        x[i] = x[i] - x[i - 1]
    return x

def diffdf(df: pd.DataFrame):
    for c in df.columns:
        delta = diffthatgayshit(df[c].to_numpy())
        df[c] = delta

from ds.model.spec import Hyperparameters, defspec

class ProtoFauxNet:
    data:dsd.DataLoaderResult
    spec:Hyperparameters
    
    def __init__(self, spec=None, name=None):
        self.name = name
        self.spec = spec
        self.cache_name = 'fnetcache'
        if name is not None:
            self.cache_name = f'{self.cache_name}_{name}'
        self.cache_path = P.join(P.dirname(__file__), self.cache_name)
        
        self.cache = {}#shelve.open(self.cache_path)
        self.data = None
        self.nn = None
        self.cells = None
        self.conv_thresh = 0.008
        
        self.model_path = 'saved_model'
        if name is not None:
            self.model_path = f'{name}_{self.model_path}'
        
        self.rebuild_data = False
        self.rebuild_model = False
        
        # self.cache['hyperparams'] =         
        global PFN
        PFN = self
    
    def _restore(self):
        kl = list(self.cache.keys())
        
        try:
            self.cache['testing...'] = (1, 2, 3)
        except KeyError as err:
            if 'cannot add item to database' in str(err):
                self.cache.close()
                os.remove(f'{self.cache_name}.db')
                self.cache = shelve.open(self.cache_path)
                raise Exception('what the fuck')
                kl = []
        
        if 'hyperparams' not in kl:
            self.cache['hyperparams'] = self.spec
            self.rebuild_model = True
            self.rebuild_data = True
            
        elif self.spec is None:
            self.spec = self.cache['hyperparams']
        
        if self.spec != self.cache['hyperparams']:
            #? rebuild dataset and Models if the hyperparameters have been changed
            self.rebuild_model = True
            self.rebuild_data = True
            self.cache['hyperparams'] = self.spec

        # elif self.spec == self.cache['hyperparams']:
        #     print(self.spec.to_kwds(), self.cache['hyperparams'].to_kwds())
        
        if 'data' in kl and not self.rebuild_data:
            #TODO load our data from the cache
            self.data = self.cache['data']
            self.X = conv2d_compat(self.data.X_train)
        else:
            self.procure_data()
            self.X = conv2d_compat(self.data.X_train)
            
    def load_saved_nns(self):
        nns = {}
        if not self.rebuild_model:
            try:
                print('Loading model from {}'.format(self.model_path))
#                model = keras.models.load_model(self.model_path, custom_objects=custobjs)
                print(model.summary())
                nns['main'] = model
            
            except Exception as error:
                # raise Exception('Failed to restore saved model with error: %s' % error)
                print(error)
                
#                #? try loading only the model's tensorflow graph
                #? this will mean that the model is callable, but lacks its custom objects
#               model = tf.saved_model.load(self.model_path)
                nns['main'] = model
        
        return nns
    
    # @cachedmethod
    def build_testing_set(self, seed=None):
        spec = self.spec
        #? kwds = dsd.default_cfg(dict(n_steps=spec.n_steps, feature_columns=spec.feature_columns, target_columns=spec.target_columns))
        loader = dsd.DataLoader(hparams=spec)
        menu = dsd.all_available_symbols(kind=dsd.Crypto)
        menu = list(filter(lambda s: s.endswith('usd'), menu))
        sub = random.sample(menu, math.floor(len(menu) * 0.05))
        
        testsets = []
        for sym in sub:
            r = loader.load(sym)
            if r is None:
                continue
            testsets.append(r)
        
        self.cache['robust_testsets:raw'] = testsets
        
        return testsets
            
    def procure_data(self, **params):
        if self.data is not None:
            return self.data
        else:
            data = self.data = mkdata(self, **params)
            _X, _y = data.X_train, data.y_train
            channelsets = named_split_y_channels(_X, _y, self.spec.target_columns)
            data.__dict__.update(channelsets)
            
            self.cache['data'] = data
            
            return data
    
    def build_final_network(self):
        nns = self.load_saved_nns()
        nn = nns.get('main', None)
        
        if self.rebuild_model or nn is None:
            # nn = TotemicOHLC(hyperparams=self.spec)
            nn = OHLC(name='tits', celltype=1, hyperparams=self.spec)
            
#            o = keras.optimizers.RMSprop(learning_rate=0.0022)
            nn.compile(optimizer=o, loss='mse')
            
            nn(self.data.X_train)
            print(nn.summary(expand_nested=True))
            
            train_model(nn, self.data, conv_thresh=0.008, spec=self.spec)
            #TODO:
            #! now, split the column-forecasting layers into individual Models of their own and train each of them
            #! to a convergence point of at most 0.05%

            # colmodels = []
            # for i, cell in enumerate(nn.cells):
            #     colmodl = Sequential([cell])
#            #     o = keras.optimizers.Adam(learning_rate=0.022)
            #     colmodl.compile(optimizer=o, loss='mse')
            #     colmodels.append(colmodl)
            
            # train_column_model(colmodels, self.data, conv_thresh=0.005, spec=self.spec)
        # evaluate_model(nn, self.spec, 'eth', self.data)
        # evaluate_model(nn, self.spec, 'aapl', self.data)
        
        self.nn = nn
        
        return nn
    
    def test_forecaster(self, nnf, symbols):
        loader = dsd.DataLoader(hparams=self.spec)
        spec = self.spec
        datas = [loader.load(symbol) for symbol in symbols]
        for data in datas:
            x, y = data.eget('X_train', 'y_train')
            ypred = nnf(x)
            scalers = data.column_scaler
            sm = {i:scalers[v] for i, v in enumerate(spec.target_columns)}
            y = descale(sm, y)
            ypred = descale(sm, ypred)
            
            _score = np.zeros((len(spec.target_columns), 3))
            for i in range(len(spec.target_columns)):
                _score[i, :] = score(y[i, :], ypred[i, :])
            
            for i, name in enumerate(spec.target_columns):
                print(name, _score[i, :].tolist())

    def restore_ensemble(self, *specvars):
        if not 'ensemble' in self.cache:
            return None

        saved = self.cache['ensemble']
        loaded = []
        for (path, params, data) in saved:
            c = MdlCtrl(params=params, data=data, model=path)
            loaded.append(c)
        return loaded

    def build_ensemble(self, *specvars):
        rest_e = self.restore_ensemble(*specvars)
        if rest_e is not None:
            units = rest_e
        else:
            variants = specvars[:]
            mkunit = ensembleUnitCtor(self.spec)
            unit_args = [mkunit(variant) for variant in variants]
            
            units = [unit for (params, data, path, unit) in unit_args]
                
            self.cache['ensemble'] = tuple([(path, params, data) for (params, data, path, unit) in unit_args])
        
        ensemble = MultiCtrl(units)
        
        return ensemble
        
    def build(self, data=None, rebuild=False, **kwargs):
        self._restore()
        
        if rebuild:
            self.data = None
        
        if data is not None:
            self.data = data
        
        elif data is None and self.data is None:
            dparams = kwargs.pop('dataloader', None)
            print(dparams)
            
            if dparams is not None:
                self.procure_data(**dparams)
            else:
                self.procure_data()
            
        data = self.data
        self.cache['data'] = data
        
        from ds.model.kfuncs import moe
            
        params = self.spec
        
        controller = MdlCtrl(
            params=self.spec, 
            data=self.data, 
            model=f'{self.name}_saved_model', 
            debug=False, 
            autoload=(not rebuild)
        )
        
        if rebuild or controller.model is None:
            nnx = kwargs.pop('nn_ctor')
            if nnx is not None:
                assert 'new' in nnx
                nn_ctor = nnx.pop('new')
                nn_params = nnx
                nn = nn_ctor(**nn_params)
            else:
                nn = cnnpred_2d2(seq_len=params.n_steps, n_features=len(
                    params.feature_columns), n_features_out=len(params.target_columns))
                
#            dnnc = dict(optimizer=keras.optimizers.RMSprop(learning_rate=0.001), loss='mae')
            nnc = kwargs.pop('nn_compile', None)
            if nnc is None:
                nnc = dnnc
            else:
                lr = dnnc['optimizer'].learning_rate
                opt = type(dnnc['optimizer'])
                
                if 'optimizer' in nnc.keys():
                    opt = nnc.pop('optimizer')
#                    opt = type(keras.optimizers.get(opt))
                    
                if 'learning_rate' in nnc.keys():
                    # opt = dnnc['optimizer'] = type(dnnc['optimizer'])(learning_rate=nnc.pop('learning_rate'))
                    lr = nnc.pop('learning_rate')

                dnnc['optimizer'] = opt(learning_rate=lr)
                nnc = merge(dnnc, nnc)
            
            nn.compile(**nnc)
            controller.model = nn
        
        fitp = kwargs.pop('fit', None)
        fit_parameters = dict(epochs=7, verbose=1)
        if fitp is not None:
            fit_parameters.update(fitp)
        
        summary = controller.summary()
        print('printing summary')
        pprint(summary)
        
        controller.train(fit_parameters=fit_parameters, keep_best=(not IS_TUNING))
        ctrl_a = controller
        return ctrl_a
        
def dwith(data, X, y):
    c = data.copy()
    c.X_train = X
    c.y_train = y
    return c
     
def named_split_y_channels(x, y, channel_names):
    ch = split_y_channels(x, y)
    assert len(ch) == len(channel_names)
    d = {}
    for i in range(len(channel_names)):
        d[channel_names[i]] = ch[i]
    return d
      
@jit
def split_y_channels(x, y:ndarray):
    y = y.T
    nch = y.shape[0]
    return [(x, y[i, :]) for i in range(nch)]

def mkdata(n, **params):
    from engine.data import dataloader as dl
    from engine.data.prep import compileFromHyperparams as extractor
    from ds.datasets import all_available_symbols
    
    spec = n.spec if not isinstance(n, Hyperparameters) else n
    n_steps = spec.n_steps
    n_channels = spec.nchannels
    columns = spec.target_columns
    
    defaults = dsd.default_cfg(dict(
        parallel=False,
        nsamples=150000,
        prioritize=all_available_symbols(kind='kraken/')
    ))
    defaults.update(params)
    
    if isinstance(n, ProtoFauxNet):
        setattr(n, 'dataloader_params', defaults)
    
    st = time.time()
    fex = extractor(spec)
    
    # @fex.add
    # def difference_df(df: pd.DataFrame):
    #     ndoc = df[columns].copy(deep=True)
    #     for c in columns:
    #         ndoc[c] = ndoc[c].diff()
    #     print(ndoc)
    #     return ndoc
    
    builder = dsd.DatasetBuilder(spec, fex=fex)
    
    dset = builder.load(**defaults)
    et = time.time()
    #TODO add the kraken_cache data onto this dataset, to train the networks on the data they'll be applied to
    print(f'took {et-st}secs to load dataset of {dset.nsamples} samples')
    
    return dset


def add_astrological_columns(df=None):
    from ds.astrol import moon
    phases = moon.phase.afast(df.index.to_julian_date().to_numpy())
    df['moon_illum'] = phases[:, 1]
    
    print(df)
    
class Phoenix:
    cells: List[Model] = None

    def __init__(self, cell_columns=['open', 'high', 'low', 'close']):
        self.cell_columns = cell_columns[:]
        self.dataset_config = dsd.default_cfg()
        self.dataset = None
        self.cells = [None for n in range(len(cell_columns))]
        # TODO ...

    def build(self):
        self.dataset = dsd.multivariate_dataset(
            target_columns=self.cell_columns, **self.dataset_config)
        print(self.dataset.keys())
        ds = self.dataset[self.cell_columns[0]]
        X, y = ds['X_train'], ds['y_train']
        shapes = [n.shape[1:] for n in [X, y]]
        ishape, oshape = shapes
        for i, c in enumerate(self.cell_columns):
            cell = mh_model_alt(ishape, oshape, name=c)
            self.cells[i] = cell

    def fit(self):  # , X, y):
        for i, cell in enumerate(self.cells):
            ds = self.dataset[self.cell_columns[i]]
            train(cell, ds, epochs=50, conv_thresh=0.012)

    def predict_single(self, X):
        # TODO
        pass

    def predict(self, X):
        pass

    def forecast(self, X, n_steps: int = 1):
        # ? fancy clever schmootsy-poo
        pass
