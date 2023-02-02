# from numpy import *

# from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
# from scipy.stats import pearsonr, linregress, zscore

# from ds.common import *
# from ds.gtools import ojit, Struct, mpmap, njit, jit, vprint, TupleOfLists
# import ds.data as dsd
# from ds.maths import *
# from numba import objmode, types, typeof

# #import tensorflow as tf
# ##import tensorflow.keras as keras
# ### from tensorflow.keras import *
# ### from tensorflow.keras import Sequential, Model
# ### from tensorflow.keras.layers import *
# #from keras import *
# #from keras.layers import *
# #from keras.callbacks import *
# ##from tensorflow.keras.models import clone_model, model_from_json
# #from tensorflow.python.ops.numpy_ops import np_config
# # np_config.enable_numpy_behavior()

# import warnings, shelve, random, collections, threading, math
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from pprint import pprint
# from typing import *
# from numba.types import *
# from cachetools import cached, cachedmethod, LRUCache, TTLCache

# import toolz.functoolz as funcs
# import toolz.itertoolz as its
# from cytoolz.functoolz import *

# # from ds.model_classes import conv2d_compat

# Model = Any

# #!++++++++++++
# #!  TODO
# #!++++++++++++
# """
#  -
#   + refactor functions to reduce redundant computations:
#     + funcs which evaluate the accuracy of the Model should be passed the `ytrue` and `ypred` values, not the Model and the data with which to compute them over again
# """

# def conv2d_compat(x: ndarray):
#     N, H, W = x.shape
#     # print(W)
#     x2 = x.reshape(N, H, 1, W)
#     return x2

# Afl = float32[:]
# Adbl = float64[:]

# def descale(sm, x:ndarray, inverse=True):
#    """
#    reverses scaling transformations on the given ndarray 
#    """
#    apply = lambda sc, v: (sc.inverse_transform if inverse else sc.transform)(v)
   
#    if x.ndim == 1:
#       return apply(sm, x.reshape(-1, 1))[:, 0]
   
#    elif x.ndim == 2:
#       rows, cols = x.shape
#       x = x.copy()
#       # print(cols)
#       for j in range(cols):
#          # print(j)
#          v = apply(sm[j], x[:, j].reshape(-1, 1))
#          x[:, j] = v[:, 0]
#       return x
   
#    elif x.ndim == 3: # for multistep predictions
#       rows, steps, cols = x.shape
#       x = x.copy()
#       for col in range(cols):
#          for step in range(steps):
#             part = x[:, step, col]
#             part = apply(sm[col], part.reshape(-1, 1))[:, 0]
#             x[:, step, col] = part
#       return x
#    else:
#       raise ValueError('fuck dude')
   
# def transpose(*x): return tuple(a.T for a in x)

# def applymodel(model, data, mode='test', scale=True, column=None):
#    x, y = data.eget(f'X_{mode}', f'y_{mode}')
#    try:
#       x, y = model.preprocess(x, y)
#    except Exception as e:
#       pass
      
#    # ypred = model.predict(x, verbose=0, use_multiprocessing=True, workers=8)
#    ypred = model(x)
#    if not isinstance(ypred, np.ndarray):
#       ypred = ypred.numpy()
   
#    usy, usypred = y, ypred
   
#    if scale:
#       if not hasattr(data, 'column_scaler'):
#          print('TODO-WARNING: returning still-scaled data')
#          return (y, ypred)
      
#       if column is None:
#          channels = model.params.target_columns
         
#          sm = {i: data.column_scaler[v] for i, v in enumerate(channels)}
#          y = descale(sm, y)
#          ypred = descale(sm, ypred)
#       else:
#          scaler:MinMaxScaler = data.column_scaler[column]
         
#          y = scaler.inverse_transform(y.reshape(-1, 1))[:, 0]
#          ypred = scaler.inverse_transform(ypred.reshape(-1, 1))[:, 0]
         
#       assert not np.array_equal(usy, y)
#       assert not np.array_equal(usypred, ypred)
         
#    return (y, ypred)

# _imp_cache = {}

# def impdf(ticker, **params):
#    if isinstance(ticker, str) and ticker in _imp_cache.keys():
#       return _imp_cache[ticker]
   
#    print(params)
#    l = dsd.DataLoader(**params)
#    if isinstance(ticker, str):
#       df = dsd.load_dataset(ticker)
#    else:
#       df = ticker
   
#    state = l.import_df(df, wholestate=True)
#    res = l.samples_from(state)
#    # _imp_cache[ticker] = res
   
#    return res

# def ensure_has_module(data:dsd.Dataset, params, ticker:str):
#    if not ticker in data.modules.keys():
#       m = impdf(ticker, hparams=params)
#       data.modules[ticker] = m
#       data.results.append(m)
#    else:
#       m = data.modules[ticker]
      
#    return m

# def batches(*args, batch_size=250):
#    if len(args) == 1:
#       x = args[0]
#       xbatches = np.array_split(x, len(x) / batch_size)
#       return xbatches
#    else:
#       return tuple(batches(x, batch_size=batch_size) for x in args)

# # @ojit
# def catbatches(bl: List[ndarray], indices):
#    return np.vstack([bl[i] for i in indices])

# def domainaware_evaluate(ytrue:ndarray, ypred:ndarray):
#    cscores = []
#    ytrue = ytrue.T
#    ypred = ypred.T
   
#    for i in prange(0, ytrue.shape[0]):
#       c_true = ytrue[i, :] # ground-truth column values 
#       c_pred = ypred[i, :] # predicted column values
#       c_score = score(c_true, c_pred)
#       cscores.append(c_score)
      
#    #TODO combine cscores
#    return cscores

# def partbyz(a:ndarray, zthresh=3):
#    zscores = np.abs(zscore(a))
#    non_outliers = a[zscores < zthresh]
#    outliers = a[zscores >= zthresh]
#    return non_outliers, outliers

# def without_outliers(a:ndarray, zthresh=3):
#    return partbyz(a, zthresh=zthresh)[0]

# @jit
# def dediffcol(start: float = 1.0, c: np.ndarray = None):
#    return (start + c.cumsum())

# def score(ytrue:Afl, ypred:Afl, robust=False):
#    """
#    generates some performance scores based on ytrue and ypred

#    Parameters
#    ----------
#    ytrue : float32[:]
#    ypred : float32[:]

#    Returns
#    -------
#    margin_of_error, mean offset from ytrue, correlation coefficient between ytrue and ypred
#    """
   
#    #TODO make the list of metrics computed here configurable
#    assert ypred.ndim == ytrue.ndim, f'ytrue and ypred must be of the same shape, shapes given are {ytrue.shape}, {ypred.shape}'
#    ndim = ytrue.ndim
#    # ytrue = dediffcol(1.0, ytrue)
#    # ypred = dediffcol(1.0, ypred)
#    # for i, name in range(ytrue.shape[0]):
#    #    ytrue[i] = 
   
#    if ndim == 1:
#       check = lambda x: (np.isnan(x)|(~np.isfinite(x)))
#       # compute basic statistics
#       if True in check(ytrue) or True in check(ypred):
#          ytrue, ypred = npclean(ytrue, ypred)
#       offset:ndarray = abs(ypred - ytrue)
#       offset_mean = mean(offset)
#       # print('no. of outliers:', len(b))
#       offset[offset == 0] = 1e-8
#       ytrue[ytrue == 0] = ytrue.mean()
#       assertNoInvalid(offset, (0,))
#       assertNoInvalid(ytrue, (0,))
#       offset_pct = offset/ytrue
#       offset_pct_smoothed = without_outliers(offset_pct)
#       if len(offset_pct_smoothed) == 0:
#          offset_pct_smoothed = offset_pct
      
#       margin_of_error = mean(offset_pct_smoothed)

#       assertNoInvalid(ypred)
#       directional_accuracy = dacc(ytrue, ypred)
#       try:
#          corr, _ = pearsonr(ytrue, ypred)
#       except ValueError:
#          ytrue, ypred = npclean(ytrue, ypred)
#          corr = np.nan
      
#    # assertNoInvalid(margin_of_error)
#    # assertNoInvalid(corr)
#    # assertNoInvalid(directional_accuracy)
#    if robust:
#       return dict(
#          offset=offset,
#          mean_offset = offset_mean,
#          error=offset_pct,
#          mean_error=margin_of_error,
#          directional_accuracy=directional_accuracy,
#          correlation=corr
#       )
   
#    return margin_of_error, offset_mean, corr, directional_accuracy

# from cytoolz import *

# def npclean(a, b):
#    check = lambda x: (np.isnan(x)|(~np.isfinite(x)))
#    msk = check(a)|check(b)
   
#    idxs = np.where(msk)
#    return np.delete(a, idxs), np.delete(b, idxs)

# def dzip(a, b, acc=dict):
#    _acc = acc()
#    keys = list(set(a.keys()) & set(b.keys()))
   
#    for k in keys:
#       l, r = a.get(k, None), b.get(k, None)
#       a = tuple()
#       if isinstance(l, tuple):
#          a = (a, *l)
#       else:
#          a = (l,)
#       if isinstance(r, tuple):
#          a = (*a, *r)
#       else:
#          a = (*a, r)
#       _acc[k] = a
   
#    return _acc

# def wjoin(join, x, y):
#    if x == '' or x is None:
#       return y
#    elif y == '' or y is None:
#       return x
#    return join(x, y)

# def dflatten(d:Dict[str, Any], join=None, acc=None, prefix='')->Dict[str, Any]:
#    if join is None:
#       join = lambda x, y: f'{x}_{y}'
   
#    if acc is None:
#       acc = {}
#    for k, v in d.items():
#       if isinstance(v, dict):
#          dflatten(v, join=join, acc=acc, prefix=wjoin(join, prefix, k))
#       else:
#          acc[wjoin(join, prefix, k)] = v
#    return acc

# def polyscore(ytrue, ypred, columns=None, robust=True):
#    assert ytrue.ndim == 2 and ytrue.ndim == ypred.ndim
   
#    ytrue, ypred = transpose(ytrue, ypred)
#    cols = ytrue.shape[0]
#    keys = columns if columns is not None else list(range(cols))
#    scores = {}
#    for i, k in enumerate(keys):
#       scores[k] = score(ytrue[i, :], ypred[i, :], robust=True)
#    return scores
   
# # @njit
# def isinvalid(x:ndarray, additionalInvalids=None):
#    cond = (x == np.inf)|(x == -np.inf)|(x == np.nan)
#    if additionalInvalids is not None:
#       for v in additionalInvalids:
#          cond |= (x == v)
#    return cond

# # @jit
# def assertNoInvalid(x:np.ndarray, invalids=None):
#    inv = isinvalid(x, invalids)
#    err = ValueError('x contains inf, -inf, or NaN')
#    if np.isscalar(inv):
#       if inv == True:
#          raise err
#    elif np.count_nonzero(inv) > 0:
#       raise err
#    return True
      

# def score_model(model:Model, data:dsd.DataLoaderResult):
#    ytrue, ypred = applymodel(model, data, mode='test', scale=False)

#    if np.count_nonzero(ypred[isinvalid(ypred)]) > 0:
#       locs,_ = np.where(ypred[isinvalid(ypred)])
#       raise ValueError(f'values in ypred are not valid, located at {locs}')
#    if np.count_nonzero(ytrue[isinvalid(ytrue)]) > 0:
#       locs, _ = np.where(ytrue[isinvalid(ytrue)])
#       raise ValueError(f'values in ypred are not valid, located at {locs}')

#    meand = np.mean(np.abs(ypred - ytrue)/ytrue)  # mean difference between y and ypred
#    # corr, pval = pearsonr(y, ypred)

#    return meand

# def preprocess(model, x, y):
#    try:
#       return model.preprocess(x, y)
#    except AttributeError:
#       return x, y

# def simple_evaluate_model(model:Model, spec, x, y, scalers=None):
#    n_steps = spec.n_steps
#    feature_columns = spec.feature_columns
#    target_columns = spec.target_columns

#    # y, ypred = applymodel(model, data, 'test')
#    x, y = preprocess(model, x, y)
#    ypred = model(x.astype('float32')).numpy()


#    if scalers is not None:
#       sm = {i:scalers[v] for i, v in enumerate(spec.target_columns)}
#       y = descale(sm, y)
#       ypred = descale(sm, ypred)
   
#    channel_scores = {}
#    y, ypred = transpose(y, ypred)
#    res = np.zeros((len(target_columns), 3))

#    for i, name in enumerate(target_columns):
#       r = channel_scores[name] = score(y[i, :], ypred[i, :])
#       res[i, :] = r

#    return res

# #! marked for merge with domainaware_evaluate
# def evaluate_model_multivar(model:Model, spec, ticker, data:dsd.DataLoaderResult, multistep=None):
#    """
#     #TODO 
#     split into:
#      - historical data
#       + from which a baseline margin-of-error can be measured
#       + when that MOE is above a certain threshold:
#         - if the correlation between true and predicted values is still sufficiently high, 
#           derive an offset coefficient by which to multiply the predicted values to bring them 
#           closer to the true values
#         - otherwise, mark as non-viable and abort
#      - evaluation data
#        - X, y
#    """
   
#    n_steps = spec.n_steps
#    feature_columns = spec.feature_columns
#    target_columns = spec.target_columns
   
#    data = impdf(ticker, hparams=spec)
   
#    assert type(data) is dsd.DataLoaderResult
   
#    if multistep is None:
#       model.trainable = False
#       y, ypred = applymodel(model, data, 'test')
#       ytrue, ypred = transpose(y, ypred)
#       print('evaluating model')
      
#       channel_scores = {}
#       res = np.zeros((len(target_columns), 4))
      
#       for i, name in enumerate(target_columns):
#          r = channel_scores[name] = score(ytrue[i,:], ypred[i,:])
#          res[i, :] = r
         
#       return res
      
   
#    else:
#       assert isinstance(multistep, int)
#       X, y = data.eget('X_test', 'y_test')

#       call_model = lambda x: model(x)[0].numpy()
#       ytrue, ypred = invoke_as_multistep(call_model, X, y, steps=multistep)
      
#       sm = {i: data.column_scaler[v] for i, v in enumerate(model.channels)}
#       dsy = descale(sm, y)
#       ytrue, ypred = descale(sm, ytrue), descale(sm, ypred)
      
#       stepsets = lambda y: [y[:, i, :] for i in range(multistep)]
      
#       ytss, ypss = stepsets(ytrue), stepsets(ypred)
#       channel_scores = {k:[] for k in target_columns}
#       for j in range(multistep):
#          tt, pp = ytss[j], ypss[j] #? equivalent to `ytrue` and `ypred` under non-multistep conditions, but at each numbered step
#          for i, name in enumerate(target_columns):
#             channel_scores[name].append(score(tt[i, :], pp[i, :]))

#       print(channel_scores)

# # from engine.utils import dediffcol

# def evaluate_model_univar(model:Model, spec, ticker=None, data:dsd.DataLoaderResult=None, multistep=None, mode='test', robust_return=False):
#    """
#     #TODO 
#     split into:
#      - historical data
#       + from which a baseline margin-of-error can be measured
#       + when that MOE is above a certain threshold:
#         - if the correlation between true and predicted values is still sufficiently high, 
#           derive an offset coefficient by which to multiply the predicted values to bring them 
#           closer to the true values
#         - otherwise, mark as non-viable and abort
#      - evaluation data
#        - X, y
#    """

#    n_steps = spec.n_steps
#    feature_columns = spec.feature_columns
#    target_column = spec.target_column

#    # data = impdf(ticker, hparams=spec)

#    assert isinstance(data, (dsd.DataLoaderResult, dsd.Dataset))

#    if isinstance(data, dsd.Dataset):
#       data = ensure_has_module(data, spec, ticker)
#       data = spec.prep_data(data)
#    else:
#       # data= spec.prep_data(data)
#       pass
   
#    model.trainable = False
   
#    X = getattr(data, f'X_{mode}')
#    y, ypred = applymodel(model, data, mode, column=target_column)

#    res = score(y, ypred, robust=robust_return)
#    if robust_return:
#       res.update(dict(X=X, y=y, ypred=ypred))
   
#    elif np.nan in res:
#       print(pd.DataFrame({'y': y, 'ypred': ypred}))
#       raise ValueError('invalid score')
   
#    print(pd.DataFrame({'y': y, 'ypred': ypred}))
#    model.trainable = True
   
#    return res

# def evaluate_model_rigorously(model:Model, spec, datas, multistep=None):
#    scores = []
   
#    avgmap = compose_left(funcs.attrgetter('T'), curry(map)(lambda x: np.mean(x)), tuple)
   
#    for name,data in datas.modules.items():
#       y, ypred = applymodel(model, data, 'test')
#       ytrue, ypred = transpose(y, ypred)
#       assertNoInvalid(ytrue)
#       assertNoInvalid(ypred)

#       res = np.zeros((len(spec.target_columns), 3))

#       for i, name in enumerate(spec.target_columns):
#          try:
#             r = score(ytrue[i, :], ypred[i, :])
#             res[i, :] = r
#          except ValueError as e:
#             raise ValueError(f'.score() failed on {data.ticker} with {e}')
      
#       res = dict(symbol=data.ticker, score=res, accum_score=avgmap(res))
#       scores.append(res)
      
#    for res in scores:
#       print(res['symbol'], res['score'])
   
#    return scores

# @ojit
# def invoke_as_multistep(invoke:Callable[[ndarray], ndarray], X:ndarray, y:ndarray, steps=14):
#    N, H, W = X.shape
#    X = X.reshape(N, 1, H, W)
#    nX = X.shape[0]
#    tstar = np.zeros((nX-steps, steps, y.shape[-1]))
#    ystar = np.zeros((nX-steps, steps, y.shape[-1]))
   
   
#    for i in range(nX-steps):
#       x = X[i, :]
#       ytrue = y[i:i+steps, :]
#       ypred = multistep(invoke, x, steps=steps, nchannels=y.shape[1])
#       if ystar is None:
#          ystar = np.zeros((nX-steps, *ypred.shape))
#       ystar[i, :] = ypred
#       # tstar.append(ytrue)
#       tstar[i, :] = ytrue
      
#    return tstar, ystar

# @ojit
# def multistep(f:Callable[[ndarray], ndarray], X:ndarray, steps=14, nchannels:int=4):
#    """
#    -
#       helper function for generating multistep predictions using a single-step estimator function.

#       Parameters
#       ----------
#       f : Callable[ndarray]
#          any function of the form `f(X) = y` taking a single argument of type ndarray, and returning an ndarray
#          any dimensionality should work. The only assumption made is that `X` is an array of `y`
#       X : ndarray
#       steps : int, optional
#          number of steps to calculate, by default 14

#       Returns
#       -------
#       y* : ndarray
#          the multistep predictions
#    """
#    rets = np.zeros((steps, nchannels), np.float32)
#    for x in range(steps):
#       # with objmode(y=float32[:, :]):
#       y = f(X)
#       if rets is None:
#          rets = np.zeros((steps, *y.shape))
#          # print(rets.shape)
#       # print(y.shape)
#       rets[x, :] = y
#       # rets.append(y)
#       nx = np.roll(X, -1, axis=0)
#       nx[-1, :] = y
#       X = nx
#    # return array(rets)
#    return rets

# def inspect_model(model:Model, data:dsd.Dataset, ticker:str=None):
#    """
#    -
#      generate live visualizations for model's performance
#      needs a fair bit of work
  
#      Parameters
#      ----------
# #     model : keras.Model
#      data : dsd.DataLoaderResult
#    """
#    ticker = random.choice(list(data.modules.keys()))
#    data = data.modules[ticker]
#    x, y = model.preprocess(*data.eget('X_test', 'y_test'))
#    ypred = model.predict(x, verbose=0, use_multiprocessing=True, workers=8)
#    sm = {i:data.column_scaler[v] for i,v in enumerate(model.hyperparams.target_columns)}
#    y = descale(sm, y)
#    ypred = descale(sm, ypred)
#    # trunc_size = math.floor(len(y) * 0.15)
   
   
#    y = y.T
#    ypred = ypred.T
#    # y = y[:, -trunc_size:]
#    # ypred = ypred[:, -trunc_size:]
#    # print(y.shape, ypred.shape)
   
#    # plt.lin
#    fig = plt.figure(constrained_layout=True, figsize=(12, 8))
#    fid = fig.number
#    print(fig.get_size_inches())
#    axd = fig.subplots(nrows=4, ncols=1, sharex=True, sharey=True)
#    plots = []
#    # plt.plot
#    x = np.arange(0, len(y[0]))
#    for i in range(4):
#       a,b = y[i], ypred[i]
#       # print(x.shape, a.shape, b.shape)
#       l1 = axd[i].plot(x, a, color='#439c5b')
#       l2 = axd[i].plot(x, b, color='red', linewidth=1.2)
#       plots.append((l1[0], l2[0]))
   
#    plt.show(block=True)
   
#    def update():
#       if plt.fignum_exists(fid):
#          y, ypred = tuple(map(lambda x:x.T, applymodel(model, data)))
         
#          for i in range(4):
#             a, b = y[i], ypred[i]
#             aline,bline = plots[i]
#             aline.set_ydata(a)
#             bline.set_ydata(b)
         
#          fig.canvas.draw()
#          fig.canvas.flush_events()
   
#    return update

# # @jit(parallel=True, cache=True)
# t = lambda x: tf.convert_to_tensor(x)
# def pairs(x, y):
#    # for i in range(len(x)):
#       # yield array([x[i]]), array([y[i]])
#    yield t(x), t(y)
   
# def dacc(x, y):
#    dx = directions(x)
#    dy = directions(y)
#    return (dx == dy).sum()/len(x)
   
# @jit
# def directions(x: float32[:]) -> int32[:]:
#    #? can be rewritten as a "stencil" function
#    r = np.zeros(len(x), dtype='int32')
#    for i in prange(1, len(x)):
#       p,c = x[i-1], x[i]
#       d = 0
#       if c > p:
#          d = 1
#       elif c < p:
#          d = -1
#       r[i] = d
#    return r


# def named_split_y_channels(x, y, channel_names):
#     ch = split_y_channels(x, y)
#     assert len(ch) == len(channel_names)
#     d = {}
#     for i in range(len(channel_names)):
#         d[channel_names[i]] = ch[i]
#     return d

# @jit
# def split_y_channels(x, y: ndarray):
#    nchannels = y.shape[-1]
#    bag = np.zeros((nchannels, *y.shape[:-1]))
#    for j in range(nchannels):
#       for i in range(len(y)):
#          bag[j, i] = y[i, j]
#    return bag

# fts_datas = []

# @curry
# def fitthemshits(x, y, nn):
#    return nn.fit(x, y, epochs=25, use_multiprocessing=True, workers=3)

   
# def train_model(model, data:dsd.DataLoaderResult, conv_thresh=0.0048, spec=None, export_path='./saved_model', fit_parameters=None, on_progress=None, incremental=False):
#    (x_train,  y_train) = model.preprocess(*data.eget('X_train', 'y_train'))
#    from ds.model.ctrl import MdlCtrl
#    # assert isinstance(model, MdlCtrl)
#    model:MdlCtrl = model
#    if spec is None:
#       spec = model.hyperparams
   
#    convergence = False
   
#    epochi = 5
#    nepochs = 0
   
#    avgmap = compose_left(funcs.attrgetter('T'), curry(map)(lambda x: np.mean(x)), tuple)
#    scoresByEpoch = {}
   
#    callbacks = []
#    fitKwargs = dict(
#        epochs=1,
#        use_multiprocessing=True,
#        workers=12,
#        validation_split=0.05,
#        callbacks=callbacks
#    )

#    if fit_parameters is not None:
#       fitKwargs.update(fit_parameters)
      
#    training_epochs = fitKwargs.get('epochs')
#    verbosity = fitKwargs.get('verbose', 1)
   
#    if not callable(on_progress):
#       on_progress = lambda *a: None
      
#    nn:Model = model.nn
#    best = None
#    capture_nn_state = nn.get_weights
      
#    def _score_nn():
#       #TODO due for update
#       if spec.target_univariate:
#          _score = evaluate_model_univar(model, spec, data=data, ticker='btc')
#          moe, offset, corr, dac = _score
#          moe, corr, dac = [n * 100 for n in (moe, corr, dac)]
#          scoresByEpoch[epoch] = (moe, offset, corr, _score)
#       else:
#          _score = evaluate_model_multivar(model, spec, data=data, ticker='btc')
#          if _score.shape[0] == 5:
#             _score = _score[:-1, :]

#          print(_score)
#          tscore = avgmap(_score)
#          moe, offset, corr, dac = tscore
#          moe, corr, dac = [n * 100 for n in (moe, corr, dac)]
#       return moe, corr, dac
   
#    _scorenn = _score_nn
      
#    def inlineEvaluation(epoch:int, logs):
#       from builtins import abs
      
#       moe, corr, dac = _scorenn()

#       status = moe
#       progress = (epoch / training_epochs)
#       on_progress(progress, status)
#       nonlocal best
#       if best is None or abs(moe) < abs(best.moe):
#          best = Struct(
#             epoch=epoch,
#             moe=moe,
#             corr=corr,
#             dac=dac,
#             weights=capture_nn_state()
#          )
      
#       report = [
#          '\n',
#          f'epoch #{epoch}:',
#          f'  margin_of_error: {moe}',
#          f'  dir. accuracy: {dac}',
#          f'  correlation w/ y: {corr}'
#       ]
#       # print(f'\nepoch #{epoch}: model scores a {moe*100}% margin of error\nwith a {corr*100}% correlation with ytrue')
#       if not verbosity == 0: 
#          print('\n'.join(report))
   
# #   evalCb = keras.callbacks.LambdaCallback(on_epoch_end=inlineEvaluation)
# #   pltredlr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.15, verbose=1, patience=5, cooldown=12)
   
#    callbacks.extend([
#       evalCb,
#       pltredlr
#    ])
   
#    import datetime
#    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# #   tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='batch', write_images=True, profile_batch=100)
#    callbacks.append(tensorboard_callback)
   
#    if incremental:
#       #? incremental mode allows for custom training-termination conditions and logic to be employed
#       epoch_ct = 0
      
#       def callforward(done=False, epochs=None, keepbest=False, score_fn=None):
#          if score_fn is not None:
#             nonlocal _scorenn
#             _scorenn = score_fn
#          #? callforward (inverse of a callback) function to perform a batch of training_epochs
# #         nn: keras.Model = model.nn
#          nonlocal epoch_ct
#          fkw = fitKwargs.copy()
#          if epochs is not None:
#             fkw['epochs'] = epochs
#          else:
#             epochs = fkw.get('epochs', 1)
         
#          if done:
#             if hasattr(model, 'nn_path'):
#                export_path = model.nn_path
            
#             if keepbest and best is not None:
#                nn.set_weights(best.weights)

#             model.save(export_path)
            
#             return model
         
#          else:
#             #? perform a batch of training epochs
#             fithistory = nn.fit(
#                x_train, 
#                y_train,
#                initial_epoch=epoch_ct,
#                **fkw
#             )
#             epoch_ct += epochs
#             return fithistory
      
#       return callforward
   
#    else:
#       model.fit(x_train, y_train, **fitKwargs)
      
#       if best is not None:
#          nn.set_weights(best.weights)
      
#       if hasattr(model, 'nn_path'):
#          export_path = model.nn_path
      
#       model.save(export_path)
#       inlineEvaluation(-1, None)

#       return model
