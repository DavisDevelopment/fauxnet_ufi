from sklearn.base import TransformerMixin
from builtins import id as oid
from json import dumps as json_encode
from time import sleep
from pymitter import EventEmitter
from datetime import datetime, timedelta
from datetime import date as date
from typing import *
import kraken
import shelvex as shelve
from pprint import pprint
from math import floor, ceil
import numpy as np
import re
import pandas as pd
#import modin.pandas as pd
from numba import jit, njit, vectorize
from numba.experimental import jitclass
from functools import *
import toolz as tlz
from cytoolz import *
from operator import attrgetter
from cytoolz import dicttoolz as dicts
from cytoolz import itertoolz as iters
from fn import _
from bidict import bidict
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

from engine.utils import *
from engine.datasource import *
from engine.utypes import Order

import ds.utils.bcsv as bcsv
from ds.utils.book import Book, TsBook, FlatBook
from ds.model.spec import Hyperparameters
from ds.forecaster import NNForecaster
from ds.model.ctrl import MdlCtrl
from ds.model_train import descale
from ds.gtools import TupleOfLists, Struct
from collections import deque
import asyncio as aio
# collections.deque
import pickle
import os
P = os.path
dt, td = datetime, timedelta
kr = kraken


class PeonMethods:
    def _get_forecaster(self, ref: Union[str, NNForecaster]) -> NNForecaster:
        return ref if not isinstance(ref, str) else self.forecasters[ref]

    def convert_forecasts_to_rows(self, nnf, forecasts: Dict[str, np.ndarray]) -> Dict[str, pd.Series]:
        columns = nnf.params.target_columns
        d = self.book[self.book.keys()[0]]
        freq = self.freq

        ret = {}

        for name in forecasts.keys():
            pred = flatpred(forecasts[name])
            if len(pred) == 1 and len(columns) > 1:
                pred = np.asarray([pred[0]] * len(columns))
            pred_row = pd.Series(data=pred, index=columns)
            pred_row['time'] = (self.current_date + (1 * freq))

            ret[name] = pred_row

        return ret

# TODO: split this into two classes, {paper,real}_trading_engine with a base class


@profileclass
class TradingEngineBase(EventEmitter, PeonMethods):
    balance: Dict[str, float] = None
    book: DataSource
    current_date: datetime
    forecasters: Dict[str, NNForecaster]
    maximum_investment: float
    order_buffer: List[Order] = None
    pair2symbol: bidict[str, str] = None
    delta_mode: bool = False

    def __init__(self):
        super().__init__(wildcard=True)

        # self.exec_mode = exec_mode
        self.deployed = False
        self.maximum_investment = 25000
        self.balance = None
        self.liquidity_symbol = 'USD'

        self.forecasts = None
        self.book = None
        self.order_buffer = []
        self.pair2symbol = None
        # self.pending_actions.clear
        # self._dates = None

        self.forecasters = {}
        self.scalers = None
        self._initialized = False
        self.freq = pd.DateOffset(days=1)

        self.forecast_history = []

        self._Scaler = MinMaxScaler

    def init(self):
        assert self.book is not None
        assert self.forecasters is not None

        # if not self.book.isuptodate():
        #    self.book.update()
        #    self.emit('data.update', self)

        pmap = self.pairmap = dict()

        psmap = self.pair2symbol = dict()
        # psmap.
        tpairs = self.book.pairs

        liqsym = krakenSymbol(self.liquidity_symbol)

        for pair_id, pair in tpairs.iterrows():
            pmap[nonkrakenSymbol(pair.base)] = pair
            psmap[nonkrakenSymbol(pair.base)] = pair_id

        self.pair2symbol = psmap = bidict(psmap.items())

        # * build our mapping from symbol to kraken-pair so that we don't have to do this every time we need the pair identifier
        #!

        self._initialized = True

        self.emit('initialized', self)

    def source(self, ds: DataSource):
        self.book = ds
        if hasattr(ds, 'freq'):
            self.freq = ds.freq
        return self

    def forecaster(self, name: str = None, fc: NNForecaster = None):
        if isinstance(name, dict):
            for k, v in name.items():
                self.forecaster(name=k, fc=v)
            return self
        print(f'forecaster "{name}"={fc}')
        assert fc is not None, 'cannot add null forecaster'
        if name is None:
            name = fc.params.target_column if fc.params.feature_univariate else tuple(
                fc.params.target_columns)
        self.forecasters[name] = fc
        return self

    def configure(self, var: str, value: Any):
        pass

    def config(self, cfg: Dict[str, Any]):
        for name, value in cfg.items():
            self.configure(name, value)
        return self

    def add_order(self, order=None, **kwargs):
        if order is not None:
            self.order_buffer.append(order)
        else:
            self.order_buffer.append(Order(**kwargs))

    def process_orders(self):
        orders: List[Order] = self.order_buffer[:]

        self.order_buffer = []

        for o in orders:
            if o.ordertype == 'market':
                if o.type == 'buy':
                    sym = self.pair2symbol.inverse[o.pair]
                    # TODO
                    print('TODO: not fully implemented')
        return False

    @property
    def symbols(self): return self.book.keys()

    @property
    def fckeys(self): return list(self.forecasters.keys())

    def _fcn(self, fc: NNForecaster) -> str:
        """
        get the name of the given Forecaster

        Parameters
        ----------
        fc : NNForecaster

        Returns
        -------
        str
            the key under which the Forecaster is stored

        Raises
        ------
        KeyError
            when no key is found
        """
        for k, nn in self.forecasters.items():
            if nn is fc:
                return k
        raise KeyError(fc)

    def holdings(self, symbol: str):
    # TODO [REFACTOR] offload to separate `TradeEnginePortfolio` class
        assert self._initialized, 'TradingEngine must be initialized before most memthods may be used'
        if notnone(self.balance):
            return self.balance.get(symbol, 0)
        else:
            raise ValueError('.balance is None')

    # TODO [REFACTOR] offload to separate `TradeEnginePortfolio` class
    def addHoldings(self, symbol: str, bal: float):
        symbol = nonkrakenSymbol(symbol)
        v = self.holdings(symbol)
        # print(valfilter(_ > 0, self.balance))
        self.balance[symbol] = v + bal

    # TODO [REFACTOR] offload to separate `TradeEnginePortfolio` class
    @property
    def liquidity(self):
        return self.holdings(self.liquidity_symbol)

    # TODO [REFACTOR] offload to separate `TradeEnginePortfolio` class
    def invest(self, symbol: str, weight: float = 1.0):
        pass

    # TODO [REFACTOR] offload to separate `TradeEnginePortfolio` class
    def divest(self, symbol: str, weight: float = 1.0):
        pass

    # TODO [REFACTOR] offload to separate `TradeEnginePortfolio` class
    def liquidate(self, exclude=[]):
        # exclude =
        for sym in self.balance.keys():
            if sym != self.liquidity_symbol and self.balance[sym] > 0:
                self.divest(sym)

    def sync(self):
        """
        syncronizes internal queue of actions to the system underlying the relevant implementation of TradingEngineBase
        """
        self.emit('sync', self)
        pass

    def build_scalers(self):
        me = self
        if me.scalers is not None:
            return me.scalers

        # * intelligently build a list of the columns that will need scaling
        cols = set()
        for f in me.forecasters.values():
            hp = f.params
            cols |= set(hp.target_columns)
        cols = list(cols)

        def scalermap(name: str):
            sm = {k: me._Scaler(copy=True, clip=True) for k in cols}
            for k, scaler in sm.items():
                setattr(scaler, 'symbol', name)
                setattr(scaler, 'column_name', k)
            return sm

        bk = me.book
        scalers = {}
        me.sc_statemap = bidict()

        # ? for each document (historical DataFrame) we have
        for name in bk.keys():
            doc = bk[name]
            doc = doc[cols]
            token_scalers = scalermap(name)
            scalers[name] = token_scalers
            # ? and for each column that will need to have scaling handled
            for c in cols:
                # ? using the available historical data for that column of that document
                data = doc[c].to_numpy()
                sc = token_scalers[c]
                # ? fit the Scaler to said data
                sc.fit(data.reshape(-1, 1))

        me.scalers = scalers

    def apply_scaling(self, nnf: NNForecaster, args: Dict[str, np.ndarray], invert=False):
        if args is None:
            return None
        elif self.scalers is None:
            self.build_scalers()

        args = valfilter(notnone, args)
        outputs = args.copy()
        columns = nnf.params.target_columns

        for name in args.keys():
            features = args[name]

            token_scalers = self.scalers[name]

            if nnf.params.feature_univariate:
                sc = self.scalers[name][nnf.params.target_column]
                scaled_features = (sc.transform if not invert else sc.inverse_transform)(
                    features.reshape(-1, 1))[:, 0]

            else:
                for k, scaler in token_scalers.items():
                    coli = columns.index(k)
                    assert scaler.symbol == name, 'u dun fukked up'
                    assert scaler.column_name == k, f'u steil fukked up; {k} != {scaler.column}'
                    call = (
                        scaler.transform if not invert else scaler.inverse_transform)
                    features[:, coli] = call(
                        features[:, coli].reshape(-1, 1))[:, 0]
                scaled_features = features

            if not invert and (scaled_features[scaled_features > 1].sum() != 0 or scaled_features[scaled_features < 0].sum() != 0):
                scaled_features[scaled_features > 1] = 1
                scaled_features[scaled_features < 0] = 0

            outputs[name] = scaled_features

        return outputs

    def get_all_args(self, nnf: NNForecaster, asarray=True):
        """
        -
         compute NN inputs for all available timestamps on all available/applicable symbols

         Parameters
         ----------
         nnf : NNForecaster
             the forecaster to be used 
         asarray : bool, optional
             deprecated, ignored, by default True

         Returns
         -------
         Dict[str, DataFrame]
             a dictionary of DataFrames, each with the same index as its corresponding 
             price-data frame, with 'X' and 'y' columns for the input/output pairs
        """

        from engine.data.prep import compileFromHyperparams as chp

        names = self.symbols
        hp = nnf.params
        seq_len = hp.n_steps
        columns = hp.target_columns
        fex = extractor = chp(hp)

        def scale(name: str, window, inplace=False):
            res = window.copy() if not inplace else window

            sm = {i: self.scalers[name][col] for i, col in enumerate(columns)}

            for i, col in enumerate(columns):
                scaler = sm[i]
                colval = window[:, i]
                colval = scaler.transform(colval.reshape(-1, 1))[:, 0]
                if ((colval > 1) | (colval < 0)).sum() > 0:
                    colval[colval > 1] = 1
                    colval[colval < 0] = 0
                res[:, i] = colval

            return res

        from tqdm import tqdm
        from engine.data.prep import feature_seqs

        if self.delta_mode:
            @fex.add
            def difference_df(df: pd.DataFrame):
                ndoc = df[columns].copy(deep=True)
                for c in columns:
                    ndoc[c] = ndoc[c].diff()
                return ndoc

        # * add the scaling method to the feature-extraction pipeline
        @fex.add
        def scaledf(df: pd.DataFrame):

            from ds import t
            ndoc = df.copy(deep=True)
            #! returned scalers are discarded, since we already have scalers of the same type, fitted to the same data
            _scalers = t.scale_dataframe(
                ndoc, columns=columns, scaler=self._Scaler)
            rdf = pd.DataFrame(data=ndoc, index=df.index, columns=df.columns)
            return rdf

        results = {}

        dropsyms = []
        pnames = tqdm(names)
        for name in pnames:
            # the historical data for the symbol denoted by `name`
            doc: pd.DataFrame = self.book[name]

            # ? drop those symbols which do not have sufficient data available
            if len(doc) < hp.n_steps:
                dropsyms.append(name)
                results[name] = None
                continue

            else:
                rdf = fex(doc)
                results[name] = rdf

        return results

    def get_args(self, nnf: NNForecaster, date: datetime = None, asarray=False, tailscale=False) -> Dict[str, np.ndarray]:
        raise ValueError('No arg-manager')
        """
      -
       get the NN inputs (for all applicable symbols), at the given date
 
       Parameters
       ----------
       nnf : NNForecaster
           the forecaster in question
       date : datetime, optional
           the date for which to compute the arguments, by default None
       asarray : bool, optional
           _description_, by default False
       tailscale : bool, optional
           _description_, by default False
 
       Returns
       -------
       Dict[str, np.ndarray]
           dictionary mapping symbols to the `X` value to be given to the forecaster as its argument
      """
        assert date is not None

        bk: TsBook = self.book
        names = self.symbols
        hp: Hyperparameters = nnf.params
        date = pd.to_datetime(date).date()

        assert date is not None

        findToday = indexOf(date, True)

        def locate(doc, row) -> int:
            if not hasattr(row, 'dtime'):
                print(row)

            return doc.index.get_loc(row.dtime)

        todayRows = dzip(bk.data, bk.at(date))
        todayRows = valfilter(lambda x: x[1] is not None, todayRows)
        todayIdxs = valmap(monasterisk(locate), todayRows)

        column = hp.target_columns if not hp.target_univariate else hp.target_column

        def extractargs(doc: pd.DataFrame, i: int) -> np.ndarray:
            subdf = doc.iloc[(i - hp.n_steps):i]
            if len(subdf) < hp.n_steps:
                return None

            data: np.ndarray = subdf[column].to_numpy()

            return data

        ret = {}
        for name in names:
            if not name in todayIdxs:
                ret[name] = None
                continue

            todayIdx = todayIdxs[name]
            if todayIdx == -1:
                ret[name] = None
                continue

            doc: pd.DataFrame = bk[name][bk.columns]
            ret[name] = extractargs(doc, todayIdx)

        if tailscale:
            ret = self.apply_scaling(nnf, ret)

        if asarray:
            keys = list(ret.keys())
            values = list(ret.values())
            return keys, values

        return ret

    def get_todays_args(self, nnf: NNForecaster, asarray=False):
        return self.get_args(nnf, date=self.current_date, asarray=asarray)

    def forecast(self, nnf: NNForecaster, args: Union[datetime, List[np.ndarray], Dict[str, np.ndarray]] = None, argkeys: List[str] = None, asrows=False):
        column = nnf.params.target_columns if not nnf.params.feature_univariate else nnf.params.target_column

        outputs = {}

        if isinstance(args, (datetime, date)):
            dti = args
            args = None
            args = self.get_args(nnf, dti, tailscale=True)

        if isinstance(args, dict):
            for name in args.keys():
                input = args[name]
                try:
                    output = nnf([input])
                    outputs[name] = output

                except Exception as e:
                    print(input)
                    raise e

        elif isinstance(args, (list, tuple, np.ndarray)):
            assert argkeys is not None
            assert len(argkeys) == len(args)

            outs = nnf(args)

            for i, name in enumerate(argkeys):
                outputs[name] = outs[i]

        else:
            raise ValueError('yo wtf u doin')

        outputs = self.apply_scaling(nnf, outputs, invert=True)

        if asrows:
            outputs = self.convert_forecasts_to_rows(nnf, outputs)

        return outputs

    def todays_advice(self, forecasts: Dict[str, pd.Series]):
        todays = self.book.at(self.current_date)

        projected_returns = []
        for sym, tmrw in forecasts.items():
            today = todays[sym]

            cur_price = today.close
            nxt_price = tmrw.close

            # pred_price = pred
            roi_pred = (nxt_price / cur_price) - 1.0
            projected_returns.append(Struct(sym=sym, roi=roi_pred))

        projected_returns = list(filter(_.roi > 0, projected_returns))

        if True or self.liquidity < self.maximum_investment:
            k = 3
        else:
            k = int(self.liquidity // self.maximum_investment)

        ranked = list(topk(k, projected_returns, key=_.roi))

        return ranked

    def fcapply(self, fcname: str = 'a') -> Dict[str, np.ndarray]:
        assert fcname in self.forecasters, f'No forecaster named {fcname} found in {list(self.forecasters.keys())}'

        nnf = self.forecasters[fcname]
        args = self.get_todays_args(nnf)
        # print(args)

        if args is None:
            raise ValueError('no arguments')

        # scaled_args = self.apply_scaling(nnf, args)
        scaled_args = args
        # scaled_args = valmap(lambda mat: mat, valfilter(notnone, scaled_args))

        forecasts = self.forecast(nnf, scaled_args)

        self.forecast_history.append(
            merge(forecasts, {'date': self.current_date})
        )

        return forecasts

    def run(self):
        raise NotImplementedError()

def flatpred(fc: np.ndarray):
    if fc.ndim == 2 and fc.shape[0] == 1:
        return fc[0, :]
    elif fc.ndim == 1:
        return fc
    else:
        raise ValueError('unrecognized forecast signature', fc.shape)
