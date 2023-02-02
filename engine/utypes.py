import pandas as pd
#import modin.pandas as pd
from enum import Enum
import inspect

from datetime import datetime, timedelta, date
from dataclasses import dataclass, asdict, astuple
from typing import *
from asyncio import Future
# from concurrent.futures import CFuture
from functools import *
from tools import closure, reduction
from numpy import ndarray
from cytoolz import *

@dataclass(eq=True, repr=True)
class Order:
    """
        Parameters
        ----------
        pair : str
            Asset pair.

        type : str
            Type of order (buy/sell).

        ordertype : str
            Order type, one of:
            market
            limit (price = limit price)
            stop-loss (price = stop loss price)
            take-profit (price = take profit price)
            stop-loss-profit (price = stop loss price, price2 = take profit
                price)
            stop-loss-profit-limit (price = stop loss price, price2 = take
                profit price)
            stop-loss-limit (price = stop loss trigger price, price2 =
                triggered limit price)
            take-profit-limit (price = take profit trigger price, price2 =
                triggered limit price)
            trailing-stop (price = trailing stop offset)
            trailing-stop-limit (price = trailing stop offset, price2 =
                triggered limit offset)
            stop-loss-and-limit (price = stop loss price, price2 = limit price)
            settle-position

        volume : str
            Order volume in lots. For minimum order sizes, see
            https://support.kraken.com/hc/en-us/articles/205893708

        price : str, optional (default=None)
            Price (optional). Dependent upon ordertype

        price2 : str, optional (default=None)
            Secondary price (optional). Dependent upon ordertype

        leverage : str, optional (default=None)
            Amount of leverage desired (optional). Default = none

        oflags : str, optional (default=None)
            Comma delimited list of order flags:
            viqc = volume in quote currency (not available for leveraged
                orders)
            fcib = prefer fee in base currency
            fciq = prefer fee in quote currency
            nompp = no market price protection
            post = post only order (available when ordertype = limit)

        starttm : int, optional (default=None)
            Scheduled start time:
            0 = now (default)
            +<n> = schedule start time <n> seconds from now
            <n> = unix timestamp of start time

        expiretm : int, optional (default=None)
            Expiration time:
            0 = no expiration (default)
            +<n> = expire <n> seconds from now
            <n> = unix timestamp of expiration time

        userref : int, optional (default=None)
            User reference id.  32-bit signed number.

        validate : bool, optional (default=True)
            Validate inputs only. Do not submit order (default).

        optional closing order to add to system when order gets filled:
            close[ordertype] = order type
            close[price] = price
            close[price2] = secondary price

        otp : str
            Two-factor password (if two-factor enabled, otherwise not required)
        """
    pair: str = None
    type: str = None
    ordertype: str = 'market'
    volume: Optional[float] = None
    price: Optional[float] = None
    price2: Optional[float] = None
    leverage: Optional[int] = None
    oflags: Optional[str] = None
    starttm: int = 0
    expiretm: int = 0
    userref: Optional[str] = None
    validate: bool = True
    close_ordertype: Optional[str] = None
    close_price: Optional[float] = None
    close_price2: Optional[float] = None
    otp = None
    trading_agreement: str = 'agree'
    filled = None

    on_placed: Optional[Callable[[Any, Any], Any]] = None
    on_filled: Optional[Callable[[Any, Any], Any]] = None
    on_flushed: Optional[Callable[[Any, Union[Exception, dict]], Any]] = None
    txid: Optional[str] = None


    def __post_init__(self):
        f: Future = Future()
        self.filled = f

    def update(self, **kwargs):
        self.__dict__.update(**kwargs)

    def to_dict(self):
        return dissoc(asdict(self), 'on_placed', 'on_filled', 'on_flushed', 'txid')

# class Order(_Order):
#     on_placed: Optional[Callable[[Any, Any], Any]] = None
#     on_filled: Optional[Callable[[Any, Any], Any]] = None
#     on_flushed: Optional[Callable[[Any, Union[Exception, dict]], Any]] = None

#     txid: Optional[str] = None


T = TypeVar('T')
class ServerValue(Generic[T]):
    __slots__ = ['data', 'utime']
    data: T
    utime: datetime

    def __init__(self, v):
        self.data = v
        self.utime = datetime.now()

    @property
    def age(self) -> timedelta:
        return (datetime.now() - self.utime)


class RVariance:
    def __init__(self, min=None, max=None, var=None):
        if min is None and max is None and var is not None:
            min, max = var
        self.min_value = min
        self.max_value = max

    def range(self, base=None):
        if base is None:
            base = 1.0
        return base * self.min_value, base * self.max_value


Tv = TypeVar('Tv')


def _rvinit(self, value, variance: RVariance = None):
    self.value = value
    self.variance = variance if variance is not None else RVariance()


class RVal(Generic[Tv]):
    def __init__(self, value=None, max=None, min=None, plusminus=None, variance=None):
        assert nn(value)
        def nn(v): return v is not None
        nor = reduction(lambda x, y: x if nn(x) else y)
        if nn(plusminus):
            min = nor(min, plusminus)
            max = nor(max, plusminus)
        if variance is None:
            variance = RVariance(min=min, max=max)
        self.value = value
        self.variance = variance

    @property
    def variance_range(self): return self.variance.range(self.value)

    @property
    def minimum(self): return self.variance_range[0]

    @property
    def maximum(self): return self.variance_range[1]

    def __float__(self): return float(self.value)
    def __int__(self): return int(float(self))
    def __str__(
        self): return f'{self.value:f} \u00B1 ~{self.value/((self.minimum+self.maximum)/2)*100:2f}%'


class MultiAdvisorPolicy(Enum):
    """
    the strategy used by the engine for converting output from multiple Forecasters into a trading plan

     - MergeThenFollow
      combine the multiple price forecasts into one, then generate a single set of advisements from that
     - FollowConsensus
      generate advisements for each forecast-set, then keep only the suggestions present in both advice-sets 
      ?(todays_advice_for(A) & todays_advice_for(B)) where A,B=
    """
    MergeThenFollow = 0
    FollowConsensus = 1


@dataclass(eq=True, unsafe_hash=True, frozen=False)
class FrozenSeries:
    index: Tuple[str]
    data: ndarray

    _fwd_directly_ = True

    def toseries(self): return pd.Series(data=self.data, index=self.index)
    def thaw(self): return self.toseries()
    
    def __getstate__(self):
        return dict(index=self.index, data=self.data)
    
    def __setstate__(self, state):
        self.index = state['index']
        self.data = state['data']

    def __repr__(self):
        return repr({k: v for k, v in zip(self.index, self.data)})

    def __getattr__(self, name):
        hasOwn = (name in self.index)
        if not FrozenSeries._fwd_directly_:
            assert hasOwn, NameError(
                f'FrozenSeries({repr(self)}) has no attribute "{name}"')
        if hasOwn:
            return self.data[self.index.index(name)]

        elif not hasOwn and hasattr(pd.Series, name) and callable(getattr(pd.Series, name)):
            return getattr(self.thawed, name)
        raise NameError(
            f'FrozenSeries({repr(self)}) has no attribute "{name}"')
        
    def __getitem__(self, index):
        return self.data[index]

    # ? this is used only for convenience, to allow execution of pd.Series methods on this class

    @cached_property
    def thawed(self): return self.thaw()

    @staticmethod
    def fromseries(s: pd.Series):
        return FrozenSeries(tuple(s.index), s.values)

import numpy as np
from numpy import ndarray, asarray
from tools import isiterable, hasmethod, ismapping
# from numba
from numba.experimental import jitclass


class npdict:
    _keys: ndarray
    _vals: ndarray
    
    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            a = args[0]
            if ismapping(a):
                return self.__init__(list(a.items()))
            elif isiterable(a):
                kl, vl = [], []
                for k, v in a:
                    kl.append(k)
                    vl.append(v)
                return self._init_(kl, vl)
            
        elif len(args) == 2:
            keys, vals = args[0], args[1]
            return self._init_(keys, vals)
        
        else:
            pairs = list(kwargs.items())
            return self.__init__(pairs)

    def _init_(self, keys, values):
        if not isinstance(keys, ndarray):
            keys = asarray(keys).astype('str')
        if not isinstance(values, ndarray):
            values = asarray(values)
        
        self._keys = keys.astype('str')
        self._vals = values
        self._frozen = False

    def keys(self): return iter(self._keys)
    def values(self): return iter(self._vals)
    def items(self): return zip(self._keys, self._vals)
    
    def freeze(self):
        self._frozen = True
    
    def __iter__(self): return self.keys()
    def __len__(self): return len(self._keys)
    
    def __contains__(self, key:str):
        return self._keys.__contains__(key)
    
    def _index(self, key:str):
        k = self._keys
        idxl, _ = np.where(k == key)
        if len(idxl) == 0:
            return -1
        else:
            return idxl[0]
    
    def __getitem__(self, key):
        i = self._index(key)
        if i == -1:
            raise KeyError(f'npdict key("{key}") not found')
        return self._vals[i]
    
    def __setitem__(self, key, value):
        if self._frozen:
            raise ValueError('cannot set value of npdict')
        elif self.__contains__(key):
            i = self._index(key)
            self._vals[i] = value
        else:
            raise KeyError(f'npdict key("{key}") not found')
    
    def __delitem__(self, key):
        if not self._frozen:
            i = self._index(key)
            if i == -1:
                return self
            k = np.delete(self._keys, i)
            v = np.delete(self._vals, i)
            self._keys = k
            self._vals = v
        else:
            raise ValueError('cannot delete item "{key}" of frozen npdict')
    
    def copy(self):
        return npdict(self._keys.copy(), self._vals.copy())
    
    def get(self, key, *rest, **kwargs):
        if 'default' in kwargs or len(rest) == 1:
            default = kwargs.pop('default') if 'default' in kwargs else rest[0]
            if not self.__contains__(key):
                return default
            return self.__getitem__(key)
        else:
            return self.__getitem__(key)
        
    def pop(self, key, *rest, **kwargs):
        if self._frozen:
            raise ValueError('pop cannot be called on frozen npdict')
        else:
            if self.__contains__(key):
                v = self.__getitem__(key)
                del self[key]
                return v
            else:
                if 'default' in kwargs or len(rest) == 1:
                    default = kwargs.pop('default') if 'default' in kwargs else rest[0]
                    return default
                else:
                    raise KeyError(f'key "{key}" not found')
    
    def valmap(self, val):
        return npdict(self.keys, list(map(val, self._vals)))
