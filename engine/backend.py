from pymitter import EventEmitter
from datetime import datetime, timedelta
from typing import *
import kraken
import shelvex as shelve
from pprint import pprint
from math import floor, ceil
import numpy as np
import re
import pandas as pd
#import modin.pandas as pd
from numba import jit, njit
from numba.experimental import jitclass
from functools import *
import toolz as tlz
from cytoolz import *
from operator import attrgetter
from cytoolz import dicttoolz as dicts
from cytoolz import itertoolz as iters
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

from engine.utils import *
from engine.datasource import *
from engine.utypes import Order
from engine.trading import TradingEngineBase
from engine.mixin import EngineImplBase as Mixin

import ds.utils.bcsv as bcsv
from ds.utils.book import Book, TsBook, FlatBook
from ds.forecaster import NNForecaster
from ds.model.ctrl import MdlCtrl
from ds.model_train import descale
import pickle
import os, re
P = os.path
dt, td = datetime, timedelta
kr = kraken
partialmethod
import inspect

from .portfolio import Portfolio, LongPosition
from tools import waitForNetworkConnection, once, closure, dotget, dotset, flat, nor
import traceback as tb

from fn import _, F

"""
 TODO: implement an OrderQueue & OrderQueueContext interface to allow for the "invest"/"divest" 
       interface to still work as expected, but with the ability to have the corresponding calls push
       their Order objects onto the OrderQueue rather than be immediately submitted
"""

@curry
def find_pair(base:str, quote:str, table:pd.DataFrame):
   return (table['base'] == base)&(table['quote'].str.endswith(quote))

def divestible_holdings(e, bal:Dict[str, float]):
   b = bal.copy()
   cash = b[e.liquidity_symbol]
   # del b[e.liquidity_symbol]
   
   for sym, vol in bal.items():
      if vol == 0.0 or not np.isfinite(vol) or sym == e.liquidity_symbol:
         del b[sym]
         continue
      
      ksym = krakenSymbol(sym)
      pair = nor(e.get_pair(sym), e.get_pair(ksym))
      if pair is None:
         print(f'no pair found for {sym}')
      
      if vol < pair.ordermin:
         del b[sym]
         continue
   
   b[e.liquidity_symbol] = cash
   return b

class RemoteBackend:
   client: kraken.KrakenAPI
   portfolio:Portfolio
   order_queue:List[Order]

   def __init__(self):
      self.client = kraken.krakenClient()
      # self.pairs = kraken.get_pairs(self.client)
      self.pair2symbol = None
      self.portfolio = Portfolio()
      self.portfolio.attach(self)
      self.order_queue = []
      self.balance_reserved = {}
      
      self._quick_fill = True
      
   def reserveBalance(self, sym:str, amnt:float):
      self.balance_reserved.setdefault(sym, 0.0)
      self.balance_reserved[sym] += amnt
      
   # @TradingEngineBase.liquidity
   @property
   def liquidity(self):
      return (self.balance[self.liquidity_symbol] - self.balance_reserved.get(self.liquidity_symbol, 0))
      
   @property
   def pairs(self): return self.book.pairs
      
   def push_order(self, order:Order):
      if order is None:
         return None
      order.validate = self.test_mode
      self.order_queue.append(order)
      
   def flush_order_queue(self):
      waitForNetworkConnection()
      toAwait = []
      
      while len(self.order_queue) > 0:
         order = self.order_queue.pop(0)
         print(order)
         
         if order.on_flushed is not None:
            try:
               res = self.client.add_standard_order(**order.to_dict())
            except Exception as e:
               res = e.with_traceback(tb.extract_stack())
            print(res)
            order.on_flushed(self, res)
            if order.txid is not None:
               toAwait.append(order)
         else:
            res = self.client.add_standard_order(**order.to_dict())
      
      if len(toAwait) > 0:
         awaitFilled(self, toAwait)
         
   def position_for(self, sym:str)->Optional[LongPosition]:
      return self.portfolio.query(lambda p: p.symbol == sym and not p.isclosed)
   
   def get_pair(self, symbol:str=None):
      from krakendb import ExchangePair, get_pair
      # try:
      #    qs = ExchangePair.objects(base=symbol)
      #    print(qs, dir(qs))
      #    return self.pairs[self.pair2symbol[symbol]]
      # except KeyError:
      #    return self.pairs.loc[find_pair(symbol, self.liquidity_symbol)].iloc[0]
      altsym = krakenSymbolsymbol)
      _pair = (
         get_pair(base=symbol, quote=f'{self.liquidity_symbol}'),
         get_pair(base=symbol[1:], quote=self.liquidity_symbol),
         get_pair(base=altsym, quote=self.liquidity_symbol)
      )
      
      for v in _pair:
         if v is None:
            continue
         else:
            return v
      raise ValueError(f'No pair found for {symbol}')
      return None
         
      
   def liquidate(self, exclude=[]):
      marked = divestible_holdings(self, self.balance)
      for sym in marked.keys():
         if sym == self.liquidity_symbol:
            continue
         
         self.divest(sym, weight=1.0)
      
      self.flush_order_queue()
      
      return self

   def invest(self, symbol=None, weight:float = 0.02, volume=None, instant=False):
      if self.book is None:
         raise ValueError('No data available')  # for symbol %s' % symbol)
      
      price = self.cur_price_of(symbol)

      investable_dollars = min(self.maximum_investment, self.liquidity)
      volume = ((investable_dollars * weight)/price) * 0.9
      _symbol = symbol

      symbol = krakenSymbol(symbol)
      pair_code = self.pair2symbol[symbol]
      pair = self.get_pair(symbol)
      
      volume = round(volume, pair.lot_decimals)
      volume = max(pair.ordermin, volume)
      cost = (volume * price)
      fee_cost = (cost * 0.0026)
      self.reserveBalance(self.liquidity_symbol, cost + fee_cost)
      
      lpos = LongPosition(symbol=symbol, pair=pair_code, quantity=volume)
      if not instant:
         lpos.buy_price = price
         
      oporder = self.portfolio.open_position(lpos)
      oporder.validate = self.test_mode
      
      if self._quick_fill:
         #! are we positive we want to be doin' this?
         #? it's obviously not a fuck-up-proof situation, but it's more expedient at the moment
         oporder.price = None
         oporder.ordertype = 'market'
         
      import traceback as tb
      
      def _done(e, status):
         if isinstance(status, Exception):
            err_report = tb.format_exception(type(status), status, status.__traceback__)
            self.portfolio.positions.remove(lpos)
            print('\n', err_report)
         else:
            from termcolor import colored
            print(colored(f'Position opened on {symbol}!! (x{volume}', 'green'))
            if isinstance(status, dict):
               status:Dict[str, Any] = status
               if 'txid' in status:
                  oporder.txid = status['txid']
                  # awaitFilled(self, oporder)
               
               descr = dotget(status, 'descr.order', None)
               if descr is not None:
                  print('\n', colored(descr.upper(), 'green', attrs=['dark']))
      
      oporder.on_flushed = _done
      # raise ValueError('everything is fine')
      return lpos

   def divest(self, symbol, weight: float = 1.0, **kwargs):
      symbol = krakenSymbol(symbol)
      pos = self.position_for(symbol)
      if pos is None:
         return self.noposition_divest(krakenSymbolsymbol), weight, **kwargs)
      
      corder = self.portfolio.close_position(pos)
      
      if self._quick_fill:
         #! are we positive we want to be doin' this?
         #? it's obviously not a fuck-up-proof situation, but it's more expedient at the moment
         oporder.price = None
         oporder.ordertype = 'market'
      
      #TODO handle selling positions for which we have no record of opening as well
      print(corder)
      
      return pos
   
   def noposition_divest(self, symbol, weight=1.0, **kwargs):
      held = self.balance.get(symbol, 0.0)
      if held == 0:
         self.sync_balance()
         held = self.balance.get(symbol, 0.0)
         assert held > 0, f'Cannot divest from token "{symbol}" in which we have no holdings'
      
      amntToLiq = (self.balance[symbol] * weight)
      
      usrSymbol = symbol
      symbol = krakenSymbol(symbol)
      pairId = nor(self.pair2symbol.get(symbol, None), self.pair2symbol.get(symbol[1:], None))
         
      pair = self.get_pair(symbol)
      if pair is not None:
         print(pair.name, pair.base, pair.quote)
         assert pair.quote.endswith(self.liquidity_symbol)
         if pairId is None:
            pairId = pair.name
      
      # amntToLiq = amntToLiq
      if amntToLiq < pair.ordermin:
         if self.ignore_balance is None: 
            self.ignore_balance = []
         self.ignore_balance.append(symbol if symbol in self.balance else usrSymbol)
         return None
      order = Order(
         pair=pairId, 
         type='sell', 
         ordertype='market', 
         volume=amntToLiq,
         userref=np.int32(hash('mlbot_v0')),
         validate=(self.test_mode),
         starttm=None,
         expiretm=None
      )
      print(order)
      self.push_order(order)

   def liquidate_lowvolume(self, symbol):
      if self.is_live:
         symbol = ksymbol
      pair_match = self.pairs[self.pairs.eval(
          f'(base == "{symbol}" or base == "X{symbol}") and quote == "ZUSD"')]
      if len(pair_match) == 0:
         print('yo wtf u talkin bout, bro')
         return None

      pair = pair_match.iloc[0]
      pair_code = pair_match.index[0]
      o = Order(pair=pair_code, type='buy', ordertype='market',
                volume=pair.ordermin, validate=(not self.deployed))
      self.client.add_standard_order(**o.to_dict())
      sleep(1.0)
      self.sync()
      return self.divest(symbol)
   
from time import sleep

def awaitFilled(be:RemoteBackend, order:Iterable[Order]):
   iter
   txids = flat(filter(
      lambda x: (x is not None), 
      map(lambda x: x.txid, order)
   ))

   if len(txids) == 0:
      return True
   
   while len(txids) > 0:
      # because this loop could actually run for a while sometimes, let's try and handle it gracefully if the network is lost
      waitForNetworkConnection() 
      print(txids)
      
      cof,nb = be.client.get_closed_orders()
      cof:pd.DataFrame = cof
      # print(cof)
      closed_txids = cof.index
      print(closed_txids)
      
      closed = [i for i in txids if i in closed_txids]
      print(closed)
      
      for x in closed:
         txids.remove(x)
      
      sleep(3.0)
   
   # orderInfoFrame:pd.DataFrame = be.client.query_orders_info(','.join(txids))
   # oif = orderInfoFrame
   # oif['opentm'] = pd.to_datetime(oif.opentm, unit='s')
   return True
   