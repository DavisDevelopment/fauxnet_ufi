
import numpy as np
import pandas as pd
#import modin.pandas as pd
from fn import F, _
from cytoolz import *
from dataclasses import dataclass, asdict, astuple
from datetime import datetime, date
from .utils import *
from .utypes import Order
from tools import nor, nn
from typing import *

@dataclass
class LongPosition:
   pair:str = None
   symbol:str = None
   quantity:float = None
   
   opened_at:datetime = None
   closed_at:Optional[datetime] = None
   
   buy_price:float = None
   sell_price:float = None
   
   return_on_investment:Optional[float] = None
   
   @property
   def isopen(self):
      return nn(self.opened_at) and not nn(self.closed_at)
   
   def isclosed(self):
      return nn(self.opened_at) and nn(self.closed_at)
   
   def buy_order(self):
      okw = dict(pair=self.pair, type='buy')
      if nn(self.buy_price):
         okw = merge(okw, dict(
            ordertype='limit',
            volume=self.quantity,
            price=self.buy_price
         ))
      else:
         okw = merge(okw, dict(
            ordertype='market'
         ))
      
      return Order(validate=True, **okw)
   
   def sell_order(self):
      okw = dict(pair=self.pair, type='sell')
      if nn(self.sell_price):
         okw = merge(okw, dict(
             ordertype='limit',
             volume=self.quantity,
             price=self.sell_price
         ))
      
      else:
         okw = merge(okw, dict(
             ordertype='market'
         ))

      return Order(**okw)

class Portfolio:
   positions: List[LongPosition]
   owner:'EngineImpl'

   def __init__(self):
      self.position_history = None
      self.positions = []
      self.owner = None
      
   def attach(self, engine:'EngineImpl'):
      self.owner = engine
      return self

   def new_position(self, pos:LongPosition):
      self.positions.append(pos)
      
   def open_position(self, pos:LongPosition):
      if pos not in self.positions:
         self.positions.append(pos)
      
      if pos.opened_at is None:
         open_order:Order = pos.buy_order()
         assert nn(self.owner), 'open_position cannot be used with detached Portfolio instance, please use .attach() to attach Portfolio to an engine instance'
         # server_response = self.owner.client.add_standard_order(**open_order.to_dict())
         self.owner.push_order(open_order)
         # print(server_response)
         pos.opened_at = datetime.now()
         return open_order
      else:
         print('Warning: Cannot open a position which is already open')
         print('TODO: return the order that was used to open this position')
         return None
      return None
      
   def close_position(self, pos:LongPosition):
      if pos.closed_at is None:
         close_order = pos.sell_order()
         assert nn(self.owner), 'open_position cannot be used with detached Portfolio instance, please use .attach() to attach Portfolio to an engine instance'
         self.owner.push_order(close_order)
         pos.closed_at = datetime.now()
         
         return close_order
         
      return None
   
   def query(self, predicate:Callable[[LongPosition], bool]=None):
      if predicate is None:
         return None
      else:
         for p in self.positions:
            if predicate(p) == True:
               return p
         return None