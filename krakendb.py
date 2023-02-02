from mongoengine import *
from mongoengine.queryset.visitor import Q

import uuid

connect('fauxnet')

class ExchangePair(DynamicDocument):
   name = StringField(max_length=200, required=True, primary_key=True)
   altname = StringField(max_length=200, required=True)
   wsname = StringField(max_length=200, required=True)
   base = StringField(max_length=200, required=True)
   quote = StringField(max_length=200, required=True)

   cost_decimals = LongField(required=True)
   pair_decimals = LongField(required=True)
   lot_decimals = LongField(required=True)
   lot_multiplier = LongField(required=True)

   leverage_buy = ListField(default=[], required=False)
   leverage_sell = ListField(default=[], required=False)
   fees = ListField(null=True)
   fees_maker = ListField(null=True)

   fee_volume_currency = StringField(max_length=10, default='ZUSD')

   margin_call = LongField(required=True)
   margin_stop = LongField(required=True)
   ordermin = FloatField(required=True)
   
class OHLC(DynamicDocument):
   dtime = DateTimeField(required=True)
   symbol = StringField(required=True)
   open = FloatField(required=True)
   high = FloatField(required=True)
   low  = FloatField(required=True)
   close = FloatField(required=True)
   volume = FloatField(required=True)
   count = FloatField(required=True)
   
   # def __str__(self):
   #    self.__dict__

def get_pair(name=None, base=None, quote=None):
   # return ExchangePair.objects(name__regex=f'^\w?{base}\w?{quote}$')
   return ExchangePair.objects(base__regex='X?'+base+'$', quote__regex='(Z|X)?'+quote+'$').first()
   # qkw = dict(name=name, base=base, quote=quote)
   # if name is None:
   #    del qkw["name"]
   #    assert base is not None and quote is not None
   #    rq = Q(quote__endswith=quote)
   #    # _q = ((Q(base=base) & rq) | (Q(base=f'X{base}') & rq))
   #    _q = Q(base__endswith=base)
   #    q = Q(name__regex=f'X?{base}Z?{quote}')
   #    q |= _q
      
   # elif name is not None:
   #    del qkw['base']
   #    del qkw['quote']
   #    q = Q(name=name)
   
   # return ExchangePair.objects(name__regex=f'\w?{base}\w?{quote}').first()

# OHLC.inde
# import kraken as kr
import pandas as pd
#import modin.pandas as pd
import json
from cytoolz import *

# pairs_csv = pd.read_csv('./kraken_cache/tradable_pairs.csv', index_col='name')
# for c in ['leverage_buy', 'leverage_sell', 'fees', 'fees_maker']:
#    pairs_csv[c] = pairs_csv[c].apply(lambda s: json.loads(s))

def upsert_csv_to_db():
   for name, row in pairs_csv.iterrows():
      if ExchangePair.objects(name=name).count() == 0:
         kw = dict(name=name, **row.to_dict())
         print(kw)
         pair = ExchangePair(name=name, **row.to_dict())
         pair.save()
         
def upsert_ohlc_to_db(id:str, doc:pd.DataFrame):
   rows = []
   for dt, row in doc.iterrows():
      kw = dissoc(row.to_dict(), 'dtime')
      qs = OHLC.objects(symbol=id, dtime=dt).as_pymongo()
      print(qs)
      
      if len(qs) != 0:
         print(qs[0])
         continue
      
      dbrow = OHLC(dtime=dt, symbol=id, **kw)
      rows.append(dbrow)
   
   for x in rows:
      x.save()
   
   return rows

def upsert_book(docs):
   if not isinstance(docs, dict) and hasattr(docs, 'data') and isinstance(docs.data, dict):
      docs = docs.data
   if isinstance(docs, dict):
      for name, doc in docs.items():
         upsert_ohlc_to_db(name, doc)
         
def query_ohlc(symbol:str, *qparts):
   qs = Q(symbol=symbol)
   
   for pt in qparts:
      if isinstance(pt, Q):
         qs |= pt
      elif isinstance(pt, dict):
         qs |= Q(**pt)
      else:
         raise TypeError(f'Unhandled qpart of type {type(pt)}', pt)
   
   return OHLC.objects(qs)

if __name__ == '__main__':
   upsert_csv_to_db()