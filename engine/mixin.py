from .trading import *
from .utils import *
from .utils import _reducer
from .utypes import MultiAdvisorPolicy, FrozenSeries
from fn import _, F
from time import time
from numba import jit, vectorize, generated_jit, prange
from numba.experimental import jitclass
import numba as nb
from typing import *
from tools import closure, nor, unzip
from ds.gtools import ojit
from .trading import TradingEngineBase
from frozendict import frozendict

from cytoolz import *
from tools import getcols, getcolsmatching, dotget

import engine.eco.forecasts as fc
from engine.eco.forecasts import *
from engine.data.argman import ArgumentManager
from engine.data.fcman import ForecastManager
from engine.logging import LogFile, PickleLogFile

class EngineImplBase(TradingEngineBase):
   adv_policy:MultiAdvisorPolicy
   _todays_advice:List[Forecast]
   _yesterdays_advice:List[Forecast]
   
   def __init__(self):
      super().__init__()
      
      self.adv_policy = MultiAdvisorPolicy.MergeThenFollow
      self.adv_policy_params = None
      self.prev_advice = None
      self.ignore_balance = None
      self.fc_log = []
      
      self.logger = PickleLogFile('./engine_logs.pklog')
      # self.logger = MongodbLogFile()
      self.logger.clear()
      self.argm = ArgumentManager(owner=self)
      self.fm = ForecastManager(owner=self)
      
      self._logging_enabled = False
      self._todays_advice = None
      self._yesterdays_advice = None
      
   def log(self, *message, label='generic'):
      if len(message) == 1:
         message = message[0]
      self.logger.put(message, label=label)
   
   def liquidate(self, exclude=[]):
      if self.ignore_balance is not None:
         exclude = (*self.ignore_balance, *exclude)
      
      super().liquidate(exclude=exclude)
      
      tokens_still_held = list(valfilter(_ > 0.0, dissoc(self.balance, self.liquidity_symbol, *exclude)).keys())
      # print(tokens_still_held, self.balance)
      
      if hasattr(self, 'test_mode') and self.ignore_balance is None:
         self.ignore_balance = tokens_still_held
         tokens_still_held = []
         
      # elif hasattr(self, 'test_mode') and self.ignore_balance 
      
      if len(tokens_still_held) > 0:
         print(tokens_still_held)
         if set(tokens_still_held).issubset(self.ignore_balance):
            return True
         
         raise ValueError('Failed to entirely liquidate holdings')
   
   def cfor(self, date=None, val=None, safe=False):
      date = date if date is not None else self.current_date
      r = {}
      if val is None:
         val = identity
      getrow = lambda doc: doc.loc[self.current_date]
      if (self.freq == pd.DateOffset(days=1) or self.freq == pd.DateOffset(hours=1)):
         getrow = lambda doc: doc.iloc[doc.index.get_loc(self.current_date, method='nearest')]

      if safe:
         for name, doc in self.book.items():
            doc: pd.DataFrame = doc
            try:
               today = getrow(doc)
               r[name] = val(today)
            except:
               # r[name] = None
               continue
      else:
         for name, doc in self.book.items():
            doc:pd.DataFrame = doc
            today = getrow(doc)
            r[name] = val(today)
      return r
   
   def ctoday(self, date=None, val=None):
      return self.cfor(date=date, val=val)

   def cwin(self, val=None, n=2):
      r = {}
      if val is None:
         val = identity
      for name, doc in self.book.items():
         try:
            indices = doc.index
            i = indices.get_loc(self.current_date)
            s, e = indices[i-n], indices[i]
            snip = doc.loc[s:e].tail(n)
            assert len(snip) == n, f'{len(snip)} != {n}'
            r[name] = val(snip)
         except KeyError:
            r[name] = None
            continue
      return r

   @memoize
   def prices_at(self, date=None):
      return self.argm.pricemap(date)

   def todays_prices(self):
      return self.prices_at(date=self.current_date)
   
   def cur_price_of(self, symbol:str):
      pm = self.todays_prices()
      return pm[symbol]
   
   def get_args(self, nnf, date=None, tailscale=True, asarray=False):
      features = valfilter(notnone, 
         self.argm.get_features(None, date, nnf.params.n_steps, scaled=tailscale)
      )
      
      return features
   
   def get_todays_args(self, nnf, asarray=False):
      return self.get_args(nnf, date=self.current_date, asarray=asarray, tailscale=True)
   
   def descale_y(self, sym:str, y:ndarray)->ndarray:
      sm = self.scalers[sym]
      transformers = [sm[c] for c in self.argm.columns]
      if y.ndim == 2:
         y = y.T
         res = np.empty_like(y)
         for i in range(len(self.argm.columns)):
            res[i] = transformers[i].inverse_transform(y[i].reshape(-1, 1))[:, 0]
         return res.T
      
      elif y.ndim == 1:
         res = np.zeros(len(y))
         for i in range(len(self.argm.columns)):
            res[i] = transformers[i].inverse_transform([[y[i]]])
         return res
      
      raise TypeError(y)
   
   def forecast_inner(self, fc:NNForecaster, X:Dict[str, np.ndarray], descaled=True)->Dict[str, np.ndarray]:
      out = {}
      bX = X
      #* would probably get a speed boost here by explicitly calling the underlying XLA graph for the Neural Network directly
      for sym, X in bX.items():
         y = fc([X])
         # pre = repr(y)
         if descaled:
            y = self.descale_y(sym, y)
         # post = repr(y)
         out[sym] = y
      
      return out

   def fcapply(self, fc=None, date=None, fcn=None):
      if date is None: date = self.current_date
      if fcn is not None: fc = fcn
      
      if fc is not None:
         if isinstance(fc, str): 
            fcn = fc
            fc = self.forecasters[fc]
         elif fcn is None:
            fcn = self._fcn(fc)
         
         X = self.get_args(fc, date=date)
         ypred = self.forecast_inner(fc, X)
         if not self._logging_enabled:
            return ypred
         
         usX = self.get_args(fc, date=date, tailscale=False)
         usypred = self.forecast_inner(fc, X, descaled=False)
         
         y = {sym:self.argm.get_target(sym, date, scaled=False) for sym in self.symbols}
         
         for sym in self.symbols:
            if not all((sym in a.keys() for a in (X,ypred,usX,usypred))):
               continue
            
            sample_entry = dict(
               date=date,
               sym=sym,
               fcid=fcn,
               X=X[sym],
               ypred=ypred[sym],
               usX=usX[sym],
               usypred=usypred[sym],
               y=y[sym]
            )
            
            self.log(sample_entry, label='call_sample')
         
         return ypred
      
      else:
         return {k:self.fcapply(fc=k, date=date) for k in self.forecasters.keys()}
   
   def todays_advice_for(self, forecaster:Union[str, NNForecaster]=None, forecasts:Dict[str, np.ndarray]=None, k:int=None):
      if forecasts is None:
         return []
      
      #TODO strip
      fc = self._get_forecaster
      todays = self.book.at(self.current_date)
      
      projected_returns:List[Forecast] = []
      if hasattr(self, '_blacklist'):
         blacklist = set(self._blacklist)
      else:
         blacklist = set()
      
      for sym, tmrw in forecasts.items():
         if sym in blacklist:
            continue
         
         today = todays[sym]
         if isinstance(tmrw, np.ndarray):
            tmrw = pd.Series(data=tmrw.squeeze(), index=fc(forecaster).params.target_columns)
         
         cur_price = today.close
         nxt_price = tmrw.close

         roi_pred = (nxt_price / cur_price) - 1.0
         
         node = Forecast(sym=sym, current_data=today, predicted_data=tmrw, roi=roi_pred)
         if roi_pred > 0:
            projected_returns.append(node)
      
      ranked:List[Forecast] = list(sorted(projected_returns, key=lambda o: o.roi, reverse=True))
      fc_id = self._fcn(forecaster)
      
      for x in ranked:
         x.generated_by = fc_id
      
      return ranked
      
   def todays_advice(self):
      self._yesterdays_advice = self._todays_advice
      prev_nodes = self._yesterdays_advice
      if prev_nodes is not None:
         for node in prev_nodes:
            node.true_data = self.book[node.sym].loc[self.current_date].copy()
      
      fckeys = list(self.forecasters.keys())
      print(f'len(forecasters)={len(fckeys)}')
      
      if len(fckeys) == 0:
         raise ValueError("No forecaster provided")
      
      elif len(fckeys) == 1:
         nodes:List[Forecast] = self.fcaa()
         self.advice_nodes = nodes
      
      elif len(fckeys) >= 2:
         if self.adv_policy is MultiAdvisorPolicy.FollowConsensus:
            fc_maps = {k:(self.forecasters[k], self.fcapply(k)) for k in fckeys}
            advice = [self.todays_advice_for(forecaster=forecaster, forecasts=forecasts, k=None) for forecaster, forecasts in fc_maps.values()]
            centroid = compute_vinn_center(advice)
            nodes = self.advice_nodes = centroid
         
         elif self.adv_policy is MultiAdvisorPolicy.MergeThenFollow:
            forecasts:Dict[str, ndarray] = self.fcapply_all(merge_method='min')
            self.fc_log.append(forecasts)
            
            nodes:List[Forecast] = []
            for s,y in forecasts.items():
               if len(y.squeeze()) == 0:
                  continue
               
               fckw = dict(sym=s, predicted_data=Series(data=y.squeeze(), index=self.argm.columns))
               doc = self.book[s]
               
               if self.current_date in doc.index:
                  fckw['current_data'] = doc.loc[self.current_date][self.argm.columns].copy()
                  
                  price_now = price(fckw['current_data'])
                  price_pred = price(y.squeeze())
                  
                  roi_pred = (price_pred / price_now) - 1.0
                  fckw['roi'] = roi_pred
                  
                  try:
                     td = fckw['true_data'] = doc.loc[self.current_date + (self.freq * 1)][self.argm.columns]
                     price_next = price(td)
                     roi_true = (price_next/price_now)-1.0
                     
                     roi_moe = (roi_pred - roi_true)
                     print(f'moe.roi({s}) = {roi_moe}')
                     # print('That have AIDs')
                  except KeyError:
                     pass
               
               node = Forecast(**fckw)
               
               nodes.append(node)
      
      if self.liquidity < self.maximum_investment:
         k = 20
      else:
         k = int(self.liquidity // self.maximum_investment)
      
      if isinstance(nodes, dict): 
         nodes = list(nodes.values())
      
      nnodes = []
      
      flags = []
      
      if self._yesterdays_advice is not None:
         for node in self._yesterdays_advice:
            if node.true_data is not None and node.current_data is not None:
               true_dir = (node.true_data.close / node.current_data.close)-1.0 > 0
               pred_dir = (node.roi > 0)
               
               if true_dir != pred_dir:
                  flags.append(lambda o: o.sym != node.sym)
            
            else:
               print(f'true_data not set for the "{node.sym}" node')
      
      nodes = list(filter(lambda o: all(f(o) for f in flags), nodes))
      ranked:List[Forecast] = list(topk(k, nodes, key=_.roi))
      
      self.advice_nodes = ranked
      self._todays_advice = ranked
      
      return ranked
   
   def fcaa(self):
      forecasts:Dict[str, ndarray] = self.fcapply_all(merge_method='min')
      self.fc_log.append(forecasts)
      
      nodes:List[Forecast] = []
      for s,y in forecasts.items():
         if len(y.squeeze()) == 0:
            continue
         
         fckw = dict(sym=s, predicted_data=Series(data=y.squeeze(), index=self.argm.columns))
         doc = self.book[s]
         
         if self.current_date in doc.index:
            fckw['current_data'] = doc.loc[self.current_date][self.argm.columns].copy()
            
            price_now = price(fckw['current_data'])
            price_pred = price(y.squeeze())
            
            roi_pred = (price_pred / price_now) - 1.0
            fckw['roi'] = roi_pred
            
            try:
               td = fckw['true_data'] = doc.loc[self.current_date + (self.freq * 1)][self.argm.columns]
               price_next = price(td)
               roi_true = (price_next/price_now)-1.0
               
               roi_moe = (roi_pred - roi_true)
               # print(f'moe.roi({s}) = {roi_moe}')
            
            except KeyError:
               pass
         
         node = Forecast(**fckw)
         
         nodes.append(node)
      
      return nodes
   
   def score_advice(self):
      return None
      if notnone(self.prev_advice):
         def getroi(sub: pd.DataFrame):
            prev,cur = sub.close.tolist()
            roi = (cur/prev)-1.0
            return roi
         todays_rois = self.cwin(getroi)
         
         lines = []
         todays = self.ctoday()
         for node in self.prev_advice:
            ytrue = todays_rois[node.sym]
            ypred = node.roi
            rep = f'{node.sym}   =>   ypred={ypred*100:.2f}%, y={ytrue*100:.2f}%'
            
            row_pred:pd.Series = node.forecast
            row_true:pd.Series = todays[node.sym][row_pred.index]
            
            offset = (row_pred - row_true).abs()
            row_moe = (offset/row_true)
            rep += f'   (moe={row_moe.mean()*100:.2f}%)'
            lines.append(rep)
            
         print('\n'.join(lines))
   
   def trade_on_forecasts(self):
      self.score_advice()
      
      cand:List[Forecast] = self.todays_advice()
      self.advice_nodes = cand

      if len(cand) == 0:
         return None
      
      # when we don't have enough liquidity to need to throttle investment magnitude
      elif self.liquidity <= self.maximum_investment:
         coef = min(3, len(cand))
         for i in range(coef):
            self.invest(cand[i].sym, (1 / coef))
         # self.invest(cand[0].sym)
         
      else:
         #* otherwise, call .invest() for each token suggested by the Advisor engine
         for rec in cand:
            #? it's up to the implementation of .invest() to handle the particulars of the distribution of liquidity among the suggested investments
            self.invest(rec.sym)
      
      self.prev_advice = cand
      
   def fcapply_all(self, date=None, merge_method='min'):
      if date is None: date = self.current_date
      appl = self.fm.get_forecast_primitives(date)
      return self.fm.agg_forecast_primitives(
         valmap(lambda d: list(d.values()), appl)
      )
      
   def set_adv_policy(self, policytype:MultiAdvisorPolicy, *params):
      self.adv_policy = policytype
      self.adv_policy_params = params if len(params) > 0 else None
      
   def set_adv_policy_params(self, *params):
      self.adv_policy_params = tuple(params)
      
from typing import *
      
def apply_star(self:EngineImplBase, *names:Iterable[str], merge_method='min', aggmap=None):
   if len(names) == 0:
      return None
   fcm = self.forecasters
   b:List[Dict[str, np.ndarray]] = [self.fcapply(fcn=name) for name in names]
   if all((len(d) == 0 for d in b)):
      return None
   
   zipped = dzip(*b)

   # ?(Dict of Tuple of Arrays) as opposed to DoDoA (Dict of Dicts (of arrays))
   DoToA = zipped
   for k,v in DoToA.items():
      assert isinstance(v, tuple), type(v).__name__
      for tv in v:
         assert isinstance(tv, ndarray), type(tv).__name__
   redux2 = partial(_reducer, method=merge_method)
   
   DoA = dunitize(DoToA, reducer=redux2)
   _names = self.forecasters[names[0]].params.target_columns
   merged_fc_rows = valmap((F() >> flatpred >> partial(Series, index=_names)), DoA)
   
   return merged_fc_rows

def apply_two(self:EngineImplBase, left_name:str, right_name:str, merge_method='min', aggmap=None):
   fcm = self.forecasters
   assert left_name in fcm, NameError('Unknown forecaster "{left_name}"')
   assert right_name in fcm, NameError('Unknown forecaster "{right_name}"')
   
   left, right = fcm[left_name], fcm[right_name]
   lcols, rcols = left.params.target_columns, right.params.target_columns
   
   assert set(lcols) == set(rcols), f'All forecasters must have the same output signature in order to be merged. {lcols} != {rcols}'
   cols = lcols

   l:Dict[str, np.ndarray] = self.fcapply(fcn=left_name)
   r:Dict[str, np.ndarray] = self.fcapply(fcn=right_name)
   
   # if 'aggmap' is provided, simply write our results onto it (it's a result dictionary), treating this function like a reduction callback
   if aggmap is not None:
      aggmap[left_name] = l
      aggmap[right_name] = r
      return aggmap

   redux = partial(_reducer, method=merge_method)

   np_res = dunitize(
       dzip(l, r),
       reducer=redux
   )

   return np_res
            
def chain_predicate(*predicates):
   return reduce(
      lambda l, r: (lambda *a, **b: l(*a, **b) and r(*a, **b)),
      predicates
   )  
   
def price(data: Union[np.ndarray, pd.Series]):
   if data is None:
      return None
   return data.mean()


from dataclasses import dataclass, asdict

@dataclass(repr=True)
class AdvNode:
   sym:str = None
   roi:float = None
   
   moe_min:float = None
   moe_max:float = None
   moe_avg:float = None
   
   forecast:Optional[pd.Series] = None
   
   def __post_init__(self):
      # if self.forecast is not None and isinstance(self.forecast, pd.Series):
      #    self.forecast = FrozenSeries.fromseries(self.forecast)
      pass

def isunscaled(arr: np.ndarray):
   cond = (arr > 1.0)|(arr < 0.0)
   
   return cond
   
from builtins import min, max
# def merge_advice_entries(a:AdvNode, b:AdvNode):
#    da, db = asdict(a), asdict(b)
#    dab = dzip(da, db)
#    dab = dissoc(valmap(lambda p: nor(p[0], p[1]), dab), 'sym', 'roi')
   
#    return AdvNode(
#       sym=a.sym, 
#       roi=max(a.roi, b.roi),
#       **dab
#    )

# def compute_vinn_center(advice_lists):
#    toadvmaps = F() >> (lambda dm: {d.sym:d for d in dm})
#    toadvmaps = F() >> (map, toadvmaps)
#    toadvmaps >>= list
   
#    gets = lambda d, *keys: tuple(d.get(k, None) for k in keys)
#    findlap = lambda m: reduce(lambda x, y: x & y, map(lambda x: set(x.keys()), m))
   
#    adv_maps = toadvmaps(advice_lists)
#    center = findlap(adv_maps)
#    return dzip(*toadvmaps([gets(m, *center) for m in adv_maps]))
   
def nd_to_row(columns:List[str], values:np.ndarray):
   values = flatpred(values)
   return pd.Series(data=values, index=columns)


def _getprice(doc:pd.DataFrame, date=date, safe=False, getrow=None):
   if not safe:
      today = getrow(doc) if getrow is not None else doc.loc[date]
      return today.close
   else:
      try:
         today = getrow(doc) if getrow is not None else doc.loc[date]
         return today.close
      except:
         return None

def _pricemap_at(self, date=None, val=None, safe=False):
   date = date if date is not None else self.current_date
   if val is None:
      val = identity
   getrow = None
   if not (self.freq == pd.DateOffset(days=1) or self.freq == pd.DateOffset(hours=1)):
      getrow = lambda doc: doc.iloc[doc.index.get_loc(self.current_date, method='nearest')]
      
   price = partial(_getprice, safe=safe, getrow=getrow)

   r = valmap(price, self.book.data)
   
   return r

def graph_pricemap(pricemap:Dict[str,float64], currency:str='USD'):
   from ds.graph import graph, bellman_ford
   pairs = [(sym, 'USD') for sym in pricemap.keys()]
   pg = graph()
   for (l, r) in pairs:
      pg.append(l, r, pricemap[l])
      pg.append(r, l, 1/pricemap[l])
   
   print(pg.reveal())
   print(bellman_ford(pg, 'ETH', 'DOGE'))
   
   return pg

def total_holdings_value(self:EngineImplBase):
   holdings = self.balance.copy()
   holdings = valfilter(lambda x: x > 0, holdings)
   currency = self.liquidity_symbol
   pricemap = self.prices_at(self.current_date)
   for sym, vol in holdings.items():
      if sym == currency:
         continue
      holdings[currency] += (vol * pricemap[sym])
      holdings[sym] = 0
   holdings = valfilter(lambda x: x > 0, holdings)
   return holdings[currency]