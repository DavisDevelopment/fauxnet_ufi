from .trading import *
from .utils import *
from .utils import _reducer
from .utypes import RVal
from fn import _, F
from time import time
from numba import jit, vectorize, generated_jit, prange
from numba.experimental import jitclass
import numba as nb
from typing import *
from tools import closure, diffdicts, gets, _capctx
from ds.gtools import ojit
from .mixin import *

def correctfc(ideal):
   dates, syms, rois = ideal.date, ideal.sym, ideal.roi
   ideal['pairs'] = None
   ideal['best_roi'] = None
   ideal['best_sym'] = None
   
   for date in dates:
      mask = (ideal.date == date).to_numpy()
      sub = ideal[mask]
      s = sub[['sym', 'roi']]
      maxi = s.roi.to_numpy().argmax()
      pairs = ','.join([sym for (sym, roi) in sorted(zip(s.sym, s.roi), key=lambda x: x[1])])
      best = s.sym.iloc[maxi]
      ideal.loc[mask, 'pairs'] = pairs
      ideal.loc[mask, 'best_sym'] = best
      ideal.loc[mask, 'best_roi'] = s.roi.max()

@profileclass
class PaperTrading(EngineImplBase):
   metrics:Dict[str, Any]
   _dates:Optional[Iterable[pd.Timestamp]] = None
   
   def __init__(self, initial_balance=300.0):
      super().__init__()
      
      self.current_date = None
      self.balance = {
         self.liquidity_symbol: initial_balance
      }
      self.forecasts = {}
      self.argcache = None
      self._pmcache = None
      self.fccache = None
      self.metrics = {}
      self.transactions = []
      self.debug = False
      
      self._cache_pricemaps = True
      #TODO enumerate and blacklist all of the stablecoins, right out the gate
      self._blacklist = set(['USDT', 'USDC', 'DAI', 'PAXG'])
      self._use_local_cache = False
      
   @TradingEngineBase.liquidity.setter
   def liquidity(self, nv):
      assert np.isfinite(nv) and not np.isnan(nv) and nv > 0, f'invalid value for .liquidity {nv}'
      self.balance[self.liquidity_symbol] = nv
      
   def invest(self, symbol=None, weight:float=1.0):
      symbol = nonkrakenSymbol(symbol)
      price = self.cur_price_of(symbol)
      
      investable = min(self.maximum_investment, self.liquidity)
      volume = ((investable * weight)/price) * 0.9
      cost = (volume * price)
      fee_cost = (cost * 0.0026)
      # cost += fee_cost
      
      self.liquidity -= cost
      self.addHoldings(symbol, volume)
      self.transactions.append(('BUY', symbol, volume, price))
      
   def divest(self, symbol=None, weight:float=1.0, **kwargs):
      symbol = nonkrakenSymbol(symbol)
      assert weight > 0
      
      if self.balance[symbol] == 0:
         raise ValueError('cannot sell token we do not have')
      
      volume = (self.balance[symbol] * weight)
      
      price = self.cur_price_of(symbol)
      value = (volume * price)
      fee = (value * 0.0026)
      
      self.balance[symbol] -= volume
      # self.liquidity += (value - fee)
      self.liquidity += value
      
      self.transactions.append(('SELL', symbol, volume, price))
      
      return self.liquidity
    
   def init(self):
      super().init()
      
      begin, end = self.book.time_range
      _dates = pd.date_range(begin, end, freq=self.freq).tolist()
      self._dates = _dates
      
      self.argm.attach(self)
      self.argm.init()
      self.fm.attach(self)
      self.fm.init()
      
      self.build_scalers()
      self.argm.precompute_cache()
   
   def correct_forecasts(self):
      bk = self.book
      entries = []

      cols = {}
      for name, doc in bk.items():
            price_mvmnt = ((doc.open + doc.close)/2).pct_change()
            cols[name] = price_mvmnt

      rois = pd.DataFrame.from_dict(cols)
      res = pd.DataFrame(data=None, index=rois.index)
      res['best_roi'] = None
      res['correct_investment'] = None

      for date, row in rois.iterrows():
         res.loc[date].best_roi = row.max()
         res.loc[date].correct_investment = row.idxmax()

      return res
      
   def ideal(self, dates=None):
      cf = self.correct_forecasts()
      return cf
              
   def step(self):
      self.sync()
      self.trade_on_forecasts()
      
   def run(self, asgenerator=False):
      running = self._run(asgenerator=asgenerator)
      if asgenerator == True:
         return running
      else:
         for me in running:
            pass
      
   def _run(self, asgenerator=False):
      _dates = self._dates
      begin, end = self.book.time_range
      if self._use_local_cache:
         self.cache_all_args()
      
      ndays = 0
      starttime = time()
      sess = self.sess = pd.DataFrame(columns=('date', 'balance', 'score', 'params', 'bought', 'metrics'))
      
      prev_forecasts = None
      prev_bal = None
      init_balance = self.balance.copy()
      prev_balance = self.prev_liquidity = None
      balance_journal = []
      withdrawals = []
      pricemaps = []
      self.dayno = 0
      
      from termcolor import colored
      yield self
      
      for i, d in enumerate(_dates[:-1]): 
         d:pd.Timestamp = d
         self.dayno = i
         self.current_date = d

         sinceSessionBegan:timedelta = (d.to_pydatetime() - begin.to_pydatetime())
         
         entry = dict(date=d)
         prices = self.todays_prices()
         pricemaps.append((self.current_date, prices))
         
         if prev_forecasts is not None:
            p, score = cmpPricesToForecasts(prices, valmap(lambda a: a[-1][0], prev_forecasts))
            entry['score'] = score
            entry['params'] = p

         bought = valfilter(gt0, self.balance)
         del bought[self.liquidity_symbol]
         self.liquidate()
         
         if not asgenerator and self.liquidity <= (init_balance[self.liquidity_symbol] * 0.75):
            raise ItsAllShitty()
         
         balance_journal.append(self.balance.copy())
         
         entry = merge(entry, dict(balance=self.liquidity, bought=','.join(list(bought.keys()))))
         entry['metrics'] = self.metrics.copy()

         print(f'day #{sinceSessionBegan.days}:   ${self.liquidity:,.2f}')
         
         if i > 1 and sinceSessionBegan.days % 7 == 0 and self.liquidity > 7000:
            salary = min(self.liquidity * 0.07, 9000)
            self.liquidity -= salary
            withdrawals.append((self.current_date, salary))
         
         self.step()
         
         sess.loc[len(sess)] = entry

         prev_forecasts = None
         prev_bal = self.prev_liquidity = total_holdings_value(self)
         ndays += 1
         
         yield self
      
      self.liquidate()
      
      final_cash = self.liquidity
      bal = 150.0
      for t in self.transactions:
         kind, sym, volume, price = t
         if kind == 'BUY':
            bal -= (price * volume)
         elif kind == 'SELL':
            bal += (price * volume)
      
      print(f'reproducing the transaction log yielded a final balance of ${bal:,.2f}')
      
      endtime = time()
      print(f'took {(endtime - starttime)/60}min to run full simulation')
      
      pmapDump = pd.DataFrame(data=pricemaps)
      pmapDump.columns = ['date', 'prices']
      pmapDump.set_index('date', inplace=True)
      pmapDump.to_pickle('pmap_dump.pickle')
      
      pickle.dump(balance_journal, open('balances.pickle', 'wb+'))
      pickle.dump(self.transactions, open('transactions.pickle', 'wb+'))
      
      # sess = pd.DataFrame.from_records(sess)
      sess.set_index('date', inplace=True)
      sess['roi'] = sess['balance'].pct_change()
      sess.to_pickle('paper_sess.pickle')
      # if prev_sess is not None:
      #    sess['prev_bought'] = prev_sess['bought']
      #    sess['prev_balance'] = prev_sess['balance']
      #    sess['prev_roi'] = prev_sess['roi']
      print(sess)
      ret:pd.Series = sess.roi
      self.divest
      
      abstain_rate = ret[ret == 0].count()/len(sess) * 100
      loss_rate = ret[ret < 0].count()/len(sess) * 100
      total_roi = (self.liquidity / init_balance[self.liquidity_symbol])*100
      
      print(f"{abstain_rate:.2f}% of the time, the bot fails to identify any trade at all")
      print(f"{loss_rate:.2f}% of the time, the bot trades at a loss")
      
      import matplotlib.pyplot as plt
      
      ideal = self.ideal()
      ideal.to_pickle('ideal_sess.pickle')
      idret = ideal.best_roi
      sel = thestuff(ideal.correct_investment.tolist(), sess.bought.tolist())
      
      optimality_rate = len(sel)/len(sess)*100
      print(f'opt. rate: {optimality_rate:.2f}%')
      arg_cache = self.argcache
      
      pickle.dump(self.argcache, open('simargcache.pickle', 'wb+'))
      pickle.dump(balance_journal, open('sim_holdings.journal', 'wb+'))
      pickle.dump(self.fc_log, open('forecast_log.pickle', 'wb+'))
      
      vars = ('sess', 'ideal', 'balance_journal', 'arg_cache',
              'abstain_rate', 'loss_rate', 'optimality_rate',
              'total_roi')
      
      run_info = {k:v for k,v in zip(vars, gets(locals(), *vars))}
      
      self.gen_final_report(run_info)
      
      return sess
   
   def gen_final_report(self, run_info:Dict[str, Any]):
      vars = ['sess', 'ideal', 'balance_journal', 'arg_cache', 'abstain_rate', 'loss_rate', 'optimality_rate','total_roi']
      sess, ideal, balance_journal, arg_cache, abstain_rate, loss_rate, optimality_rate, total_roi = gets(run_info, *vars)
      
      pickle.dump(run_info, open('sim_run_info.pickle', 'wb+'))
      
      sess = run_info['sess']
      ideal = run_info['ideal']
      balance_journal = run_info['balance_journal']
      arg_cache = run_info['arg_cache']
      
      #* okay, let's talk numbers..
      the_numbers = ['loss_rate', 'abstain_rate',
                     'optimality_rate', 'total_roi']
      
      nbdisplay = pd.Series(
         index=the_numbers, 
         data=gets(locals(), *the_numbers)
      )
      
      print(nbdisplay)
      
@profile
def thestuff(correct, bought):
   indices = []
   print(len(correct), len(bought))
   for i in range(len(bought)):
      a:str = correct[i]
      b:str = bought[i]
      if a in b.split(','):
         indices.append(i)
   return indices

@profile
def cmpPricesToForecasts(prices, expected):
   err_moe = {}
   err_params = {}
   
   # print(set(prices.keys()) - set(expected.keys()))
   for k in prices.keys():
      y = prices.get(k, None)
      ypred = expected.get(k, None)
      if ypred is not None and y is not None:
         d = abs(y - ypred)/y
         err_moe[k] = d
         err_params[k] = (y, ypred)
      else:
         err_moe[k] = None
         err_params[k] = (y, ypred)
   
   return err_params, err_moe

@singledispatch
def gt0(x: Union[int, float, Any]):
   return x > 0
@gt0.register
def _0(x: np.ndarray):
   return (x > 0).all()

_noop = lambda x: x
_pgetstep = curry(lambda f, x: f(x) if x is not None else None)
_getkey = curry(lambda k, d: d.get(k, None))
_getkeys = curry(lambda keys, d: tuple([d.get(k, None) for k in keys]))
_nonanp = compose_left(
   lambda a: filter(notnone, a),
   list,
   lambda x: np.asarray(x, dtype='float64')
)

_moe_key_inner = lambda sym: compose_left(
    pgetstep(getkey('score')),
    pgetstep(getkey(sym))
)

def takefirst(it):
   return first(it)

@profile
def _errs_(yup:Iterable[Tuple[Dict[str, Tuple[float, float]], Dict[str, Optional[float]]]])->Dict[str, np.ndarray]:
   acc,names,extract,params = None, None, None, None
   
   for (pd, d) in yup:
      if names is None:
         names = list(d.keys())
         extract = _getkeys(names)
         params = [TupleOfLists(2) for name in names]
         # acc = TupleOfLists([TupleOfLists(2) for n in range(len(names))])
      
      for i, p in enumerate(extract(pd)):
         params[i].append(*p)
   
   nparams = []
   for i, name in enumerate(names):
      y, ypred = params[i]
      a2d = np.asarray([y, ypred]).T
      nparams.append(a2d)
   nparams = np.asanyarray(nparams)
   
   dparams = dpck(zip(names, nparams))
   return valmap(_nonanp, dparams)

@vectorize(cache=True)
def nmoe(y, ypred):
   return abs(y - ypred)/y

@vectorize(cache=True)
def overestimates(y, ypred):
   return ypred > y

@profile
def _moe_(params:Dict[str, np.ndarray], length=14):
   return dmoe(params)

def pctfor(a, mask):
   return np.count_nonzero(mask)/len(a)
         
def compmasks(masks:Dict[str, Callable[[np.ndarray], nb.boolean[:]]], a:np.ndarray):
   mskmap = valmap(lambda mkmsk: mkmsk(a), masks)
   probmap = valmap(partial(pctfor, a), mskmap)
   
   return dzip(mskmap, probmap)

# @jit(forceobj=True)
def aggparams(params:np.ndarray):
   res = None
   names = None
   for i in range(len(params)):
      pd = params[i]
      if pd is None:
         continue
      elif isinstance(pd, dict):
         names = list(pd.keys())
         break
   
   return {
      k: np.asarray([p[k] for p in params[i:]]).astype('float64') 
      for k in names
   }

@curry
def castto(t, a:np.ndarray):
   return a.astype(t)

@curry
def moegt(thresholds, a):
   masks = [(a > thresh) for thresh in thresholds]
   return [a[m] for m in masks]

def trirange(a: np.ndarray):
   return (
      np.min(a),
      np.mean(a),
      np.max(a)
   )

cleanAndAggregateParameters = F() >> aggparams >> (valmap, npdropna)

dmoe = F() >> (valmap, F() >> _.T >> monasterisk(nmoe) >> trirange)

def moe1(ytrue, ypred):
   d:np.ndarray = np.abs(ypred - ytrue)/ytrue
   return d.mean()

@njit
def check_coherence(ypred:float32[:, :])->float32:
   total = (len(ypred) * 3)
   score = 0
   
   for i in range(len(ypred)):
      y = ypred[i]
      o,h,l,c = y
      rules = (
         # (h > l),
         np.max(y) == h,
         np.min(y) == l,
         np.all(y > 0),
      )
      
      score += len(rules)
      
      for r in rules:
         if not r:
            score -= 1
            
   return (score / total)

class ItsAllShitty(Exception): pass