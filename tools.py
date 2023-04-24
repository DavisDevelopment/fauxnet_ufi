from collections import MutableMapping, MutableSequence
from itertools import chain
import operator as opfn
import asyncio
import asyncio as aio
from functools import partial, wraps
from typing import Any, Awaitable, Callable, TypeVar, cast

from torch.types import Number

T = TypeVar("T", bound=Callable[..., Any])

def sync_to_async(func: T):
   @wraps(func)
   async def run_in_executor(*args, **kwargs):
      loop = asyncio.get_event_loop()
      pfunc = partial(func, *args, **kwargs)
      return await loop.run_in_executor(None, pfunc)

   return cast(Awaitable[T], run_in_executor)

fut = asyncio.futures
Future = fut.Future
FT = TypeVar('FT')

async def farts(r: Awaitable[FT]):
   await aio.sleep(0.01, aio.get_event_loop())
   return await r

def succeed(v: FT):
   l = aio.get_event_loop()
   f = Future(loop=l)
   l.call_soon(f.set_result, v)
   return f
accept = succeed

def reject(e: Exception):
   l = aio.get_event_loop()
   # aio.sleep(0.00001, l)
   f = Future(loop=l)
   l.call_soon(f.set_exception, e)
   # raise e
   return f
fail = reject

def mpmap(f, it, partialargs=None, parallelize=True):
   if partialargs is not None:
      f = partial(f, *partialargs)
   if parallelize:
      from multiprocessing import Pool
      with Pool(11) as pool:
         r = pool.map(f, it)
         pool.close()
         return r
   else:
      return list(map(f, it))
   
def once(func):
   s = [False, None]
   def _wf(*a,**kw):
      if not s[0]:
         s[1] = func(*a, **kw)
         s[0] = True
      return s[1]
   return _wf

def isiterable(x):
   try:
      i = iter(x)
      return True
   except TypeError as e:
      return False
   return isinstance(x, Iterable)

def hasmethod(o, name):
   return hasattr(o, name) and callable(getattr(o, name))

def ismapping(o):
   return hasmethod(o, 'items') and hasmethod(o, 'keys') and hasmethod(o, 'values')

def flat(lol, lvl=0, depth=None):
   fl = []
   for items in lol:
      if isinstance(items, list):
         if depth is None:
            items = flat(items)
         else:
            if lvl < depth:
               items = flat(items, lvl=lvl+1, depth=depth)
            else:
               items = [items]
            
      else:
         items = [items]
      
      fl.extend(items)
   return fl

from collections.abc import MutableMapping

def _flatten_dict_gen(d, parent_key, sep):
   for k, v in d.items():
      new_key = parent_key + sep + k if parent_key else k
      if isinstance(v, MutableMapping):
         yield from flatten_dict(v, new_key, sep=sep).items()
      else:
         yield new_key, v

def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str = '.', factory=dict):
   return factory(_flatten_dict_gen(d, parent_key, sep))

def maxby(key, seq):
   return sorted(seq, key=key, reverse=True)[0]

def minby(key, seq):
   return sorted(seq, key=key)[0]

def have_internet(host="8.8.8.8", port=53, timeout=3):
   """
   Host: 8.8.8.8 (google-public-dns-a.google.com)
   OpenPort: 53/tcp
   Service: domain (DNS/TCP)
   """
   import socket

   try:
      socket.setdefaulttimeout(timeout)
      socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
      return True
   except socket.error as ex:
      return False

def waitForNetworkConnection():
   while not have_internet():
      sleep(5.0)
   return True

from itertools import *
from cytoolz import *

from typing import *
from time import time_ns as nanosecs
from time import time as secs
import inspect
import pandas as pd
import numpy as np
import sys, os

class Struct:
   def __init__(self, *args, **entries):
      if len(entries) == 0 and len(args) == 1 and isiterable(args[0]):
         print(args)
         # entries = dict(args[0])
      self.__dict__.update(entries)
      
   def __getitem__(self, name): 
      return self.__dict__[name]
   
   def __setitem__(self, name, value):
      setattr(self, name, value)
      
   def __delitem__(self, name):
      del self.__dict__[name]
   
   def keys(self):
      return self.__dict__.keys()

   def __repr__(self):
      return repr(self.__dict__)
   
   def copy(self):
      return Struct(**self.__dict__)
   
   def asdict(self):
      return dict(**self.__dict__)

def diffdf(hp, df:pd.DataFrame):
   columns = hp.feature_columns
   for c in columns:
      col = df[c]
      
      df[c] = col.diff()
   
   return df

def npify_entries(e):
   e = getattr(e, '_pentry')
   id = e['id']
   
   times = np.asarray(e['calltimes'])
   
   return dict(
      id=id, 
      meandur=times.mean(),
      totaldur=e['totaltime'],
      callcount=e['count']
   )
   
app_start_time = secs()

def print_profiling():
   import pandas as pd
   #import modin.pandas as pd
   app_run_time = secs() - app_start_time
   npentries = list(map(npify_entries, prof_entries))
   pr = pd.DataFrame.from_records(npentries)
   pr.meandur = pr.meandur * 1e3
   pr.to_csv('profiling.csv')
   pr.meandur = pr.meandur.apply(lambda x: "{:,.2f}ms".format(x))
   
   def fmtdur(n: float):
      pctn = (n / app_run_time)*100
      return f"{n:,.2f} seconds ({pctn:,.2f}%)"
   pr.totaldur = (pr.totaldur).apply(fmtdur)
   pr.sort_values('meandur', ascending=False)
   
   pr.to_csv('profiling.csv')
   
noprofile = True#'--noprofile' in sys.argv

def fnpath(f):
   m = f.__module__
   qn = f.__qualname__
   
   return f'{m}.{qn}'

from types import FunctionType

def profile(*args, fid=None):
   def profile_inner(func: FunctionType):
      if noprofile: 
         return func
      fname = fid if fid is not None else fnpath(func)
      entry = Struct(id=fname, calltimes=[], count=0, totaltime=0)
      setattr(func, '_pentry', entry)
      prof_entries.append(func)
      
      @wraps(func)
      def wrapper(*args, **kwargs):
         st = secs()
         retval = func(*args, **kwargs)
         dur = secs() - st
         # prof_entries[fname] += dur
         entry.calltimes.append(dur)
         entry.count += 1
         entry.totaltime += dur
         return retval
      
      return wrapper

   if len(args) == 1:
      return profile_inner(args[0])
   
   return profile_inner

def closure(func):
   inner = func()
   assert callable(inner), "@closure argument must return a callable"
   return inner

def reduction(func):
   def wrapper(*args, **kwargs):
      pfunc = partial(func, **kwargs)
      if len(args) < 2:
         print(args)
         raise ValueError("insufficient no. of arguments")
      return reduce(pfunc, args)
   return wrapper

def nn(x): return x is not None

@reduction
def nor(a, b):
   return a if a is not None else b

def ternary(cond:bool, tval, fval):
   return tval if cond else fval

def fnor(a, b, reject=None):
   if reject is None:
      reject = lambda x: False
   
   def wrapped(*args, **kwargs):
      try:
         ret = a(*args, **kwargs)
         assert not reject(ret)
      except:
         ret = b(*args, **kwargs)
         assert not reject(ret)
      return ret
   return wrapped
            
from operator import itemgetter, attrgetter, getitem
_getitem = getitem
_undef = object()
def getitem(seq, i, default=_undef):
   if default is _undef:
      return _getitem(seq, i)
   
   try:
      return _getitem(seq, i)
   except KeyError: pass
   except IndexError: pass
   return default

def split_dot_key(path:str):
   alts = path.split('|')
   path = tuple(s.split('.') for s in alts)
   return path

def getattroritem(obj, key, default=None):
   if isinstance(obj, (dict, MutableMapping)):
      return obj.get(key, default)
   else:
      return getattr(obj, key, default)
   # else:

def dotset(d, path, value):
   out = path.split('.', 1)
   key = out[0]
   if len(out) > 1:
      path = out[1]
      try:
         o = d[key]
      except KeyError as e:
         try:
            o = getattr(d, key)
         except AttributeError:
            raise e
      return dotset(o, path, value)
   
   else:
      try:
         d[key] = value
      except KeyError:
         setattr(d, key, value)
      

def dotget(obj, path: str, default=None):
   """
   Forgiving get dot prop.
   If some level doesn't exist, it returns the default.
   """
   value = obj
   for key in path.split('.'):
      if isinstance(value, list):
         index = int(key)
         if index < len(value):
               value = value[index]
         else:
               return default
            
      elif isinstance(value, dict):
         if key in value:
               value = value[key]
         else:
               return default
            
      else:
         if hasattr(value, key):
               value = getattr(value, key)
         else:
               return default
   
   return value

def dotgets(o, *paths, **kwargs):
   _gets = [
      *[(p, None) for p in paths],
      *[(kw_k, kw_v) for kw_k, kw_v in kwargs.items()]
   ]
   
   return tuple(
      dotget(o, path=k, default=v) for k, v in _gets
   )

def getcols(pattern:str, df:pd.DataFrame, tonp=True):
   import fnmatch
   import re
   names = df.columns.tolist() if hasattr(df, 'columns') else df.index
   mcols = fnmatch.filter(names, pattern)
   res = [df[c] for c in mcols]
   if tonp:
      res = np.asanyarray([s.values for s in res])
   return res

def getcolsmatching(pattern:str, df:pd.DataFrame):
   import fnmatch
   # import re
   names = df.columns.tolist() if hasattr(df, 'columns') else df.index
   mcols = fnmatch.filter(names, pattern)
   return mcols

def diffdicts(a, b, equality=opfn.eq, journal_style=False):
   isupdated = complement(equality)
   # a, b = set(a.items()), set(b.items())
   c = merge(a, b)
   ka, kb = set(a.keys()), set(b.keys())
   unchanged = (ka-kb) |(kb-ka)
   shared_keys = (ka & kb)
   updated_keys = set(k for k in shared_keys if isupdated(a[k], b[k]))

   if not journal_style:
      # rest = {k:c[k] for k in unchanged}
      # updated = {k:b[k] for k in updated_keys}
      deleted = set(ka - kb) |(updated_keys)

      added = set(updated_keys) |(kb-ka)
      return deleted, added
   else:
      dropped = ((k, a[k]) for k in (ka - kb))
      changes = ((k, (a[k], b[k])) for k in updated_keys)
      changes = chain(changes, ((k, (None, b[k])) for k in (kb-ka)))
      return (
         frozenset(dropped),
         frozenset(changes)
      )


from tools import closure

@closure
def freeze():
   from frozendict import frozendict

   def isscalar(x):
      try:
         hash(x)
         return True
      except Exception:
         return False
   dsptbl = {
       (dict, MutableMapping): lambda o: frozendict(**{k: freeze(v) for k, v in o.items()}),
       (list, MutableSequence): lambda a: tuple(freeze(el) for el in a),
       set: lambda a: frozenset(a)
   }

   def _f(x):
      if isscalar(x):
         return x

      for k, tfn in dsptbl.items():
         if isinstance(x, k):
            return tfn(x)

      raise TypeError(f'Unhandled: {type(x)} {x}')
   return _f

from itertools import tee, starmap
def unzip(seq):
   """
   -
    Inverse of ``zip``
 
    >>> a, b = unzip([('a', 1), ('b', 2)])
    >>> list(a)
    ['a', 'b']
    >>> list(b)
    [1, 2]
 
    Unlike the naive implementation ``def unzip(seq): zip(*seq)`` this
    implementation can handle an infinite sequence ``seq``.
 
    Caveats:
 
    * The implementation uses ``tee``, and so can use a significant amount
    of auxiliary storage if the resulting iterators are consumed at
    different times.
 
    * The inner sequence cannot be infinite. In Python 3 ``zip(*seq)`` can be
    used if ``seq`` is a finite sequence of infinite sequences.

   """

   seq = iter(seq)

   # Check how many iterators we need
   try:
      first = tuple(next(seq))
   except StopIteration:
      return tuple()

   # and create them
   niters = len(first)
   seqs = tee(cons(first, seq), niters)

   return tuple(starmap(pluck, enumerate(seqs)))

def gets(d, *items, asitems=False):
   if len(items) == 0:
      items = list(d.keys())
   if asitems:
      return tuple((k, d.get(k, None)) for k in items)
   else:
      return tuple(d.get(k, None) for k in items)

def getsmatching(d, pattern='*', regexp=False):
   import fnmatch, regex
   
   key_filter:Callable[[str], bool]
   if not regexp:
      pattern = fnmatch.translate(pattern)
      
   # else:
   key_filter = lambda s: (regex.search(pattern, s) is not None)
   
   if hasmethod(d, 'keys') and hasmethod(d, '__getitem__'):
      return keyfilter(key_filter, d)
   
   else:
      d = d.__dict__.copy()
      
      return getsmatching(d, pattern=pattern, regexp=regexp)

def sub(d, keys):
   return keyfilter(lambda x: x in keys, d)

def after(s, pref):
   if (hasmethod(s, 'index') and hasmethod(s, '__getitem__')):
      try:
         s_i = s.index(pref)

         if isinstance(s, (str, bytes)):
            s_i += len(pref)
         else:
            s_i += 1
         return s[s_i:]

      except IndexError as e:
         print(e)
         return s
   
   return s

def before(s, suff):
   if (hasmethod(s, 'index') and hasmethod(s, '__getitem__')):
      try:
         s_i = s.index(suff)
      except IndexError as e:
         print(e)
         return s
      
      return s[:s_i]

class _capctx:
   def __init__(self, owner):
      self.owner = owner
      self.samples = None
      self.prev_samples = None
      
   def __enter__(self):
      print('capctx:enter')
      self.prev_samples = self.owner._samples
      self.samples = self.owner._samples = []
      
   def __exit__(self, exc_type, exc_value, trace):
      print('capctx:exit')
      self.owner._samples = self.prev_samples
      
   def get(self):
      return self.samples

import inspect
class capturable:
   def __init__(self, func):
      # if inspect.ismethod(func):
      #    print(inspect.getsource(func))
      #    func = staticmethod(func)
      # elif inspect.isfunction(func):
      #    print(inspect.getsource(func))
      @wraps(func)
      def _wrapped(self, *args, **kwargs):
         return func(self, *args, **kwargs)
      
      self.func = _wrapped
      
      self.samples = None
      
   def __call__(self, *args, **kwargs):
      if self.samples is not None:
         print((args, kwargs))
      
      call_return = self.func(*args, **kwargs)
      
      if self.samples is not None:
         call_signature = (args, kwargs)
         print(call_signature)
         self.samples.append((call_signature, call_return))
      
      return call_return
   
   def capture(self):
      print('capture:start')
      return _capctx(self)
   
class capturablemethod:
   def __init__(self, func):
      self.func = func
   
   def __get__(self, instance, cls):
      return capturable(partial(self.func, instance))
   
_identity = lambda x: x
def filtermap(seq, filterfunc=None, mapfunc=None):
   if mapfunc is None: mapfunc = _identity
   if filterfunc is None and mapfunc is None:
      r = seq
   elif filterfunc is None:
      r = map(mapfunc, seq)
   else:
      r = filter(filterfunc, map(mapfunc, seq))
   # if factory is None: factory = list
   return list(r)

PI256 = float('3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679821480865132823066470938446095505822317253594081284811174502841027019385211055596446229489549303819644288109756659334461284756482337867831652712019091456485')

from functools import reduce

def logopby(f:Callable[[Any], Any], it, op:Callable[[Any, Any], bool]=None):
   assert op is not None
   return reduce(lambda a, b: a if op(a[0], b[0]) else b, zip(map(f, it), it))[1]

from operator import gt, lt

def minby(it, key:Callable[[Any], Any]=None):
   return logopby(key, it, lt)

def maxby(it, key:Callable[[Any], Any]=None):
   return logopby(key, it, gt)

def argminby(it, key):
   from fn import F
   from cytoolz import nth
   
   return minby(enumerate(it), lambda x: key(x[0]))[0]

def argmaxby(it, key):
   from fn import F
   from cytoolz import nth
   
   return maxby(enumerate(it), lambda x: key(x[0]))[0]

def safe_div(a:Number, b:Number)->Number:
   """
   wraps division operation to avoid DivisionByZeroException

   Returns: 0 if either operand is 0, a/b otherwise
   """
   if a == 0 or b == 0: 
      return 0
   else:
      return a / b