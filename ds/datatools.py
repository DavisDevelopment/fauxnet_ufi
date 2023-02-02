from .common import *
import multiprocessing as mp 

def shuffle_in_unison(a, *bs):
    # shuffle two arrays in the same way
    state = np.random.get_state()
    np.random.shuffle(a)
    for b in bs:
      np.random.set_state(state)
      np.random.shuffle(b)


def _make_key(args, kwargs):
   l = [] + list(args)
   for k, v in kwargs.items():
      l += [k, v]
   return tuple(l)

is_subprocess = (mp.parent_process() is not None)

def persistent_memoize(user_func):
   """
   'memoize'-like functionality, but cache is persisted on the filesystem

   Parameters
   ----------
   user_func : callable
       

   Returns
   -------
   callable
   """
   from functools import lru_cache, wraps
   from cytoolz.functoolz import memoize
   
   import shelve
   import json
   import inspect

   if not is_subprocess:
      if not P.exists('__appcache__'):
         os.mkdir('./__appcache__')
      defpath = P.abspath(inspect.getfile(user_func))
      fname = P.abspath('__appcache__/memoized_{}.db'.format(user_func.__name__))
      cache_file:shelve.DbfilenameShelf = shelve.open(fname)
      cftime = cache_file.get('ftime', None)
      if P.exists(fname) and (cftime is not None) and P.getmtime(defpath) > cftime:
         cache_file.clear()
      cache_file['ftime'] = P.getmtime(defpath)
   else:
      from cachetools import LRUCache as Cache
      cache = Cache()

   @wraps(user_func)
   def _wrapped(*args, **kwds):
      key = json.dumps(_make_key(args, kwds))
      print('cache-key=', key)

      if key in list(cache_file.keys()):
         print('OBTAINED FROM CACHE')
         return cache_file[key]

      else:
         result = user_func(*args, **kwds)
         cache_file[key] = result
         return result

   _wrapped = memoize(_wrapped)
   
   return user_func


def single(v:Any):
   yield v
   
import string, random

def random_string_generator(str_size:int, allowed_chars:str=None):
   if allowed_chars is None:
      allowed_chars = string.ascii_lowercase
   return ''.join(random.choice(allowed_chars) for x in range(str_size))
