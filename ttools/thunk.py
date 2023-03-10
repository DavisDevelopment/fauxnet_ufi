from functools import partial
from typing import *

from cytoolz.functoolz import *
import operator
from operator import *

T = TypeVar('T')

ThunkInit0 = Union[Callable[[], T], T]
ThunkInit = Union[ThunkInit0, Callable[[], ThunkInit0]]

def thunk_getter(init:ThunkInit)->Callable[[], T]:
   if callable(init):
      return init
   else:
      return lambda : init
   
class Thunk(Generic[T]):
   kind:int
   get_value:Optional[Callable[[], T]] = None
   value:Optional[T] = None
   
   def __init__(self, init, once=False, kind=None):
      self.once = once
      
      if kind is None:
         if isinstance(init, Thunk):
            self.kind = init.kind
            self.get_value = init.get_value
            self.value = init.value
            self.once = init.once
         
         elif callable(init):
            self.get_value = init
            self.kind = 1
            
         else:
            self.kind = 0
            self.value = init
      
      elif kind in (0, 1):
         if kind == 0:
            self.kind = 0
            self.value = init
         else:
            self.kind = 1
            assert callable(init), '`init` argument must be callable when kind is forced to 1'
            self.get_value = init
         
   def get(self)->T:
      if self.kind == 0:
         return self.value #type: ignore
      elif self.kind == 1:
         return self.get_value() #type: ignore
      else:
         raise Exception('unreachable')
   
   def __repr__(self):
      return repr(self.get())
   
   def __call__(self, *args, **kwargs):
      return self.get()(*args, **kwargs)
   
   def __getattr__(self, name:str):
      return Thunk(lambda : getattr(self.get(), name))
   
   def __getitem__(self, index:Union[Thunk[Any], Any])->Thunk[Any]:
         
binary_operators = [
   'add', 'sub', 'mul', 'div',
   'eq', 'ne', 'lt', 'gt'
]

def wrappedBinOp(opfn):
   def op_method(self, other):
      if not isinstance(other, Thunk):
         other = Thunk(other)
      r = Thunk(lambda: opfn(self.get(), other.get()))
      return r
   return op_method

for op in binary_operators:
   dundername = f'__{op}__'
   # operato
   method = getattr(operator, ({
      'div': 'truediv'
   }).get(op, op))
   
   setattr(Thunk, dundername, wrappedBinOp(method))
   
         
# def thunk(init):
#    raise Exception('Stub')
def thunkv(init: ThunkInit):
   if isinstance(init, Thunk):
      return init.get()
   
   get = thunk_getter(init)
   return get()

def thunk(v, *context_vars, **kwargs):
   if len(context_vars) > 0:
      if not callable(v):
         raise TypeError('thunk(), when called with more than one argument, must provide a callable as the first argument')
      return Thunk(partial(v, *context_vars))