"""
   olbi - "overload builtins"
   
   overwrites certain builtin and library methods with an extended or alternate implementation. globally
"""

import builtins as bi
import inspect
import os

from termcolor import colored

if not hasattr(bi, 'old_print'):
   old_print = bi.print
   setattr(bi, 'old_print', old_print)

noprint = False
notrace = False

def wrapped_print(*args, **kwargs):
   if noprint:
      return None
   
   elif notrace:
      bi.old_print(*args, **kwargs)
      
   else:
      if any((type(v).__name__ == 'DataFrame') for v in args):
         from tools import flat
         args = tuple(flat([([v] if type(v).__name__ != 'DataFrame' else ['\n', v]) for v in args], depth=1))
      
      curframe = inspect.currentframe()

      calframe = inspect.getouterframes(curframe, 2)[1]
      
      loclabel = colored(
         f'{os.path.basename(calframe.filename)}:{calframe.lineno}',
         'blue',
         attrs=['underline']
      )
   
      bi.old_print(*(loclabel, *args), **kwargs)

if bi.print is not wrapped_print:
   setattr(bi, 'print', wrapped_print)


g = globals()
def configurePrinting(silent=False, tracing=True):
   g['noprint'] = silent
   g['notrace'] = not tracing
   
   
class SilenceContext:
   def __init__(self, silent=False):
      self.prev = None
      self.status = silent
      
   def __enter__(self):
      self.prev = g['noprint']
      g['noprint'] = self.status
      
   def __exit__(self, exc_type, exc_value, trace):
      g['noprint'] = self.prev
      
def printing(status=True):
   if isinstance(status, str):
      status = status.lower().strip()
      if status in ('on', 'yes'):
         status = True
      elif status in ('off', 'no'):
         status = False
      else:
         raise ValueError(f'Unknown status {status}')
   
   return SilenceContext(silent=not status)