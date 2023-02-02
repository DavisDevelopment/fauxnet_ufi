
from typing import *
import sys, os, pickle, json
from functools import *
from cytoolz import *
import atexit

def flush_atexit(lf):
   if not lf._is_closed:
      lf.flush()
      lf.close()

class LogFile:
   def __init__(self):
      self.buffer = []
      
      self.highWaterMark = 1000
      self._is_closed = False
      self._atexit_cb = atexit.register(partial(flush_atexit, self))
      self._on_encoding_error = repr
      
   def clear(self):
      pass
      
   def close(self):
      self._is_closed = True
      atexit.unregister(self._atexit_cb)
      
   def flush(self):
      pass
      
   def put(self, msg:Any, label=None):
      self.buffer.append((label, msg))
      if len(self.buffer) >= self.highWaterMark:
         self.flush()
      
      
   def _write_(self, data:List[Any]):
      pass
   
class PickleLogFile(LogFile):
   def __init__(self, filename):
      super().__init__()
      self.filename = filename
      
   def clear(self):
      import os
      if os.path.exists(self.filename):
         os.remove(self.filename)
      self.buffer = []
      
   def _write_(self, data:List[Any]):
      head = []
      if os.path.exists(self.filename):
         with open(self.filename, 'rb+') as f:
            head = pickle.load(f)
               
      with open(self.filename, 'wb+') as f:
         f.seek(0)
         content = head + data
         pickle.dump(content, f)
   
   def flush(self):
      b = self.buffer[:]
      self._write_(b)
      
      self.buffer = []
      
   def get_all(self):
      with open(self.filename, 'rb') as f:
         return pickle.load(f)

from mongoengine import *
from mongoengine.queryset.visitor import Q

import uuid

# disconnect()
connect('fauxnet')

class LogFileEntry(Document):
   label = StringField(default='')
   data = BinaryField()
   
class MongodbLogFile(LogFile):
   def __init__(self):
      super().__init__()
      
   def put(self, msg, label=None):
      msg = pickle.dumps(msg)
      entry = LogFileEntry(label=label, data=msg)
      
      self.buffer.append(entry)
      
      if len(self.buffer) >= self.highWaterMark:
         self.flush()
         
   def _write_(self, b:List[LogFileEntry]):
      for row in b:
         # row.delete
         row.save()
   
   def clear(self):
      for o in LogFileEntry.objects():
         o.delete()
      