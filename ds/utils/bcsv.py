
import io, re, mmap
import pandas as pd
#import modin.pandas as pd
import numpy as np

SEP = '{END OF DOCUMENT}'

def fault_tolerant_load_csv(b):
   #! SLOW
   #TODO
   pass

class BCsvBase:
    #? teehee
   def __init__(self, f, engine='feather'):
      self.buf = f
      self.engine = engine
      self.isbin = isbinary(self.engine)

      self._sep = SEP

   def _putter(self, b, x):
      if isinstance(x, str) and self.isbin:
         x = bytes(x, 'utf-8')
      b.write(x)

   def mkbuffer(self):
      b = (io.BytesIO if isbinary(self.engine) else io.StringIO)()
      return b
#TODO 

class Entry:
   def __init__(self, name, document):
      self.name = name
      self.document = document

class Loader(BCsvBase):
   def __init__(self, f, engine='feather'):
      super().__init__(f, engine)
      f = self.buf
      self.buf = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
      self.offset = 0
      self.loaded = {}
      # self.engine = engine
      # self.isbin = 
      
      # self._sep = SEP
      self._decode_params = {}
      self._eof = False
      
   def _read_body(self, body:bytes):
      r = (io.BytesIO if self.isbin else io.StringIO)(body)
      method = getattr(pd, f'read_{self.engine}')
      return method(r, **self._decode_params)
      
   def _load_document(self):
      b:mmap.mmap = self.buf
      idxOf = lambda s, offset: b.find(bytes(s, 'utf-8'), offset)
      # s = b'tits'
      # s.split
      seplen = len(bytes(self._sep, 'utf-8'))
      idx = idxOf(self._sep, self.offset)

      if idx != -1:
         text = bytes(b[self.offset:idx])
         self.offset = idx + seplen
      else:
         # b.seek(self.offset)
         text = bytes(b[self.offset:])
      if len(text) == 0:
         self._eof = True
         return None
      # _, title, body = text.partition(b'\n')
      # text = text
      namesize = text[0]
      name = text[1:1+namesize]
      name = name.decode('utf-8')
      body = text[namesize+1:]
      # print(name, len(body))
      
      r = (io.BytesIO if self.isbin else io.StringIO)(body)
      method = getattr(pd, f'read_{self.engine}')
      doc = method(r, **self._decode_params)
      
      return Entry(name, doc)
   
   def load(self):
      while not self._eof:
         r = self._load_document()
         if r is None:
            break
         
         self.loaded[r.name] = r.document
      return self.loaded
         
formats = dict(
   csv='txt',
   pickle='bin',
   feather='bin'
)

from functools import partial

def isbinary(format):
   m = dict(txt=False, bin=True)
   return m[formats[format]]
         
class Dumper(BCsvBase):
   #? teehee
   def __init__(self, f, engine='feather'):
      super().__init__(f, engine)
      # self.buf = f
      # self.engine = engine
      # self.isbin = isbinary(self.engine)
      # print(self.isbin)
      
      self.written = []
      # self._sep = SEP
      self._encode_params = {}
      # self.gbuffer = self.mkbuffer()
      
   # def _putter(self, b, x):
   #    if isinstance(x, str) and self.isbin:
   #       x = bytes(x, 'utf-8')
   #    b.write(x)
      
   def mkbuffer(self):
      # b = (io.BytesIO if isbinary(self.engine) else io.StringIO)()
      b = super().mkbuffer()
      return b, partial(self._putter, b)
   
   def _writedoc(self, doc:pd.DataFrame, b=None):
      if b is None:
         b = self.mkbuffer()
      encode = getattr(doc, f'to_{self.engine}')
      encode(b, **self._encode_params)
   
   def dumpdoc(self, name:str, doc:pd.DataFrame):
      tmp, put = self.mkbuffer()
      head = io.BytesIO()
      tmp.write(bytes([len(name)]))
      put(name)
      self._writedoc(doc, tmp)
      if not self.isbin:
         put('\n' + self._sep)
      else:
         put(self._sep)
      
      self.written.append(tmp)
      
      return tmp.getbuffer().nbytes
      
   def dump(self, bcsv):
      import time, timeit
      
      start = time.time()
      totalbytes = 0
      lens = []
      
      for name, doc in bcsv.items():
         doc:pd.DataFrame = doc
         encoded_size = self.dumpdoc(name, doc)
         lens.append(encoded_size)
         totalbytes += encoded_size
      
      end = time.time()
      print(f'encoding took {end-start}secs')

      f = self.buf
      f.truncate(totalbytes)
      f.flush()
      b = mmap.mmap(f.fileno(), totalbytes, access=mmap.ACCESS_WRITE)
      # f.write(fi)
      i = 0
      for size, wbuf in zip(lens, self.written):
         # print(size)
         # wbuf:io.BytesIO = wbuf
         # b[i, size] = wbuf.getvalue()
         # i += size
         wbuf.seek(0)
         spos = b.tell()
         b.write(wbuf.getvalue())
         epos = b.tell()
         assert (epos - spos) == size, 'size mismatch; u stoopid'
         
      b.flush()
      b.close()

def load(f):
   if isinstance(f, str):
      f = open(f, 'r')
   
   # io = mmap.mmap(f.fileno(), 0)
   loader = Loader(f)
   loader.load()
   
   from ds.utils.book import Book
   ld = loader.loaded
   # return Book(ld)
   return ld

def dump(f, documents):
   if isinstance(f, str):
      f = open(f, 'w+')
   with f:
      writer = Dumper(f)
      writer.dump(documents)
      print('dumped')

   