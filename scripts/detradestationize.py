import sys, os, json, pickle
P = os.path

sys.path.append(P.join(P.dirname(__file__), '..'))
from glob import glob
from typing import *

# from ds.model.ctrl import DataFrame
import pandas as pd
import numpy as np

from pathlib import Path
from pandas import DataFrame
from itertools import *
from functools import *
from operator import eq
from cytoolz import *
from fn import F, _

from datatools import ohlc_resample, unpack

def load_csv(folder=None, symbol=None, path=None):
   if path is not None:
      InPath = path
   elif symbol is not None:
      # folder = folder if folder is not None else os.getcwd()
      InPath = P.join('.', f'{symbol.upper()}.csv')
   
   if P.exists(InPath):
      df:pd.DataFrame = pd.read_csv(InPath)
      return df
   else:
      raise FileNotFoundError(InPath)
   
def chunks(s, *sizes):
   r = []
   for size in sizes:
      v = s[:size]
      s = s[size:]
      
      r.append(v)
   return r

def fmttsdate(d: int):
   y,m,d = chunks(str(d), 4, 2, 2)
   return '%s-%s-%s' % (y,m,d)

def fmttstime(t: str):
   return t

from tqdm import tqdm
from shutil import rmtree, move, copytree

def poofar(Fi:str, Fo:str):
   SymbolsTxt = P.join(Fi, 'Symbols.txt')
   
   if P.exists(SymbolsTxt):
      L = open(SymbolsTxt, 'r').readlines()
      L = filter(lambda s: len(s)>0, map(lambda x: x.strip(), L))
      Symbols = set(L)
   else:
      Symbols = set(map(lambda s: s.replace('.csv', ''), glob(P.join(Fi, '*.csv', Fi))))
   
   print(locals())
   assert Symbols is not None and len(Symbols) > 0, Symbols
      
   if False:
      pass
   else:
      E = P.join(Fi, 'export')
      
      if P.exists(E):
         rmtree(E)
      
      os.makedirs(E, exist_ok=True)
      
      manifest = dict()
      
      for s in tqdm(Symbols):
         if s[0] == '$':
            continue
         
         else:
            try:
               df = load_csv(symbol=s)
            except FileNotFoundError as e:
               print(e)
               print('Skipping ', s)
               continue
            
            D = df.Date.apply(fmttsdate).astype('string')
            T = df.Time.apply(fmttstime).astype('string')
            DT = (D + ' ' + T)
            df['DateTime'] = pd.to_datetime(DT)
            df.rename(columns=lambda s: s.lower(), inplace=True)
            df.rename(columns=dict(ttlvol='volume'), inplace=True)
            
            # df.set_index('datetime', inplace=True)
            df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
            dt = df.datetime
            manEntry = dict(
               symbol=s,
               date_range=(dt.min(), dt.max()),
            )
            manifest[s] = manEntry
            
            df.to_csv(P.join(E, f'{s}.csv'))
            df.to_feather(P.join(E, f'{s}.feather'))
      
      import pickle
      pickle.dump(manifest, open(P.join(E, f'manifest.pickle'), 'wb+'))
      
      if P.exists(Fo):
         rmtree(Fo, ignore_errors=True)
         
      else:
         move(E, Fo)
         print('export completed successfully; converted data was placed in {}'.format(Fo))
   
def format_ts_data(symbol:str, df:pd.DataFrame):
   D = df.Date.apply(fmttsdate).astype('string')
   T = df.Time.apply(fmttstime).astype('string')
   DT = (D + ' ' + T)
   df['DateTime'] = pd.to_datetime(DT)
   df = df.rename(columns=lambda s: s.lower())
   df = df.rename(columns=dict(ttlvol='volume'))
   
   df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
   
   return df
   
def survey(d:str):
   assert P.isdir(d)
   p = Path(d)

   rmtree(p.with_name(p.name + '_backup'), ignore_errors=True)
   copytree(str(p), p.with_name(p.name + '_backup'))
   
   symbolsListFile = (p / 'Symbols.txt')
   symbolsAsListed = None
   if symbolsListFile.exists():
      symbolsAsListed = set(filter(lambda s: len(s)>0, symbolsListFile.read_text().splitlines()))
   
   found = set([(Path(csv).name[:-4], csv) for csv in p.rglob('*.csv')])
   
   loaded = dict()
   for (sym, fp) in found:
      sfp, fp = fp, Path(fp)
      df = pd.read_csv(sfp)
      # df = format_ts_data(sym, df)
      # fp.rename(fp.parent / (fp.name.replace('.csv', '_tradestation.csv')))
      # df.to_csv(fp)
      loaded[sym] = df
   
   return loaded

def convert(d_in:str, d_out:str=''):
   assert P.isdir(d_in)
   p = Path(d_in)
   
   # if d_out is None or len(d_out) == 0:
   d_out_tmp = str(p / '.pending_conversion')

   rmtree(p.with_name(p.name + '_backup'), ignore_errors=True)
   copytree(str(p), p.with_name(p.name + '_backup'))
   
   rmtree(d_out_tmp, ignore_errors=True)
   os.makedirs(d_out_tmp, exist_ok=True)
   
   symbolsListFile = (p / 'Symbols.txt')
   symbolsAsListed = None
   if symbolsListFile.exists():
      symbolsAsListed = set(filter(lambda s: len(s)>0, symbolsListFile.read_text().splitlines()))
   
   found = set([(Path(csv).name[:-4], csv) for csv in p.rglob('*.csv')])
   
   converted = set()
   
   for (sym, fp) in tqdm(found):
      sfp, fp = fp, Path(fp)
      df = pd.read_csv(sfp)
      df = format_ts_data(sym, df)
      df.to_csv(P.join(d_out_tmp, f'{sym}.csv'))
      df.to_feather(P.join(d_out_tmp, f'{sym}.feather'))
      
      converted.add(sym)
      
   move(d_out_tmp, d_out)
   
   return (found, converted)

def ensureallequal(a:Iterable[Any]):
   a = list(a)
   
   l = a[0]
   for i in range(len(a)-1):
      if l != a[1+i]:
         return False
      l = a[1+i]

   return True

def pack(d:Union[Dict[str, DataFrame], str]=None, format='pickle', target='./something.pack', resample=None, homogenize=False):
   assert d is not None
   if isinstance(d, dict):
      indexes = []
      column_specs = []
      symbols = []
      frames = {}
      
      if homogenize:
         raise ValueError('waow, wtf')
      
      for (sym, df) in d.items():
         symbols.append(sym)
         df = format_ts_data(sym, df)
         indexes.append(df.index)
         if resample is not None:
            # df = df.set_index(df.datetime)
            df = ohlc_resample(resample, df)
         column_specs.append(df.dtypes.to_dict())
         # buffers.append(df.to_numpy())
         frames[sym] = df
         
      pickle.dump((symbols, indexes, frames), open(target, 'wb+'))
   
   else:
      pack(d=survey(d), format=format, target=target, resample=resample, homogenize=homogenize)
      
def unpack(packed_path:str, format='pickle'):
   index, columns, symbols, data = pickle.load(open(packed_path, 'rb'))
   print(index)
   print(columns)
   print(symbols)
   
   result = {}
   
   for i in range(data.shape[0]):
      sym = symbols[i]
      sym_data = data[i]
      df = DataFrame(data=sym_data, index=index, dtype=columns)
      result[sym] = df
      
   return result
   
if __name__ == '__main__':
   output_dir = None
   input_dir = None
   
   args = sys.argv[1:]
   
   config={}
   arguments=[]
   flags = set()
   
   for arg in args:
      isFlagOrKw = arg.startswith('-')
      while True:
         if arg.startswith('-'): 
            arg = arg[1:]
         else:
            break
      
      if isFlagOrKw and '=' in arg:
         k, v = arg[:arg.index('=')], arg[arg.index('=')+1:]
         config[k] = v
      elif isFlagOrKw:
         flags.add(arg)
      else:
         arguments.append(arg)
            
   if 'output_dir' in config or 'o' in config:
      output_dir = P.expanduser(P.expandvars(
         config.get('output_dir', config.get('o', None))
      ))
   
   input_dir = arguments[0] if len(arguments) >= 1 else os.getcwd()
   
   print('input_dir=', input_dir)
   print('output_dir=', output_dir)
   
   assert input_dir is not None
   # assert output_dir is not None
   
   # pack(input_dir, target='./sp100_daily.pickle', resample='D')
   convert(input_dir, output_dir)
   
   print('-' * 100)
   print('TradeStation-dump cleaned, exported and saved to ("csv", "feather") formats')