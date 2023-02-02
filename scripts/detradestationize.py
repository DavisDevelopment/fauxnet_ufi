import sys, os, json
from glob import glob
P = os.path
import pandas as pd
import numpy as np

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
from shutil import rmtree, move

def export(Fi:str, Fo:str):
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
   
if __name__ == '__main__':
   # CWDCSVs = glob(P.join(os.getcwd(), '*.csv'))
   # SymbolsTxt = P.join(os.getcwd(), 'Symbols.txt')
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
   assert input_dir is not None and output_dir is not None
   export(input_dir, output_dir)
   
   print('-' * 100)
   print('TradeStation-dump cleaned, exported and saved to ("csv", "feather") formats')