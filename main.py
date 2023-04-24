
import sys, os, argparse
import json, pickle
P = os.path
from functools import *
from typing import *

import pandas as pd
from tools import flat

global_parser = argparse.ArgumentParser('faux')
subparsers = global_parser.add_subparsers(required=True)

subcommands = {}
def subcmd(name, **kwargs):
   def wfn(func):
      @wraps(func)
      def wcmd(args):
         return func(args)
      
      cmd_parser = subparsers.add_parser(name, **kwargs)
      cmd_parser.set_defaults(func=wfn)
      
      setattr(wcmd, 'argparser', cmd_parser)
      setattr(wcmd, 'func', func)
      
      subcommands[name] = (cmd_parser, wcmd)
      
      return wcmd
   
   return wfn

def parse_var(s):
    """
    Parse a key, value pair, separated by '='
    That's the reverse of ShellArgs.

    On the command line (argparse) a declaration will typically look like:
        foo=hello
    or
        foo="hello world"
    """
    items = s.split('=')
    key = items[0].strip() # we remove blanks around keys, as is logical
    value = ''
    if len(items) > 1:
        # rejoin the rest:
        value = '='.join(items[1:])
    return (key, value)

def parse_vars(items):
   """
   Parse a series of key-value pairs and return a dictionary
   """
   d = {}
   if items:
      for item in items:
         key, value = parse_var(item)
         d[key] = value
   return d

cfgp:argparse.ArgumentParser = argparse.ArgumentParser('configure')
cfgp.add_argument('--kind', metavar='config_kind', required=False, default='global', choices=['global', 'hyperparameters', 'strategy', 'model', 'dataset'])
cfgp.add_argument('set', metavar='KEY=VALUE', nargs='+', help='assign configuration variables')

@subcmd('configure')
def cli_configure(args):
   print(args)
   kind = args.kind
   assignments = parse_vars([x for x in args.set if '=' in x])
   #TODO modify the folder's configuration file
   file_path = P.join(P.dirname(__file__), f'{kind}_config.json')
   if P.exists(file_path):
      config = json.load(open(file_path, 'r'))
   else:
      #TODO preload the configuration dict with default values
      config = {}
   config.update(assignments)
   json.dump(config, open(file_path, 'w+'))

def parse_date_range_component(val):
   return None

def parse_date_range(*vals)->Tuple[pd.Timestamp, pd.Timestamp]:
   vals = [parse_date_range_component(val) for val in vals]
   start, end = vals
   
   #...
   return start, end

@subcmd('backtest')
def backtest_command(args):
   print(args)
   date_range = parse_date_range(*args.date_range) if args.date_range is not None else (None, None)
   symbols = flat([sym.split(',') for sym in args.symbol])
   print(date_range, symbols)

def main():
   sysargs = sys.argv[1:]

   
   
   btp = backtest_command.argparser
   btp.add_argument('--symbol', nargs='*')
   btp.add_argument('--date_range', metavar='<start> <end>', nargs=2, help='the time range over which to backtest', default=['?', '?'])
   # btp.set_defaults(func=backtest_command)
   
   scp = subparsers.add_subparsers('sculpt', help='identify optimal strategy/model configuration for a specific symbol')
   scp.add_subparsers('')
   
   args = global_parser.parse_args(sysargs)
   try:
      args.func(args)
   except Exception as e:
      raise e
      
if __name__ == '__main__': main()