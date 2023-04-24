from .pipeline import *

class FauxTradingPipelineBackend(PipelineBackendBase):
   __id__ = 'faux:trade'
   def __init__(self, config=None):
      """
      - What the pipeline will need:
       - a training dataset (or a description of what to do to obtain one)
         - only applicable when one or more of the signal generators require 'fit' to be called before they can be applied to a datasource
       
       - Strategy-level
         - one or more signal-generators (callables that return numeric signal labels of the scheme denoted by config.strategy.signal_generator)
         - one or more signal-interpreters (callables that perform generator-specific signal handling, operating the BacktestEngine)
         - record-keeping scaffolding, to facilitate detailed inspection of performance later on
         
       - Backtest
         - datasource[s] to be used for evaluation
      """
      super().__init__(config)
      
      #TODO
      
      self.load_config(self.config)
      
   def parse_config(self, raw):
      from expcfg import parse_config
      from pprint import pprint
      
      o = parse_config(raw)
      pprint(o)
   
   def load_config(self, config):
      stratConf = config['strategy']
      
      sig_gens = stratConf['signal_generator']
      for sg_id, sg_opts in sig_gens:
         sg_type = resolve_siggen(sg_id)