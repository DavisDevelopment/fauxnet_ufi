import pandas as pd
#import modin.pandas as pd
from .trading import *
from .mixin import EngineImplBase as Impl
from .backend import RemoteBackend
from .portfolio import Portfolio
from .utils import *
from datetime import datetime, timedelta
import schedule as cron
import signal, atexit, sys
from fn import F, _
from cytoolz import *
from tools import *

class LiveTrading(RemoteBackend, Impl):

   def __init__(self):
      # super(LiveTrading, self).__init__()
      TradingEngineBase.__init__(self)
      Impl.__init__(self)
      RemoteBackend.__init__(self)
      
      self.portfolio = Portfolio()
      self.portfolio.attach(self)
      
      self.test_mode = False
      self.running = False
      
   # def invest(self, symbol, volume=None, weight:float=1.0):
      # super().invest(symbol, )
      
   def init(self):
      if self.balance is None:
         self.sync_balance()

      super(LiveTrading, self).init()
      
      #TODO probably a bunch more shit
      self.book.update()
      
      self._initialized = True
      
   def sync_balance(self):
      bal = self.client.get_account_balance()
      bal = bal.to_dict()['vol']
      bal = format_balance(bal)
      bal = valfilter(_ != 0, bal)
      self.balance = bal
      print(self.balance)
      
   def sync(self):
      super().sync()
      
      waitForNetworkConnection()
      
      if not self.book.isuptodate(deep=True):
         self.book.update()
      
      self.sync_balance()
      
   def fcapply(self, fcn=None):
      return super().fcapply(fcn)
   
   def get_args(self, nnf, date=None, asarray=False, tailscale=True):
      ret = super().get_args(nnf, date=date, asarray=asarray, tailscale=tailscale)
      return ret
      
   def step(self):
      waitForNetworkConnection()
            
      now = pd.Timestamp.now()
      self.current_date = now
      
      self.sync()
      
      for k, nnf in self.forecasters.items():
         args = self.get_args(nnf, date=self.current_date)
         if all(map(lambda v: v is None, args.values())):
            raise Exception('yo man what the fuck')
            
      self.trade_on_forecasts()
      print(self.portfolio.positions)
      print(self.order_queue)
      
      self.flush_order_queue()
      self.sync()
      
   def wstep(self):
      try:
         # self.sync()
         self.step()
      
      except Exception as fatal_error:
         #TODO do some stuff...
         raise fatal_error
   
   def run(self):
      interval = tsUnitToSeconds(self.freq)
      fmt_seconds(interval)
      crontab = cron.every(interval).seconds
      
      def _hackstep():
         waitForNetworkConnection()
         
         self.book.update(force=True)
         self.step()
         self.liquidate()
      
      cron.every(12).hours.do(_hackstep)
      
      self.running = True
      
      self.liquidate()
      
      print(self.ignore_balance)
      if self.ignore_balance is not None:
         assert 'LUNA' not in self.ignore_balance
      
      # run startup step
      self.step()
      
      # #? just hang out for a half hour or so
      # sleep(60 * 1)
      
      #* repeat the 'liquidate' step, to undo this test-run
      # self.liquidate()
      
      # exit(0)
      
      safe_run_forever(self)
      
   def shutdown(self, error=None):
      self.emit('shutdown', self)
      
def fmt_seconds(secs:float):
   unit = 'S'
   t = secs
   if t >= 60:
      t, unit = (t/60), 'Min'
   if t >= 60:
      t, unit = (t/60.0), 'H'
   if t >= 24:
      t, unit = (t/24.0), 'D'
   formatted = f'{t}{unit}'
   print(formatted)
   return t, unit, formatted
      
def safe_run_forever(self:LiveTrading):
   def _on_signal_interrupt(sig, frame):
      self.running = False
      self.shutdown()
      sys.exit(0)
   
   signal.signal(signal.SIGINT, _on_signal_interrupt)
   
   def _on_exit():
      exception = sys.last_value if hasattr(sys, 'last_value') else None
      if exception is not None:
         import traceback as tb
         ltb = sys.last_traceback
         with open('crash_report.txt', 'w+') as report_file:
            tb.print_last(file=report_file)
   
   atexit.register(_on_exit)
   
   while self.running:
      cron.run_pending()
      
      sleep(5.0)
   
def safe_run_forever_threaded(self: LiveTrading):
   import time
   import threading
   import queue

   def job():
      print("I'm working")


   def worker_main():
      while 1:
         job_func = jobqueue.get()
         job_func()
         jobqueue.task_done()

   jobqueue = queue.Queue()

   do = lambda tab, job, **kw: tab.do(partial(jobqueue.put, job, **kw))

   worker_thread = threading.Thread(target=worker_main)
   worker_thread.start()

   while 1:
      schedule.run_pending()
      time.sleep(1)