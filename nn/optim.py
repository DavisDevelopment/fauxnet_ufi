'''A wrapper class for scheduled optimizer '''
import numpy as np
import torch

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, lr_init=0.0001, n_warmup_steps=5):
        self._optimizer = optimizer
        self.lr_init = lr_init
        self.lr = lr_init
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step(self):
         self.step_and_update_lr()

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
      #   d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        lr_init = self.lr_init
        
        if n_steps <= n_warmup_steps:
           return 1.0
        
        else:
           coeff = (n_steps - n_warmup_steps) * 0.01
           return self.lr
        return (1 / ((n_steps - n_warmup_steps) / n_warmup_steps))
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    @property
    def lr(self):
      return self._lr
   
    @lr.setter
    def lr(self, lr:float):
      self._lr = lr
      for param_group in self._optimizer.param_groups:
         param_group['lr'] = lr

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        if self.n_steps > self.n_warmup_steps and (self.n_steps - self.n_warmup_steps % 5 == 0):
           lr = self.lr / 2
           self.lr = lr

        
            
import sys, os, shutil
P = os.path
from typing import *
            
class Checkpoints():
   def __init__(self, model, optimizer, chkpt_dir='./.model_checkpoints/'):
       self.model = model
       self.optimizer = optimizer
       self.dir = chkpt_dir
       self.fn_template = P.join(self.dir, 'checkpoint_e%d.pth')
       self.n_warmup_steps = 10
       self._best_key = 'accuracy'
       
       self.n_steps = 0
       self._best_score = None
       self._best_step = None
       
   def zero_grad(self):
      self.optimizer.zero_grad()
      
   def reinit_dir(self):
      shutil.rmtree(self.dir, ignore_errors=True)
      os.makedirs(self.dir, exist_ok=True)
      
   def deinit_dir(self):
      shutil.rmtree(self.dir, ignore_errors=True)
      
   def save_checkpoint(self):
      if self.n_steps == self.n_warmup_steps + 1:
         self.reinit_dir()
      
      chkpt_file_path = (self.fn_template % (self.n_steps + 1))
      torch.save(self.model.state_dict(), chkpt_file_path)
      
   def restore_best_checkpoint(self):
      if self._best_step is not None:
         best_state = torch.load(self.fn_template % (self._best_step + 1))
         self.model.load_state_dict(best_state)
         return self.model
      
      raise ValueError('No steps recorded to restore')
   
   def step(self, metrics:Dict[str, Any]={}):
      assert self._best_key in metrics, f'no "{self._best_key}" metric listed'
      score = metrics[self._best_key]
      if self.n_steps > self.n_warmup_steps:
         if self._best_score is None or score > self._best_score:
            self._best_score = score
            self._best_step = self.n_steps
         
         self.save_checkpoint()
      
      self.optimizer.step()
      self.n_steps += 1
      
   def close(self):
      apex = self.restore_best_checkpoint()
      self.deinit_dir()
      
      self.optimizer = None
      self.model = None
      self.n_steps = 0
      self._best_score = None
      self._best_step = None
      
      return apex