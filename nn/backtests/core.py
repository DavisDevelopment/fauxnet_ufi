from pprint import pprint
import sys, os

from datatools import norm_batches, percent_change, rescale
from nn.ts.classification.fcn_baseline import FCNNaccBaseline, FCNBaseline
P = os.path
import torch
import numpy as np
import pandas as pd
from numpy import ndarray
from nn.data.core import TwoPaneWindow

from torch import Tensor, from_numpy, tensor
from tools import Struct, unzip

from sklearn.preprocessing import MinMaxScaler
from typing import *
from dataclasses import dataclass, asdict, astuple

from nn.backtests.position import Action, Position

class Agent(object):
   pos:Optional[Position]
   
   def __init__(self, dollars=100.0):
      self.dollars = dollars
      self.dollars_init = dollars
      self.holdings = 0.0
      self.logs = []
      self.balances = []
      self.borrow_price = None
      self.pos = None
      self.symbol = None
      self.df = None