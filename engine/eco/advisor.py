from ds.forecasters import NNForecaster

from numpy import ndarray
from pandas import DataFrame, Series
import numpy as np
import pandas as pd
#import modin.pandas as pd

from engine.mixin import EngineImplBase as Engine
from engine.data.argman import ArgumentManager
from typing import *
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler

from engine.eco.forecaster import TradeEngineForecaster

class TradeEngineAdvisor:
   forecaster:Optional[TradeEngineForecaster] = None
   
   def __init__(self, forecaster:Optional[TradeEngineForecaster]=None):
      self.forecaster = forecaster
      
   def call(self, index=0):
      pass