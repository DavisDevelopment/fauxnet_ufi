import numpy as np
from cytoolz import *
from pathlib import Path
from torch.nn import *
# from nalu.layers import NaluLayer
# from nalu.core import NaluCell, NacCell
from torch.jit import script, freeze, optimize_for_inference
from torch.nn import Module
from torch.autograd import Variable
from torch import Tensor, tensor, asarray
import torch.nn as nn
import torch
from typing import *
import os
import sys
import math
import random
import re
from tqdm import tqdm
import pandas as pd
import ta
from sklearn.preprocessing import MinMaxScaler, minmax_scale

from ds.data import load_dataset, all_available_symbols
from tools import unzip

from ptmdl.models.nac import NacCell
from ptmdl.ops import *

class BinarizedArithmeticModule(Module):
   def __init__(self, input_spec:Tuple[int], output_spec:Tuple[int]):
      super().__init__()
      
      self.input_spec = input_spec
      self.output_spec = output_spec
      
      self._nac_input_spec = binarized_spec(input_spec, end_dim=0)
      self._nac_output_spec = binarized_spec(output_spec, end_dim=0)
      
      print(self._nac_input_spec, self._nac_output_spec)
      
      self.na_cell = NacCell(self._nac_input_spec, self._nac_output_spec)
      
   def forward(self, inputs:Tensor):
      flat_inputs = inputs.flatten()
      
      binary_inputs = tensor(binarize_float(flat_inputs.numpy()))
      binary_outputs = self.na_cell(binary_inputs)
      
      outputs = unbinarize(binary_outputs.numpy())
      
      return outputs