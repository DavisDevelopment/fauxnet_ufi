import torch
import numpy as np

from torch import tensor, Tensor, randn, empty, zeros
from torch import nn
from torch.nn import *
from torch.nn.functional import relu
from torch import optim
from torch.jit import script, ignore, optimize_for_inference, freeze, export

from torch.autograd import Variable
import torch.nn.functional as F
from typing import *

from nn.core import *

from itertools import chain
from cytoolz import *

class SeasonalTrendAnalysis(Module):
   def __init__(self) -> None:
      super().__init__()