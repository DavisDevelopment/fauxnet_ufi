
import torch
import numpy as np

from torch import tensor, Tensor, randn, empty, zeros, sigmoid
from torch import nn
from torch.nn import *
from torch.nn.functional import relu
from torch import optim
from torch.jit import script, ignore, optimize_for_inference, freeze, export

from torch.autograd import Variable
import torch.nn.functional as F
from typing import *
from nn.arch.transformer.Modules import ScaledDotProductAttention

from nn.core import *

from itertools import chain
from cytoolz import *

from nn.common import *
from nn.common import autofn
from nn.namlp import NacMlp
from nn.nac import NeuralAccumulatorCell as FLayer

from functools import reduce, partial

# from nn.arch.lstm_vae import LSTMVAE, LSTMAE
from nn.arch.vae import VAE, Encoder, Decoder
from nn.arch.lstnet import LSTNet
from nn.arch.transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward
from nn.arch.transformer.Modules import ScaledDotProductAttention
from nn.arch.transformer.TimeSeriesTransformer import TimeSeriesTransformer
from nn.arch.lstm_vae import *

class PseudoClassifier(Module):
   def __init__(self, n_steps_long:int, n_steps_short:int, n_features:int):
      super().__init__()
      
      self.n_steps_long = n_steps_long
      self.n_steps_short = n_steps_short
      self.n_features = n_features
      
      self.tst = TimeSeriesTransformer(
         input_size=n_features,
         dec_seq_len=30,
         batch_first=True,
         out_seq_len=7
      )
      
   def forward(self, X:Tensor):
      return self.tst(X)