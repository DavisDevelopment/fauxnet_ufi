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

from ptmdl.models.nac import NacCell, NeuralAccumulatorCell
from ptmdl.models.nalu import NeuralArithmeticLogicUnitCell, NeuralArithmeticLogicUnit

class wrap_submodule(nn.Module):
   def __init__(self, method_module:nn.Module):
      super().__init__()
      self.mm = method_module
   
   def forward(self, *args):
      X:Tensor = torch.asarray(
         list(map(lambda x: x.flatten() if isinstance(x, Tensor) else x, args))
      )
      return self.mm(X)
Nac = NeuralAccumulatorCell


from cytoolz import *

def nacmlp(*layers):
   layers = list(layers)
   assert len(layers) >= 2
   if len(layers) == 2:
      nin, nout = layers
      return Nac(nin, nout)
   else:
      nn = Sequential(*(Nac(nin, nout) for (nin, nout) in partition(2, layers)))
      return nn

class MarketEncoder(nn.Module):
   def __init__(self, nsyms:int):
      super().__init__()
      
      self.nsyms = nsyms
   
class MarketSymbolFrameEncoder(MarketEncoder):
   def __init__(self, nsyms:int, n_in:int, n_out:int=3):
      super().__init__(nsyms)
      self.n_in = n_in
      self.n_out = n_out
      
      self.downsample_frame = wrap_submodule(Sequential(
         Nac(1+n_in, n_out),
         Nac(n_out, n_out),
         Nac(n_out, n_out)
      ))
   
   def forward(self, X:Tensor):
      Xlr:Tensor = Variable(torch.zeros((X.shape[0], self.n_out)))
      for i in range(self.n_out):
         Xlr[i] = self.downsample_frame(i/self.n_out, *X[i])
      input()
      return Xlr
   
class NAryMarketFrameSequenceEncoder(nn.Module):
   def __init__(self, nsyms:int, nsteps:int, nchannels:int, nout:int):
      super().__init__()
      
      self.nsyms = nsyms
      self.nsteps = nsteps
      self.nchannels = nchannels
      self.nout = nout
      #wrap_submodule(nacmlp(nchannels + 2, nchannels * 16, nchannels * 16, nout))
      self.narize_frame = Sequential(Nac(nchannels+2, 16), Nac(16, 16), Nac(16, nout))
      
   def forward(self, market:Tensor):
      encoded = Variable(torch.empty((self.nsyms, self.nsteps, self.nout)))
      for symbol in range(self.nsyms):
         market_frames = market[symbol]
         assert len(market_frames) == self.nsteps
         
         for t in range(self.nsteps):
            frame = market_frames[t]
            args = (symbol, t, *frame)
            nary_frame = self.narize_frame(torch.tensor(args))
            encoded[symbol, t] = nary_frame
            
      return encoded

def percent_change(a, b):
  return ((b - a) / abs(a))

class ReallocationMapEncoder(nn.Module):
   def __init__(self, nsyms, nsteps, nchannels):
      super().__init__()
      self.nsyms = nsyms
      self.nsteps = nsteps
      self.nchannels = nchannels
      
      self.gb = Parameter(zeros((nsteps)))
      self.b = Parameter(zeros((nsteps, nsyms)))
      
      self.register_parameter('gb', self.gb)
      self.register_parameter('b', self.b)
      # self.b = Parameter(torch.zeros(()))
      #(3+nchannels, 16, 16, 2)
      self.compute_reallocation = Sequential(
         Nac(3, 16),
         Nac(16, 16),
         Nac(16, 2)
      )
      
   def forward(self, market:Tensor):
      mvs = Variable(torch.empty((self.nsteps, self.nsyms, self.nsyms, 2)))
      for symbol_a in range(self.nsyms):
         
         for symbol_b in range(self.nsyms):
            for t in range(self.nsteps):
               realloc = self.compute_reallocation(tensor([t/self.nsteps, symbol_a/self.nsyms, symbol_b/self.nsyms]))
               
               mvs[t, symbol_a, symbol_b] = realloc
      
      y = self.gb + mvs
      return y
         
class MarketVideoEncoder(nn.Module):
   def __init__(self, nframes:int):
      super().__init__()
      
      self.nframes = nframes
      self.rnn = LSTM(input_size=nframes, hidden_size=nframes, bidirectional=True, batch_first=True, dtype=torch.float32)
      
   def forward(self, X:Tensor):
      y = self.rnn(X.float())
      print(type(y))

class Encoder_v0(nn.Module):
   def __init__(self, nK, nT, nC):
      super().__init__()
      
      self.nK = nK
      self.nT = nT
      self.nC = nC
      nM = self.nM = 128 * 4
      self.input_shape = (nK, nT, nC)
      
      self.ch_zip   =  [
         Nac(nC, nC),
         NeuralAccumulatorCell(nC, 1),
      ]
      
      self.ch_unzip =  [
         NeuralAccumulatorCell(1, nC),
         # PReLU()
      ]
      
      self.t_zip = [
         Nac(nT, nM),
         Nac(nM, nM),
         Nac(nM, nM)
      ]
      
      self.delta_M = [
         Nac(nM, nM),
         Nac(nM, nM),
         Nac(nM, 1)
      ]
     
      self.ch_zip, self.ch_unzip, self.t_zip, self.delta_M = Sequential(*self.ch_zip), Sequential(*self.ch_unzip), Sequential(*self.t_zip), Sequential(*self.delta_M)
     
   def parameters(self):
      params = [
         list(self.ch_zip.parameters()),
         list(self.ch_unzip.parameters()),
         list(self.t_zip.parameters()),
         list(self.delta_M.parameters())
      ]
      
      p = []
      for ps in params:
         p.extend(ps)
         
      return ParameterList(p)
     
   def forward(self, X:Tensor):
      xshape = tuple(X.shape)
      assert (xshape == self.input_shape or xshape[1:] == self.input_shape), f'{xshape} != {self.input_shape}, {xshape[1:]} != {self.input_shape}'
      nK, nT, nC, nM = self.nK, self.nT, self.nC, self.nM
      uni_y = Variable(torch.zeros((nK, nT), requires_grad=True))
      
      for k in range(nK):
         for t in range(nT):
            ch_x = X[k, t, :]
            uni_y[k, t] = self.ch_zip(ch_x).squeeze()
      
      # mem_y = uni_y
      mem_y = Variable(torch.zeros((nK, nM), requires_grad=True))
      for k in range(nK):
         mem_y[k] = self.t_zip(uni_y[k])
      
      y = Variable(torch.zeros((nK, nC), requires_grad=True))

      for k in range(nK):
         delta_univariate = self.delta_M(mem_y[k])
         lastX = X[k][-1]
         lXmean = lastX.mean()
         baseline = (lastX - lXmean)
         delta_multivariate_variance = self.ch_unzip(delta_univariate).squeeze()
         mv_y = (delta_multivariate_variance + baseline) + (baseline * (randn((len(baseline),)) * 0.001))
         
         # y[k] = renormalize(delta_multivariate, (X[k].min(), X[k].max()), (0.0, 1.0))
         y[k] = mv_y
      
      return y

class ProjectingDecoder(nn.Module):
   def __init__(self, input_shape:Iterable[int], n_out:int, flow_shape:Iterable[int]):
      super().__init__()
      self.n_in, self.n_input_channels = input_shape
      self.n_out = n_out
      
      self._fp = FlowProjector(n_out, self.n_input_channels, zeros(tuple(flow_shape)))
      
   def compute_flow_encoding(self, X:Tensor)->Tensor:
      raise NotImplementedError()
   
   def flow_project(self, flow:Tensor):
      return self._fp(flow)
   
   def forward(self, X:Tensor)->Tensor:
      flow = self.compute_flow_encoding(X)
      return self.flow_project(flow)
   # 
class FlowProjector(nn.Module):
   def __init__(self, n_steps:int, n_terms:int, init_state:Tensor):
      super().__init__()
      self.n_steps = n_steps
      self.n_terms = n_terms
      
      self.state = Parameter(init_state)
      self.register_parameter('state', self.state)
      
      self._proj = Sequential(
         Nac(1+len(init_state.flatten()), 128),
         Nac(128, 32),
         Nac(32, self.n_terms)
      )
   
   def reset_state(self, x:Tensor):
      self.state[:] = x
      
   def forward(self, X:Tensor):
      self.reset_state(X)
      y = Variable(zeros((self.n_steps, self.n_terms)))
      for t in range(self.n_steps):
         yp = self._proj(torch.tensor((t/self.n_steps, *self.state)))
         y[t] = yp
      return y