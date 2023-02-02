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

class EconomyModule(Module):
   def __init__(self, n_symbols:int) -> None:
      super().__init__()
      
      self.n_symbols = n_symbols
   
class MarketModule(Module):
   def __init__(self) -> None:
      super().__init__()
      
class MarketImageEncoder(MarketModule):
   def __init__(self, input_shape=None, n_output_terms:int=-1):
      super().__init__()
      assert input_shape is not None
      n_steps, n_input_terms = input_shape
      assert n_output_terms > 0
      assert n_steps > 0
      assert n_input_terms > 0
      
      self.n_steps = n_steps
      self.n_input_terms = n_input_terms
      self.n_output_terms = n_output_terms
      
      self.term_encoder = autofn(n_input_terms, 1)
      self.seq_encoder = autofn(n_steps, n_output_terms)
      
   def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
      # return super().parameters(recurse)
      return chain(self.term_encoder.parameters(), self.seq_encoder.parameters())
      
   def forward(self, market_image:Tensor):
      seq = Variable(zeros((self.n_steps,)))
      for i in range(self.n_steps):
         terms = market_image[i, :]
         seq[i] = self.term_encoder(terms)
         #  = self.term_encoder
      output = self.seq_encoder(seq)
      return output      
      
   def extra_repr(self) -> str:
      return f'n_terms={self.n_input_terms}, n_steps={self.n_steps}'
   
class VecMap(Module):
   def __init__(self, f:Optional[Module]=None, axis=0, output_shape=None) -> None:
      super().__init__()
      self.f = f
      self.axis = axis
      self.output_shape = output_shape
      
   def forward(self, X:Tensor):
      view = X.flatten(end_dim=self.axis)
      out = None if self.output_shape is None else Variable(zeros(self.output_shape))
      for i in range(len(view)):
         r = self.f(view[i])
         assert isinstance(r, Tensor)
         
         if out is None:
            out = Variable(zeros((len(view), *r.shape), dtype=r.dtype))
         
         out[i] = r
         
      return out
            
class MarketImageDecoder(Module):
   def __init__(self, n_input_terms:int, output_shape=(0, 0)) -> None:
      super().__init__()
      assert 0 not in (n_input_terms, *output_shape)
      self.n_input_terms = n_input_terms
      n_output_steps, n_output_terms = output_shape
      self.n_output_steps = n_output_steps
      self.n_output_terms = n_output_terms
      
      self.step_decoder = autofn(n_input_terms, n_output_steps)
      self.term_decoder = autofn(1+n_input_terms+n_output_steps, n_output_terms)
      self.term_revisor = autofn(1+n_output_terms, n_output_terms)
      self.output_shape = output_shape
      
   def forward(self, X:Tensor):
      step_q = self.step_decoder(X)
      output_image = Variable(zeros(self.output_shape))
      
      for t in range(self.n_output_steps):
         arg = torch.cat((tensor([t]), step_q, X))
         terms = self.term_decoder(arg)
         output_image[t, :] = terms
         
      for i in range(self.n_output_steps):
         arg = torch.cat((tensor([i]), output_image[i, :]))
         output_image[i, :] = self.term_revisor(arg)
         
      return output_image
   
   def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
      # return super().parameters(recurse)
      return chain(self.term_decoder.parameters(), self.step_decoder.parameters(), self.term_revisor.parameters())
   
class EconomyForecastingModule(EconomyModule):
   market_encoder:Union[Callable[[Tensor], Tensor], List[Callable[[Tensor], Tensor]]]
   market_decoder:Union[Callable[[Tensor], Tensor], List[Callable[[Tensor], Tensor]]]
   
   forecast_shape:Tuple[int]
   n_symbols:int
   shared_transform:bool
   
   def __init__(self, n_symbols:int=0, forecast_shape=None, market_transform=(None, None), shared_transform=True):
      super().__init__(n_symbols)
      assert n_symbols is not None and n_symbols > 0
      assert None not in market_transform
      assert forecast_shape is not None
      
      self.forecast_shape = forecast_shape
      self.n_output_steps, self.n_output_terms = self.forecast_shape
      market_encoder, market_decoder = market_transform
      self.shared_transform = shared_transform
      
      if shared_transform:
         self.market_encoder = market_encoder
         self.market_decoder = market_decoder
      else:
         #! probably gonna be slow AF
         self.market_encoder = [clone_module(market_encoder) for _ in range(n_symbols)]
         self.market_decoder = [clone_module(market_decoder) for _ in range(n_symbols)]
         
   def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
      if self.shared_transform:
         assert isinstance(self.market_encoder, Module)
         assert isinstance(self.market_decoder, Module)
         return chain(self.market_encoder.parameters(recurse=recurse), self.market_decoder.parameters(recurse=recurse))
      else:
         assert isinstance(self.market_encoder, list)
         assert isinstance(self.market_decoder, list)
         return chain(
            *(l.parameters() for l in self.market_encoder),
            *(l.parameters() for l in self.market_decoder)
         )
         
   def encode_market(self, symbol:int, market:Tensor):
      if callable(self.market_encoder):
         return self.market_encoder(market)
      
      else:
         return self.market_encoder[symbol](market)
      # return (self.market_encoder if self.shared_transform else self.market_encoder[symbol])(market)
   
   def decode_market(self, symbol:int, terms:Tensor):
      # return (self.market_decoder if self.shared_transform else self.market_decoder[symbol])(terms)
      if callable(self.market_decoder):
         return self.market_decoder(terms)
      
      else:
         return self.market_decoder[symbol](terms)

   def forward(self, markets:Tensor):
      forecast = Variable(zeros((self.n_symbols, *self.forecast_shape)))
      # print(markets.shape, forecast.shape)
      for symbol in range(self.n_symbols):
         market = markets[symbol]
         forecasting_terms = self.encode_market(symbol, market)
         y = self.decode_market(symbol, forecasting_terms).squeeze()
         # print(y)
         forecast[symbol] = y
      return forecast

from nn.namlp import NacMlp

def autofn(nin:int, nout:int, hidden=None):
   # hidden = [nout, nout] if hidden is None else hidden
   return NacMlp(nin, nout, hidden)