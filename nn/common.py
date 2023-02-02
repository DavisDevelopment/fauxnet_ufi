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
      
      self.term_encoder = Linear(n_input_terms, 1)
      self.seq_encoder = Linear(n_steps, n_output_terms)
      
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
      
      self.step_decoder = Linear(n_input_terms, n_output_steps)
      self.term_decoder = Linear(1+n_input_terms+n_output_steps, n_output_terms)
      self.term_revisor = Linear(1+n_output_terms, n_output_terms)
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
   
class EconomyForecastingModule(EconomyModule):
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
         
   def encode_market(self, symbol:int, market:Tensor):
      return (self.market_encoder if self.shared_transform else self.market_encoder[symbol])(market)
   
   def decode_market(self, symbol:int, terms:Tensor):
      return (self.market_decoder if self.shared_transform else self.market_decoder[symbol])(terms)

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
      
def torch_vectorize(f, inplace=False, end_dim=None):
   #TODO: implement this such that it can be jitted by pytorch
   def wrapper(tensor:Tensor, **params) -> Tensor:
      x = tensor if inplace else tensor.clone()
      xshape = x.shape
      
      view = x.flatten() if end_dim is None else x.flatten(end_dim=end_dim)

      y = torch.reshape(f(view, **params), xshape)

      return y

   return wrapper

def clone_module(module, memo=None):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)
    **Description**
    Creates a copy of a module, whose parameters/buffers/submodules
    are created using PyTorch's torch.clone().
    This implies that the computational graph is kept, and you can compute
    the derivatives of the new modules' parameters w.r.t the original
    parameters.
    **Arguments**
    * **module** (Module) - Module to be cloned.
    **Return**
    * (Module) - The cloned module.
    **Example**
    ~~~python
    net = nn.Sequential(Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))
    clone = clone_module(net)
    error = loss(clone(X), y)
    error.backward()  # Gradients are back-propagate all the way to net.
    ~~~
    """
    # NOTE: This function might break in future versions of PyTorch.

    # TODO: This function might require that module.forward()
    #       was called in order to work properly, if forward() instanciates
    #       new variables.
    # TODO: We can probably get away with a shallowcopy.
    #       However, since shallow copy does not recurse, we need to write a
    #       recursive version of shallow copy.
    # NOTE: This can probably be implemented more cleanly with
    #       clone = recursive_shallow_copy(model)
    #       clone._apply(lambda t: t.clone())

    if memo is None:
        # Maps original data_ptr to the cloned tensor.
        # Useful when a Module uses parameters from another Module; see:
        # https://github.com/learnables/learn2learn/issues/174
        memo = {}

    # First, create a copy of the module.
    # Adapted from:
    # https://github.com/pytorch/pytorch/blob/65bad41cbec096aa767b3752843eddebf845726f/torch/nn/modules/module.py#L1171
    if not isinstance(module, torch.nn.Module):
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()

    # Second, re-write all parameters
    if hasattr(clone, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                param = module._parameters[param_key]
                param_ptr = param.data_ptr
                if param_ptr in memo:
                    clone._parameters[param_key] = memo[param_ptr]
                else:
                    cloned = param.clone()
                    clone._parameters[param_key] = cloned
                    memo[param_ptr] = cloned

    # Third, handle the buffers if necessary
    if hasattr(clone, '_buffers'):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None and \
                    clone._buffers[buffer_key].requires_grad:
                buff = module._buffers[buffer_key]
                buff_ptr = buff.data_ptr
                if buff_ptr in memo:
                    clone._buffers[buffer_key] = memo[buff_ptr]
                else:
                    cloned = buff.clone()
                    clone._buffers[buffer_key] = cloned
                    memo[param_ptr] = cloned

    # Then, recurse for each submodule
    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_module(
                module._modules[module_key],
                memo=memo,
            )

    # Finally, rebuild the flattened parameters for RNNs
    # See this issue for more details:
    # https://github.com/learnables/learn2learn/issues/139
    if hasattr(clone, 'flatten_parameters'):
        clone = clone._apply(lambda x: x)
    return clone
