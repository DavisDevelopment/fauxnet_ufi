from functools import wraps
from termcolor import colored
import torch
from torch import nn, zeros
from torch.autograd import Variable
from nn.common import VecMap

from nn.nalu import NeuralArithmeticLogicUnit, NeuralArithmeticLogicUnitCell

# from .utils import ConvBlock, ConvBlock2D
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn import *

from typing import *
from collections import namedtuple
from nn.ts.classification.utils import ConvBlock
from tools import Struct, flat, dotget, getsmatching, unzip, dotgets, gets

from cytoolz import dicttoolz as dt
from cytoolz import compose, compose_left, merge

import pandas as pd
from pandas import DataFrame, Series
import numpy as np

class RNNEncoder(nn.Module):
   def __init__(self, rnn_num_layers=1, input_feature_len=1, sequence_len=168, hidden_size=100, bidirectional=False, device='cpu', rnn_dropout=0.2):
      super().__init__()
      self.sequence_len = sequence_len
      self.hidden_size = hidden_size
      self.input_feature_len = input_feature_len
      self.num_layers = rnn_num_layers
      self.rnn_directions = 2 if bidirectional else 1
      self.gru = nn.GRU(
         num_layers=rnn_num_layers,
         input_size=input_feature_len,
         hidden_size=hidden_size,
         batch_first=True,
         bidirectional=bidirectional,
         dropout=rnn_dropout
      )
      self.device = device

   def forward(self, input_seq:Tensor) -> Tuple[Tensor, Tensor]:
      ht = torch.zeros(self.num_layers * self.rnn_directions, input_seq.size(0), self.hidden_size, device=self.device)
      
      if input_seq.ndim < 3:
         input_seq.unsqueeze_(2)
      
      gru_out, hidden = self.gru(input_seq, ht)
      
      print('gru_out=', gru_out.shape)
      print('hidden=', hidden.shape)
      
      if self.rnn_directions * self.num_layers > 1:
         num_layers = self.rnn_directions * self.num_layers
         
         if self.rnn_directions > 1:
               gru_out:Tensor = gru_out.view(input_seq.size(0), self.sequence_len, self.rnn_directions, self.hidden_size)
               gru_out = torch.sum(gru_out, axis=2)
               
         hidden = hidden.view(self.num_layers, self.rnn_directions, input_seq.size(0), self.hidden_size)
         if self.num_layers > 0:
               hidden = hidden[-1]
         else:
               hidden = hidden.squeeze(0)
         hidden = hidden.sum(axis=0)
         
      else:
         hidden.squeeze_(0)
      
      return gru_out, hidden
     
class RNNDecoderCell(nn.Module):
   def __init__(self, input_feature_len, hidden_size, dropout=0.2):
      super().__init__()
      
      self.decoder_rnn_cell = nn.GRUCell(
         input_size=input_feature_len,
         hidden_size=hidden_size,
      )
      
      self.out = nn.Linear(hidden_size, 1)
      
      self.attention = False
      
      self.dropout = nn.Dropout(dropout)

   def forward(self, prev_hidden, y) -> Tuple[Tensor, Tensor]:
      rnn_hidden = self.decoder_rnn_cell(y, prev_hidden)
      
      output = self.out(rnn_hidden)
      
      return output, self.dropout(rnn_hidden)
     
class RecurrentSeq2SeqTransformer(nn.Module):
   def __init__(self, encoder, decoder_cell, output_size=3, teacher_forcing=0.3, sequence_len=336, decoder_input=True, device='cpu'):
      super().__init__()
      
      self.encoder = encoder
      self.decoder_cell = decoder_cell
      
      self.output_size = output_size
      self.teacher_forcing = teacher_forcing
      self.sequence_length = sequence_len
      self.decoder_input = decoder_input
      self.device = device

   def forward(self, xb:Tensor, yb=None):
      print(xb.shape)
      
      if self.decoder_input:
         decoder_input = xb[-1]
         print(decoder_input)
         
         input_seq = xb[0]
         if len(xb) > 2:
            encoder_output, encoder_hidden = self.encoder(input_seq, *xb[1:-1])
         else:
            encoder_output, encoder_hidden = self.encoder(input_seq)
      else:
         if type(xb) is list and len(xb) > 1:
            input_seq = xb[0]
            encoder_output, encoder_hidden = self.encoder(*xb)
         else:
            input_seq = xb
            encoder_output, encoder_hidden = self.encoder(input_seq)

      prev_hidden = encoder_hidden
      
      outputs = torch.zeros(input_seq.size(0), self.output_size, device=self.device)
      y_prev = input_seq[:, -1, 0].unsqueeze(1)
      
      for i in range(self.output_size):
         step_decoder_input = torch.cat((y_prev, decoder_input[:, i]), axis=1)
         if (yb is not None) and (i > 0) and (torch.rand(1) < self.teacher_forcing):
            step_decoder_input = torch.cat((yb[:, i].unsqueeze(1), decoder_input[:, i]), axis=1)
         
         rnn_output, prev_hidden = self.decoder_cell(prev_hidden, step_decoder_input)
         y_prev = rnn_output
         outputs[:, i] = rnn_output.squeeze(1)
      
      return outputs

class TransformerCell(nn.Module):
   def __init__(self, enc, dec, output_size=1) -> None:
      super().__init__()
      
      self.encoder = enc
      self.decoder = dec
      self.output_size = output_size
      
   def forward(self, X:Tensor, y:Optional[Tensor]=None):
      batch_dim, in_seq_len, in_features = X.shape
      
      print(tuple(X.shape), tuple(y.shape) if y is not None else None)
      
      input_seq = X
      xb = X.view(in_seq_len, )
      decoder_input = xb[-1]
      print(decoder_input)
      
      input_seq = xb[0]
      if len(xb) > 2:
         encoder_output, encoder_hidden = self.encoder(input_seq, *xb[1:-1])
      else:
         encoder_output, encoder_hidden = self.encoder(input_seq)
      encoder_output, encoder_hidden = self.encoder(input_seq)
      
      #...
      prev_hidden = encoder_hidden
      
      outputs = torch.zeros(input_seq.size(0), self.output_size)
      y_prev = input_seq[:, -1, 0].unsqueeze(1)
      print('y_prev=', y_prev)
      
      for i in range(self.output_size):
         step_decoder_input = torch.cat((y_prev, decoder_input[:, i]), axis=1)
         if (y is not None) and (i > 0) and (torch.rand(1) < self.teacher_forcing):
            step_decoder_input = torch.cat((y[:, i].unsqueeze(1), decoder_input[:, i]), axis=1)
         
         rnn_output, prev_hidden = self.decoder(prev_hidden, step_decoder_input)
         y_prev = rnn_output
         outputs[:, i] = rnn_output.squeeze(1)
      
      return outputs
   
from main import load_frame, ExperimentConfig
from nn.data.sampler import DataFrameSampler
from cachetools import cached, Cache
from cachetools.keys import hashkey, typedkey
from fn import F, _

# @cached(Cache(2000), key=hashkey)
def samples_for(symbol:str, seq_len:int, columns:List[str]):
   df:DataFrame = load_frame(symbol, './sp100')
   df['abs_volume'] = df['volume']
   df['volume'] = df['volume'].pct_change()
   df = df.iloc[1:]
   df = df[columns]
   
   print(df)
   sampler = DataFrameSampler(df)
   sampler.preprocessing_funcs.clear()
   sampler.configure(in_seq_len=seq_len)
   
   buffer = []
   
   for ts, X, _y in sampler.samples():
      X = X.squeeze().numpy()
      buffer.append((X, X))
      
   x, y = tuple(map(list, unzip(buffer)))
   x = torch.from_numpy(np.asanyarray(x, dtype=np.float32)).swapaxes(1, 2)
   y = torch.from_numpy(np.asanyarray(y, dtype=np.float32)).swapaxes(1, 2)
   
   return x, y

class AEBase(nn.Module):
   def __init__(self, n_features:int, seq_len:int, hidden_size=800, n_recurrent_layers=1, **kwargs):
      super().__init__()
      
      self.n_recurrent_layers = n_recurrent_layers
      self.seq_len = seq_len
      self.n_features = n_features
      self.hidden_size = hidden_size
      self.bidirectional = True
      self.dropout = 0.0
      
      self.n_directions = 2 if self.bidirectional else 1
      self.bias = False
      self.activation = nn.ReLU()

class Encoder(AEBase):
   @wraps(AEBase.__init__)
   def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      # self._init_roll_state = h0
      
      rnn_kwargs = getsmatching(kwargs, '(recurrent|rnn|lstm|gru)_(.+)$', regexp=True) if ('rnn_opts' not in kwargs) else kwargs.pop('rnn_opts')
      rnn_type = kwargs.pop('rnn_type', 'gru')
      if isinstance(rnn_type, str) and rnn_type.upper() in ('GRU', 'LSTM'):
         rnn_type = getattr(nn, rnn_type.upper())
      elif callable(rnn_type):
         pass
      
      assert callable(rnn_type), 'Invalid rnn_type argument'
      
      default_rnn_kwargs = dict(
         input_size=self.n_features,
         hidden_size=self.n_features,
         num_layers=self.n_recurrent_layers, 
         dropout=self.dropout,
         bidirectional=self.bidirectional,
         batch_first=True,
         bias=False
      )
      
      self.roll = rnn_type(**merge(default_rnn_kwargs, rnn_kwargs))
      
   def forward(self, x:Tensor):
      print(x.shape)
      batch_size = x.shape[0]
      seq_len, n_directions, hidden_size = self.seq_len, self.n_directions, self.n_features
      h0 = Variable(torch.zeros((self.n_recurrent_layers * self.n_directions, batch_size, self.n_features), requires_grad=True))
      
      if self.bidirectional:
         for i in range(h0.size(0)):
            # print(h0[i])
            if i % 2 == 0:
               h0[i, :, :] = x[:, 0, :]
            else:
               h0[i, :, :] = x[:, -1, :]
         
      h0[:, :, :] = x[:, 0, :]
      
      
      print(h0)
      rolled, h1 = self.roll(x, h0)
      print('rolled shape = ', tuple(rolled.shape))
      print('roll state shape = ', tuple(h1.shape))
      
      print(tuple(rolled.shape), (batch_size, seq_len, n_directions, hidden_size))
      print(rolled.view(batch_size, seq_len, n_directions, hidden_size).shape)
      #(D ∗ num_layers, N, H[out])
      
      return rolled, h1
   
class Decoder(AEBase):
   def __init__(self, n_features:int, seq_len:int, hidden_size=800, n_recurrent_layers=1, **kwargs):
      super().__init__(n_features, seq_len, hidden_size=hidden_size, n_recurrent_layers=n_recurrent_layers, **kwargs)
      
      rnn_kwargs = getsmatching(kwargs, '(recurrent|rnn|lstm|gru)_(.+)$', regexp=True) if ('rnn_opts' not in kwargs) else kwargs.pop('rnn_opts')
      
      # n_directions = (2 if self.bidirectional else 1)
      rnn_kwargs = merge(dict(
         input_size=n_features * self.n_directions,
         hidden_size=n_features, 
         # proj_size=seq_len,
         num_layers=self.n_recurrent_layers, 
         batch_first=True,
         bidirectional=self.bidirectional,
         dropout=self.dropout,
         bias=False
      ), rnn_kwargs)
      
      self.unroll = nn.GRU(**rnn_kwargs)
      self.activ = nn.ReLU()
      
      # self.diffuse = ConvBlock(seq_len * self.n_directions, n_features, 8, 2)
      
   def forward(self, X:Tensor, encoded:Tensor, hdn_state:Tensor):
      # print('hdn_state=', tuple(hdn_state.shape))
      # print('encoded=', tuple(encoded.shape))
              
      unrolled_raw, hdn_new = self.unroll(encoded, hdn_state)
      unrolled_raw = self.activ(unrolled_raw)
      # print('unrolled shape = ', tuple(unrolled_raw.shape))
      # unrolled_raw = unrolled_raw.swapaxes(1, 2)
      # output = self.diffuse(unrolled_raw)
      # output = output.swapaxes(1, 2)
      output = unrolled_raw
         
      return output
      # if self.bidirectional:
      #    ur = unrolled_raw
      #    mid = (ur.size(-1) // 2)
      #    fwd = ur[:, :, :mid].swapaxes(1, 2)
      #    bwd = ur[:, :, -mid:].swapaxes(1, 2)
      #    print(fwd.shape, bwd.shape)
         
      #    fwd_y = self.diffuse(fwd).swapaxes(1, 2)
      #    # bwd_y = self.diffuse(bwd).swapaxes(1, 2)
         
      #    print(fwd_y[-1, -2:, -1])
      #    return fwd_y
      
      # else:
         # print(unrolled_raw.shape, hdn_new.shape)
      
   
class AutoEncoder(nn.Module):
   def __init__(self, head, tail):
      super().__init__()
      
      self.encoder = head
      self.decoder = tail
      
   
   def forward(self, X:Tensor):
      cout, hdn_new = self.encoder(X)
      out = self.decoder(X, cout, hdn_new)
      return out
      
def train_for(cfg:ExperimentConfig, mdl_cfg:Struct):
   symbol, seq_len, n_features = cfg.symbol, cfg.in_seq_len, cfg.num_input_channels
   
   x, y = samples_for(symbol, seq_len, ['open', 'high', 'low', 'close', 'volume'])
   x = x[:, :, :-1]
   y = y[:, :, :-1]
   n_features = x.size(-1)
   
   head = Encoder(n_features, seq_len)
   tail = Decoder(n_features, seq_len)
   model = AutoEncoder(head, tail)
   
   optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
   criterion = nn.MSELoss()
   
   for n in range(250):
      ypred:Tensor = model(x)
      
      loss = criterion(ypred, y)
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      print(f'epoch #{n}: {loss.detach().item()}')
      print('y=', y[-1, [0, -1], :].numpy())
      print('ypred=', ypred[-1, [0, -1], :].detach().numpy())
   
if __name__ == '__main__':
   train_for(
      ExperimentConfig(
         in_seq_len = 90,
         out_seq_len= 90,
         epochs=200,
         symbol='AMZN',
         
      ),
      Struct(betty='urinal')
   )