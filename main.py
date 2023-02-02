
from time import sleep
# from econn.model import *
# from nn.components import *
from nn.common import *
from datatools import renormalize
   
from nn.namlp import NacMlp

#* (symbols, timesteps, features)
nK, nT, nC, nM = 10, 90, 5, 128

X = randn((nK, nT, nC))
y = randn((nK, 1)).squeeze()

from datatools import get_cache, calc_Xy, load_ts_dump

kcache = load_ts_dump()
syms, ranges, buffers, move_ranges, ground, X, y = calc_Xy(kcache, 'open,high,low,close'.split(','), nT_out=1)
nK, nT, nC = X.shape
print('nK=', nK, 'nT=', nT, 'nC=', nC)

nn = EconomyForecastingModule(n_symbols=nK, forecast_shape=(1, 4), market_transform=(
   MarketImageEncoder(input_shape=(nT, nC), n_output_terms=32),
   MarketImageDecoder(n_input_terms=32, output_shape=(1, 4))
))

print(X.shape)
print(nn)

X, y = torch.from_numpy(X), torch.from_numpy(y)
X, y = X.float(), y.float()
y = y.squeeze(1)

print(nn(X))
# us_y = renormalize

# encoder = Encoder(nK, nT, nC)
opt = torch.optim.RMSprop(nn.parameters(), lr=0.0002)
crit = torch.nn.MSELoss()

epochs = 30
us_y = torch.zeros_like(y)
for k in range(nK):
   us_y[k] = renormalize(y[k], (0.0, 1.0), move_ranges[k])

for e in range(epochs):
   yhat = nn(X).squeeze()
   loss = crit(y, yhat)
   loss.backward()
   opt.step()
   
   print(f'epoch {e}: {loss}')
   
   if e % 5 == 0:
      moe = (y - yhat) / torch.abs(y) * 100.0
      # print(us_y[-1][3], us_yhat[-1][3])
      mean_moe = moe.mean().detach().item()
      print(f'margin of error={mean_moe:.2f}')
   