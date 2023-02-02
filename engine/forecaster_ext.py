
from ds.forecaster import *
from engine.mixin import MultiAdvisorPolicy

AP = MultiAdvisorPolicy
Consensus = AP.FollowConsensus
Merge = AP.MergeThenFollow

#TODO add support for Dict[str, np.ndarray] argument Signature
#TODO finish implementation and integrate into the trading engine
class EnsembleForecaster(ForecasterUnitBase):
   units: List[NNForecaster] = None
   _merge_method: str = 'max'
   # _merge_policy = Merge

   def __init__(self, units:List[NNForecaster], merge_method='max'):
      self.units = list(units)
      self.merge_method = merge_method

   def merge_outputs(self, outs: List[np.ndarray]) -> np.ndarray:
      raise NotImplementedError('EnsembleForecaster._merge_outputs')

   def call(self, inputs: np.ndarray) -> np.ndarray:
      outputs = [unit(inputs) for unit in self.units]

      return outputs
