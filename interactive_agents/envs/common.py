import numpy as np
from pettingzoo import ParallelEnv

class SyncEnv(ParallelEnv):
    """
    A convenience class that makes it easier to implement the
    PettingZoo ParallelEnv interface.  Assumes that the set of
    agents is always the same for each episode and time step.
    """

    rng = None

    @property
    def agents(self):
        return list(self.observation_spaces.keys())

    @property
    def possible_agents(self):  # NOTE: Will the ParallelEnv class do this for us?
        return list(self.observation_spaces.keys())  
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def seed(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()
