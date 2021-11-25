from gym.spaces import Discrete, Box
import numpy as np

class RepeatedGame:

    def __init__(self, config, spec_only=False):
        assert "row_payoffs" in config, "must define row player payoffs"
        assert "column_payoffs" in config, "must define column player payoffs"

        self._row_payoffs = config.get("row_payoffs")
        self._column_payoffs = config.get("column_payoffs")

        assert len(self._row_payoffs.shape) == 2, "row payoffs must be a 2-d tensor"
        assert len(self._column_payoffs.shape) == 2, "column payoffs must be a 2-d tensor"
        assert self._row_payoffs.shape == self._column_payoffs.shape, "row and column payoff matrices must be same size"
        
        self.observation_space = {
            0: Box(0, 1, self._row_payoffs.shape[1]), 
            1: Box(0, 1, self._row_payoffs.shape[0])
        }
        self.action_space = {
            0: Discrete(self._row_payoffs.shape[0]), 
            1: Discrete(self._row_payoffs.shape[1])
        }

        self._num_stages = config.get("stages", 5)
        self._current_stage = 0
    
    def reset(self):
        self._current_stage = 0
        return {
            0: np.zeros(self.observation_space[0].shape), 
            1: np.zeros(self.observation_space[1].shape)
        }
    
    def step(self, action):
        obs = {
            0: np.zeros(self.observation_space[0].shape), 
            1: np.zeros(self.observation_space[1].shape)
        }
        obs[0][action[1]] = 1
        obs[1][action[0]] = 1

        rewards = {
            0: self._row_payoffs[action[0], action[1]],
            1: self._column_payoffs[action[0], action[1]]
        }

        self._current_stage += 1
        done = self._num_stage <= self._current_stage
        dones = {0: done, 1: done}

        return obs, rewards, dones, None
