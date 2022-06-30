from gym.spaces import Discrete, Box
import numpy as np

from .common import MultiagentEnv

class RepeatedGame:
    """
    A finitely repeated bi-matrix game.  Config specifies the payoff matrices.

    If no column-player payoffs are defined, uses the row-player payoffs. If
    'zero_sum' option is true, uses the negative row-player payoffs, otherwise
    assumes the game is fully cooperative.
    """

    def __init__(self, config, spec_only=False):
        assert "row_payoffs" in config, "must at least define row player payoffs"
        self._row_payoffs = np.asarray(config.get("row_payoffs"), dtype=np.float32)

        if "column_payoffs" in config:
            self._column_payoffs = np.asarray(config.get("column_payoffs"), dtype=np.float32)
        elif config.get("zero_sum", False):
            self._column_payoffs = -self._row_payoffs
        else:
            self._column_payoffs = self._row_payoffs

        assert len(self._row_payoffs.shape) == 2, "row payoffs must be a 2-d tensor"
        assert len(self._column_payoffs.shape) == 2, "column payoffs must be a 2-d tensor"
        assert self._row_payoffs.shape == self._column_payoffs.shape, "row and column payoff matrices must be same size"
        
        self.observation_spaces = {
            "row": Box(0, 1, self._row_payoffs.shape[1]), 
            "column": Box(0, 1, self._row_payoffs.shape[0])
        }
        self.action_spaces = {
            "row": Discrete(self._row_payoffs.shape[0]), 
            "column": Discrete(self._row_payoffs.shape[1])
        }

        self._num_stages = config.get("stages", 5)
        self._current_stage = 0
    
    def reset(self):
        self._current_stage = 0
        return {
            "row": np.zeros(self.observation_space["row"].shape), 
            "column": np.zeros(self.observation_space["column"].shape)
        }
    
    def step(self, action):
        obs = {
            "row": np.zeros(self.observation_space["row"].shape), 
            "column": np.zeros(self.observation_space["column"].shape)
        }
        obs["row"][action["column"]] = 1
        obs["column"][action["row"]] = 1

        rewards = {
            "row": self._row_payoffs[action["row"], action["column"]],
            "column": self._column_payoffs[action["row"], action["column"]]
        }

        self._current_stage += 1
        done = self._num_stage <= self._current_stage
        dones = {"row": done, "column": done}

        return obs, rewards, dones, None
