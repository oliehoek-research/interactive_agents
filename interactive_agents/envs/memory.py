from gymnasium.spaces import Discrete, Box
import numpy as np

from .common import SyncEnv

class Memory(SyncEnv):
    """
    PettingZoo compatible implementation of the "Memory" environemnt from BSuite.
    """

    def __init__(self, config={}):
        self._length = config.get("length", 5)
        self._num_cues = config.get("num_cues", 2)
        self._noise = config.get("noise", 0.0)

        self._agent_id = config.get("agent_id", "agent")

        self._obs_shape = (self._num_cues + 2,)
        self.observation_spaces = {self._agent_id: Box(0, 2, shape=self._obs_shape)}
        self.action_spaces = {self._agent_id: Discrete(self._num_cues)}

        self._truncated = {self._agent_id: False}

        self._current_step = 0
        self._current_cue = 0

    def _obs(self):
        if 0 == self._noise:
            obs = np.zeros(self._obs_shape)
        else:
            obs = self.rng.uniform(0, self._noise, self._obs_shape)

        if 0 == self._current_step:
            obs[-2] += 1
            obs[self._current_cue] += 1
        elif self._length == self._current_step:
            obs[-1] += 1

        return {self._agent_id: obs}

    def reset(self, seed=None):
        self.seed(seed)
        self._current_step = 0
        self._current_cue = self.rng.integers(self._num_cues)
        return self._obs()

    def step(self, action):
        if self._current_step < self._length:
            self._current_step += 1
            reward = 0
            done = False
        else:
            reward = (1 if action[self._agent_id] == self._current_cue else 0)
            done = True

        reward = {self._agent_id: reward}
        terminated = {self._agent_id: done}
        
        return self._obs(), reward, terminated, self._truncated, None
