from gym.spaces import Discrete, Box
import numpy as np

from .common import MultiagentEnv

class MemoryGame(MultiagentEnv):  # TODO: What do they call this in the BSuite paper?
    """
    Abstract T-maze environment with noisy observations.  Similar to BSuite.

    Implemented as a multi-agent environment for compatability.
    """

    def __init__(self, config, spec_only=False):
        self._length = config.get("length", 5)
        self._num_cues = config.get("num_cues", 2)
        self._noise = config.get("noise", 0.0)

        self._agent_id = config.get("agent_id", "agent")
        self._obs_shape = (self._num_cues + 2,)
        self.observation_spaces = {self._agent_id: Box(0, 2, shape=self._obs_shape)}
        self.action_spaces = {self._agent_id: Discrete(self._num_cues)}
  
        self._current_step = 0
        self._current_cue = 0

    def _obs(self):
        if 0 == self._noise:
            obs = np.zeros(self._obs_shape)
        else:
            obs = np.random.uniform(0, self._noise, self._obs_shape)

        if 0 == self._current_step:
            obs[-2] += 1
            obs[self._current_cue] += 1
        elif self._length == self._current_step:
            obs[-1] += 1

        return {self._agent_id: obs}

    def reset(self):
        self._current_step = 0
        self._current_cue = np.random.randint(self._num_cues)
        return self._obs()

    def step(self, action):
        if self._current_step < self._length:
            self._current_step += 1
            return self._obs(), {self._agent_id: 0}, {self._agent_id: False}, None
        else:
            reward = (1 if action[self._agent_id] == self._current_cue else 0)
            return self._obs(), {self._agent_id: reward}, {self._agent_id: True}, None
