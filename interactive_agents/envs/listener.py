from gym.spaces import Discrete, Box
import numpy as np

from .common import MultiagentEnv

class ListenerEnv(MultiagentEnv):
    """
    In each stage, the agent 'listens' for one-hot encoded cues, and must then 
    take the appropriate action for the given cue to receive a reward.
    """

    def __init__(self, config, spec_only=False):  # NOTE: Do we ever actually use "spec_only" anywhere?
        self._num_stages = config.get("stages", 100)
        self._num_cues = config.get("cues", 10)
        self._agent_id = config.get("agent_id", "listener")
        self._identity = config.get("identity", False)

        self.observation_spaces = self._wrap(Box(0, 1, shape=(self._num_cues * 2,)))
        self.action_spaces = self._wrap(Discrete(self._num_cues))

        self._stage = None
        self._cue = None
        self._mapping = None

    def _wrap(self, item):
        return { self._agent_id : item}

    def _unwrap(self, item):
        return item[self._agent_id]

    def reset(self):
        if self._identity:
            self._mapping = np.arange(self._num_cues)
        else:
            self._mapping = np.random.permutation(self._num_cues)
        
        self._stage = 0
        self._cue = np.random.randint(self._num_cues)

        obs = np.zeros(self._num_cues * 2)
        obs[self._cue] = 1
        
        return self._wrap(obs)

    def step(self, action):
        action = self._unwrap(action)

        reward = 1 if self._mapping[self._cue] == action else 0

        self._stage += 1
        done = (self._stage >= self._num_stages)

        obs = np.zeros(self._num_cues * 2)
        if not done:
            obs[self._num_cues + self._mapping[self._cue]] = 1  # NOTE: Need to make sure the learner can observe the previous cue
            self._cue = np.random.randint(self._num_cues)
            obs[self._cue] = 1

        return self._wrap(obs), self._wrap(reward), self._wrap(done), None
