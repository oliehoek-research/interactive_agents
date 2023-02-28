from gymnasium.spaces import Discrete, Box
import numpy as np

from .common import SyncEnv

class MicroHanabi(SyncEnv):  # NOTE: Actually only makes sense as an asynchronous environment (need to add support for this)
    """
    An implementation of the simple two-step card game from
    the SAD and Bayesian-Action-Decoder papers.
    """

    def __init__(self, config={}):
        self._obs_size = self._num_actions * (self._num_players - 1)

        self._pids = [f"agent_{pid}" for pid in range(self._num_players)]

        self.observation_spaces = {
            "agent_0": Box(0, 1, (2,)),
            "agent_1": Box(0, 1, (5,))}
        self.action_spaces = {"agent_0": Discrete(3), "agent_1": Discrete(3)}
        self._truncated = {"agent_0": False, "agent_1": False}

        self._cards = None
        self._stage = None
        self._first_action = None
