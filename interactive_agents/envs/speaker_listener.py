from gymnasium.spaces import Discrete, Box
import numpy as np

from .common import SyncEnv

class SpeakerListener(SyncEnv):

    def __init__(self, config={}):  # NOTE: Do we ever actually use "spec_only" anywhere?
        self._num_stages = config.get("stages", 64)
        self._num_cues = config.get("cues", 5)
        self._other_play = config.get("other_play", False)
        self._meta_learning = config.get("meta_learning", False)

        self.observation_spaces = {
            "speaker": Box(0, 1, shape=(self._num_cues * 2,))
        }
        self.action_spaces = {
            "speaker": Discrete(self._num_cues)
        }
        self._truncated = {"speaker": False}

        if not self._meta_learning:
            self.observation_spaces["listener"] = Box(0, 1, shape=(self._num_cues * 2,))
            self.action_spaces["listener"] = Discrete(self._num_cues)
            self._truncated["listener"] = False
        
        self._stage = None
        self._current_cue = None
        self._previous_cue = None
        self._signal = None

        # Channel perumtation for other-play
        self._permutation = None

        # For the meta-learning case only
        self._mapping = None

    def reset(self, seed=None):
        self.seed(seed)

        self._current_cue = self.rng.integers(self._num_cues)
        self._previous_cue = None
        self._stage = 0

        obs = {}
        obs["speaker"] = np.zeros(self._num_cues * 2)
        obs["speaker"][self._current_cue] = 1
        
        if not self._meta_learning:
            obs["listener"] = np.zeros(self._num_cues * 2)
        else:
            self._mapping = self.rng.permutation(self._num_cues)
            self._signal = self.rng.integers(self._num_cues)
        
        if self._other_play:
            self._permutation = self.rng.permutation(self._num_cues)

        return obs

    def step(self, action):
        if not self._meta_learning:
            listener_action = action["listener"]
        else:
            listener_action = self._mapping[self._signal]

        if self._other_play:
            self._signal = self._permutation[action["speaker"]]
        else:
            self._signal = action["speaker"]

        self._stage += 1
        done = (self._stage >= self._num_stages)

        reward = (1 if listener_action == self._previous_cue else 0)

        obs = {}
        rewards = {}
        terminated = {}

        if not self._meta_learning:
            listener_obs = np.zeros(self._num_cues * 2)
            listener_obs[self._signal] = 1

            if self._previous_cue is not None:
                listener_obs[self._num_cues + self._previous_cue] = 1

            obs["listener"] = listener_obs
            rewards["listener"] = reward
            terminated["listener"] = done

        self._previous_cue = self._current_cue
        self._current_cue = self.rng.integers(self._num_cues)

        speaker_obs = np.zeros(self._num_cues * 2)
        speaker_obs[self._current_cue] = 1
        speaker_obs[self._num_cues + listener_action] = 1
        
        obs["speaker"] = speaker_obs
        rewards["speaker"] = reward
        terminated["speaker"] = done

        return obs, rewards, terminated, self._truncated, None
