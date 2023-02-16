from gym.spaces import Discrete, Box
import numpy as np

from .common import MultiagentEnv

class SpeakerListenerEnv(MultiagentEnv):

    def __init__(self, config, spec_only=False):  # NOTE: Do we ever actually use "spec_only" anywhere?
        self._num_stages = config.get("stages", 64)
        self._num_cues = config.get("cues", 5)
        self._meta_learning = config.get("meta_learning", False)

        self.observation_spaces = {
            "speaker": Box(0, 1, shape=(self._num_cues * 2,))
        }
        self.action_spaces = {
            "speaker": Discrete(self._num_cues)
        }

        if not self._meta_learning:
            self.observation_spaces["listener"] = Box(0, 1, shape=(self._num_cues * 2,))
            self.action_spaces["listener"] = Discrete(self._num_cues)

        self._stage = None
        self._current_cue = None
        self._previous_cue = None
        self._signal = None

        # For the meta-learning case only
        self._mapping = None

    def reset(self):
        self._current_cue = np.random.randint(self._num_cues)
        self._previous_cue = None
        self._stage = 0

        obs = {}
        obs["speaker"] = np.zeros(self._num_cues * 2)
        obs["speaker"][self._current_cue] = 1
        
        if not self._meta_learning:
            obs["listener"] = np.zeros(self._num_cues * 2)
        else:
            self._mapping = np.random.permutation(self._num_cues)
            self._signal = np.random.randint(self._num_cues)

        return obs

    def step(self, action):
        if not self._meta_learning:
            listener_action = action["listener"]
        else:
            listener_action = self._mapping[self._signal]

        self._signal = action["speaker"]

        self._stage += 1
        done = (self._stage >= self._num_stages)

        reward = (1 if listener_action == self._previous_cue else 0)

        obs = {}
        rewards = {}
        dones = {}

        if not self._meta_learning:
            listener_obs = np.zeros(self._num_cues * 2)
            listener_obs[self._signal] = 1

            if self._previous_cue is not None:
                listener_obs[self._num_cues + self._previous_cue] = 1

            obs["listener"] = listener_obs
            rewards["listener"] = reward
            dones["listener"] = done

        self._previous_cue = self._current_cue
        self._current_cue = np.random.randint(self._num_cues)

        speaker_obs = np.zeros(self._num_cues * 2)
        speaker_obs[self._current_cue] = 1
        speaker_obs[self._num_cues + listener_action]
        
        obs["speaker"] = speaker_obs
        rewards["speaker"] = reward
        dones["speaker"] = done

        return obs, rewards, dones, None
