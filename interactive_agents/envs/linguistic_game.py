from gym.spaces import Discrete, Box
import numpy as np

from .common import MultiagentEnv

# NOTE: This enviroment is better suited to the AEC representation, but AEC is hard to integrate with RL
class LinguisticCoordination(MultiagentEnv):
    """
    The 'linguistic' coordination game, in which a speaker privately observes
    a cue which determines the optimal joint action, and uses a 'cheap-talk'
    channel to communicate this before both agents act.
    """

    def __init__(self, config={}, spec_only=False):
        self._num_steps = config.get("stages", 5) * 2
        self._num_actions = config.get("actions", 8)
        self._meta_learning = config.get("meta_learning", False)

        self.observation_spaces = {
            "speaker": Box(0, 1, shape=(self._num_actions * 2 + 1,))
        }

        self.action_spaces = {
            "speaker": Discrete(self._num_actions)
        }
        
        if not self._meta_learning:
            self.observation_spaces["listener"] = Box(0, 1, shape=(self._num_actions + 1,))
            self.action_spaces["listener"] = Discrete(self._num_actions)

        self._current_step = 0
        self._current_type = 0

        self._fixed_language = None
        self._last_statement = None

    def reset(self):
        self._current_type = np.random.randint(0, self._num_actions)
        self._current_step = 0

        obs = {}
        obs["speaker"] = np.zeros(self._num_actions * 2 + 1)
        obs["speaker"][self._num_actions + self._current_type] = 1
        
        if not self._meta_learning:
            obs["listener"] = np.zeros(self._num_actions + 1)
        else:
            self._fixed_language = np.random.permutation(self._num_actions)

        return obs

    def step(self, actions):
        if self._current_step % 2 == 0:  # Last stage was a communication stage
            reward = 0

            obs = {}
            obs["speaker"] = np.zeros(self._num_actions * 2 + 1)
            obs["speaker"][self._num_actions + self._current_type] = 1
            obs["speaker"][-1] = 1

            if not self._meta_learning:
                obs["listener"] = np.zeros(self._num_actions + 1)
                obs["listener"][actions["speaker"]] = 1
                obs["listener"][-1] = 1
            else:
                self._last_statement = actions["speaker"]

        else:  # Last stage was a coordination stage
            if not self._meta_learning:
                listener_action = actions["listener"]
            else:
                listener_action = self._fixed_language[self._last_statement]
                self._last_statement = None

            reward = 1 if actions["speaker"] == listener_action else 0
            reward = reward if actions["speaker"] == self._current_type else 0

            self._current_type = np.random.randint(0, self._num_actions)

            obs = {}
            obs["speaker"] = np.zeros(self._num_actions * 2 + 1)
            obs["speaker"][listener_action] = 1 
            obs["speaker"][self._num_actions + self._current_type] = 1

            if not self._meta_learning:
                obs["listener"] = np.zeros(self._num_actions + 1)
                obs["listener"][actions["speaker"]] = 1

        rewards = {"speaker": reward}

        self._current_step += 1
        done = (self._current_step >= self._num_steps)
        dones = {"speaker": done}

        if not self._meta_learning:
            rewards["listener"] = reward
            dones["listener"] = done

        return obs, rewards, dones, None
