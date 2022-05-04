from gym.spaces import Discrete, Box
import numpy as np

from .common import MultiagentEnv

# TODO: This enviroment is better suited to the AEC representation
class LinguisticCoordination(MultiagentEnv):
    """
    The 'linguistic' coordination game, in which a speaker privately observes
    a cue which determines the optimal joint action, and uses a 'cheap-talk'
    channel to communicate this before both agents act.
    """

    def __init__(self, config={}, spec_only=False):
        self._num_steps = config.get("stages", 5) * 2
        self._num_actions = config.get("actions", 8)

        self.observation_spaces = {
            "speaker": Box(0, 1, shape=(self._num_actions * 2 + 1,)),
            "listener": Box(0, 1, shape=(self._num_actions + 1,))
        }
        self.action_spaces = {
            "speaker": Discrete(self._num_actions),
            "listener": Discrete(self._num_actions)
        }
        
        self._current_step = 0
        self._current_type = 0

    def reset(self):
        self._current_type = np.random.randint(0, self._num_actions)
        self._current_step = 0

        speaker_obs = np.zeros(self._num_actions * 2 + 1)
        speaker_obs[self._num_actions + self._current_type] = 1
        
        listener_obs = np.zeros(self._num_actions + 1)

        return {
            "speaker": speaker_obs,
            "listener": listener_obs
        }

    def step(self, actions):
        if self._current_step % 2 == 0:  # Last stage was a communication stage
            reward = 0

            speaker_obs = np.zeros(self._num_actions * 2 + 1)
            speaker_obs[actions["listener"]] = 1 
            speaker_obs[self._num_actions + self._current_type] = 1
            speaker_obs[-1] = 1
                   
            listener_obs = np.zeros(self._num_actions + 1)
            listener_obs[actions["speaker"]] = 1
            listener_obs[-1] = 1

        else:  # Last stage was a coordination stage
            reward = 1 if actions["speaker"] == actions["listener"] else 0
            reward = reward if actions["speaker"] == self._current_type else 0

            self._current_type = np.random.randint(0, self._num_actions)

            speaker_obs = np.zeros(self._num_actions * 2 + 1)
            speaker_obs[actions["listener"]] = 1 
            speaker_obs[self._num_actions + self._current_type] = 1
                   
            listener_obs = np.zeros(self._num_actions + 1)
            listener_obs[actions["speaker"]] = 1

        obs = {
            "speaker": speaker_obs,
            "listener": listener_obs
        }
        
        rewards = {
            "speaker": reward,
            "listener": reward
        }

        self._current_step += 1
        done = (self._current_step >= self._num_steps)
        dones = {
            "speaker": done,
            "listener": done
        }

        return obs, rewards, dones, None
