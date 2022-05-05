import gym
from gym.spaces import Discrete, Box
import numpy as np

class MemoryGame(gym.Env):
    '''An instance of the memory game with noisy observations'''

    def __init__(self, config={}):
        self._length = config.get("length", 5)
        self._num_cues =config.get("num_cues", 2)
        self._noise = config.get("noise", 0.1)
        self._image = config.get("image", False)

        if self._image:
            self._image_size = config.get("image_size", 100)
            self.observation_space = Box(0, 2, shape=(1, self._image_size, self._image_size))
        else:
            self.observation_space = Box(0, 2, shape=(self._num_cues + 2,))
        
        self.action_space = Discrete(self._num_cues)

        self._current_step = 0
        self._current_cue = 0

    def _vector_obs(self):
        obs = np.random.uniform(0, self._noise, self.observation_space.shape)
        if 0 == self._current_step:
            obs[-2] += 1
            obs[self._current_cue] += 1
        elif self._length == self._current_step:
            obs[-1] += 1
        return obs

    def _image_obs(self):
        obs = np.random.uniform(0, self._noise, self.observation_space.shape)

        if 0 == self._current_step:
            slope = self._current_cue * (2.0 / (self._num_cues - 1)) - 1.0
            offset = self._image_size // 2

            for x in range(self._image_size):
                y = int((x - offset) * slope)
                y = max(0, min(self._image_size - 1, y + offset))
                obs[0, x, y] += 1.0
        
        return obs

    def _obs(self):
        if self._image:
            return self._image_obs()
        else:
            return self._vector_obs()

    def reset(self):
        self._current_step = 0
        self._current_cue = np.random.randint(self._num_cues)
        return self._obs()

    def step(self, action):
        if self._current_step < self._length:
            self._current_step += 1
            return self._obs(), 0, False, {}
        else:
            reward = (1 if action == self._current_cue else 0)
            return self._obs(), reward, True, {}


if __name__ == "__main__":
    env_config = {
        "length": 40,
        "num_cues": 2,
        "noise": 0.1,
        "image": True,
        "image_size": 32,
    }

    env = MemoryGame(env_config)
    obs = env.reset()

    print(f"\nImage Observation ({obs.shape}):\n")

    for y in range(obs.shape[2]):
        row = []
        for x in range(obs.shape[1]):
            if obs[0][x][y] > 0.5:
                row.append("#")
            else:
                row.append(".")
        print(" ".join(row))
    
    print("\n")