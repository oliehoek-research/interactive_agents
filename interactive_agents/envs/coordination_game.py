from gym.spaces import Discrete, Box
import numpy as np

class CoordinationGame:

    def __init__(self, config={}):
        self._num_stages = config.get("stages", 5)
        self._num_actions = config.get("actions", 8)
        self._num_players = config.get("players", 2)
        self._shuffle = config.get("shuffle", False)
        self._focal_point = config.get("focal_point", False)
        self._focal_payoff = config.get("focal_payoff", 0.9)
        self._noise = config.get("payoff_noise", 0.0)

        self._obs_size = self._num_actions * (self._num_players - 1)

        self.observation_space = {}
        self.action_space = {}

        for pid in range(self._num_players):
            self.observation_space[pid] = Box(0, 1, shape=(self._obs_size,))
            self.action_space[pid] = Discrete(self._num_actions)
        
        self._current_stage = 0
        self._forward_permutations = None
        self._backward_permutations = None

    def _new_permutations(self):
        self._forward_permutations = {}
        self._backward_permutations = {}
        
        for pid in range(self._num_players):
            if self._focal_point:
                forward = 1 + np.random.permutation(self._num_actions - 1)
                forward = np.concatenate([np.zeros(1,dtype=np.int64), forward])
            else:
                forward = np.random.permutation(self._num_actions)

            backward = np.zeros(self._num_actions, dtype=np.int64)
            for idx in range(self._num_actions):
                backward[forward[idx]] = idx

            self._forward_permutations[pid] = forward
            self._backward_permutations[pid] = backward

    def reset(self):
        self._current_stage = 0

        if self._shuffle:
            self._new_permutations()

        obs = {}
        for pid in range(self._num_players):
            obs[pid] = np.zeros(self._obs_size)

        return obs

    def step(self, actions):
        if self._shuffle:
            for pid in range(self._num_players):
                actions[pid] = self._forward_permutations[pid][actions[pid]]

        obs = {}
        for pid in range(self._num_players):
            obs[pid] = np.zeros(self._obs_size)
            index = 0

            for id, action in actions.items():
                if pid != id:
                    if self._shuffle:
                         action = self._backward_permutations[pid][action]
                    obs[pid][index + action] = 1
                    index += self._num_actions

        if self._focal_point and all(a == 0 for a in actions.values()):
            reward = self._focal_payoff
        elif all(a == actions[0] for a in actions.values()):
            reward = 1 + self._payoff_noise * np.random.normal()
        else:
            reward = 0 + self._payoff_noise * np.random.normal()
        rewards = {pid:reward for pid in range(self._num_players)}

        self._current_stage += 1
        done = self._num_stage <= self._current_stage
        dones = {pid:done for pid in range(self._num_players)}

        return obs, rewards, dones, None
