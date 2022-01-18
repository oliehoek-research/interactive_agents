"""N-player repeated coordination game"""
from gym.spaces import Discrete, Box
import numpy as np


class CoordinationGame:

    def __init__(self, config={}, spec_only=False):
        self._num_stages = config.get("stages", 5)
        self._num_actions = config.get("actions", 8)
        self._num_players = config.get("players", 2)
        self._focal_point = config.get("focal_point", False)
        self._focal_payoff = config.get("focal_payoff", 0.9)
        self._noise = config.get("payoff_noise", 0.0)
        self._other_play = config.get("other_play", False)

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
            if 0 == pid:
                forward = np.arange(self._num_actions)
            elif self._focal_point:
                forward = 1 + np.random.permutation(self._num_actions - 1)
                forward = np.concatenate([np.zeros(1,dtype=np.int64), forward])
            else:
                forward = np.random.permutation(self._num_actions)

            backward = np.zeros(self._num_actions, dtype=np.int64)
            for idx in range(self._num_actions):
                backward[forward[idx]] = idx

            self._forward_permutations[pid] = forward
            self._backward_permutations[pid] = backward
    
    def _permuted_obs(self, actions):
        obs = {}
        for pid in range(self._num_players):
            obs[pid] = np.zeros(self._obs_size)
            index = 0

            for id, action in actions.items():
                if pid != id:
                    action = self._backward_permutations[pid][action]
                    obs[pid][index + action] = 1
                    index += self._num_actions
        
        return obs

    def _obs(self, actions=None):
        obs = {}
        for pid in range(self._num_players):
            obs[pid] = np.zeros(self._obs_size)

        if actions is not None:
            for pid in range(self._num_players):
                index = 0

                for id, action in actions.items():
                    if pid != id:
                        obs[pid][index + action] = 1
                        index += self._num_actions
        
        return obs

    def _step(self, actions):

        # Generate reward noise if needed
        if self._noise > 0:
            noise = self._noise * np.random.normal()
        else:
            noise = 0

        # Compute global reward
        if self._focal_point and all(a == 0 for a in actions.values()):
            reward = self._focal_payoff
        elif all(a == actions[0] for a in actions.values()):
            reward = 1 + noise
        else:
            reward = 0 + noise

        rewards = {pid:reward for pid in range(self._num_players)}

        # Determine if final stage reached
        self._current_stage += 1
        done = self._num_stages <= self._current_stage
        dones = {pid:done for pid in range(self._num_players)}

        return rewards, dones

    def reset(self):
        self._current_stage = 0

        if self._other_play:
            self._new_permutations()

        return self._obs()

    def step(self, actions):
        true_actions = actions.copy()  # NOTE: The action array will be used for learning, so don't modify it
        if self._other_play:
            for pid in range(self._num_players):
                true_actions[pid] = self._forward_permutations[pid][actions[pid]]  # NOTE: There was a bug here where we were using the same action dict on both sides

            obs = self._permuted_obs(true_actions)
        else:
            obs = self._obs(true_actions)
        
        rewards, dones = self._step(true_actions)

        return obs, rewards, dones, None
