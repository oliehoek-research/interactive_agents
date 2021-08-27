import gym
from gym.spaces import Discrete, Box
import numpy as np

class CoordinationGame(gym.Env):
    '''An instance of the memory game with noisy observations'''

    class Fixed:
        
        def __init__(self, actions):
            self._actions = actions
            self._current_action = None

        def reset(self):
            self._current_action = np.random.randint(self._actions)

        def act(self, opponent_action=None):
            return self._current_action

    class SelfPlay:
        
        def __init__(self, actions):
            self._actions = actions

        def reset(self):
            pass

        def act(self, opponent_action=None):
            if opponent_action is None:
                return np.random.randint(self._actions)

            return opponent_action

    class FictitiousPlay:
        def __init__(self, actions):
            self._actions = actions
            self._counts = np.zeros(actions, dtype=np.int32)

        def reset(self):
            self._counts.fill(0)

        def act(self, opponent_action=None):
            if opponent_action is None:
                return np.random.randint(self._actions)

            self._counts[opponent_action] += 1
            return np.argmax(self._counts)

    def __init__(self, rounds=5, actions=2, partners=['fixed']):
        self.observation_space = Box(0, 1, shape=(actions,))
        self.action_space = Discrete(actions)
        self._rounds = rounds

        self._partners = []
        for partner in set(partners):
            if 'fixed' == partner:
                self._partners.append(self.Fixed(actions))
            elif 'sp' == partner:
                self._partners.append(self.SelfPlay(actions))
            elif 'fp' == partner:
                self._partners.append(self.FictitiousPlay(actions))
            else:
                raise ValueError(f"No partner type '{partner}'")
        
        self._current_round = 0
        self._current_partner = None
        self._opponent_action = None

    def reset(self):
        self._current_round = 0
        self._current_partner = np.random.choice(self._partners)
        self._current_partner.reset()
        self._opponent_action = None
        return np.zeros(self.action_space.n)

    def step(self, action):
        partner_action = self._current_partner.act(self._opponent_action)
        self._opponent_action = action

        obs = np.zeros(self.action_space.n)
        obs[partner_action] = 1

        reward = 1 if action == partner_action else 0
        
        self._current_round += 1
        done = self._rounds <= self._current_round
        
        return obs, reward, done, {}
