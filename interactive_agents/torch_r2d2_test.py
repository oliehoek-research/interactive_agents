# A simplified implementation of the R2D2 architecture, using the memory game as a benchmark to test recurrent performance
from collections import namedtuple
import gym
from gym.spaces import Discrete, Box
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam


class LSTMNet(nn.Module):
    
    def __init__(self, observation_space, action_space, hidden_size=32, hidden_layers=1, deuling=False):
        super(LSTMNet, self).__init__()
        self._hidden_size = hidden_size
        self._hidden_layers = hidden_layers
        self._deuling = deuling

        self._lstm = nn.LSTM(np.prod(observation_space.shape), hidden_size, hidden_layers)
        self._q_function = nn.Linear(hidden_size, action_space)

        if deuling:
            self._value_function = nn.Linear(hidden_size, 1)

    def forward(self, obs, hidden):
        obs = obs.flatten(2)
        outputs, hidden = self._lstm(obs, hidden)
        Q = self._q_function(outputs)

        if self._deuling:
            V = self._value_function(outputs)
            Q += V - Q.mean(2, keepdims=True)

        return Q, hidden

    def get_h0(self, batch_size=1):
        hidden = torch.zeros((self._hidden_layers, batch_size, self._hidden_size), dtype=torch.float32)
        cell = torch.zeros((self._hidden_layers, batch_size, self._hidden_size), dtype=torch.float32)
        return hidden, cell


class ReplayBuffer:
    
    def __init__(self, action_space, capacity=128):
        self._action_space = action_space
        self._capacity = capacity

        self._index = 0
        self._obs = []
        self._actions = []
        self._rewards = []
        self._dones = []
    
    def add(self, obs, actions, rewards, dones):
        obs = torch.tensor(obs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        actions = nn.functional.one_hot(actions, self._num_actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        if len(obs) < self._capacity:
            self._obs.append(obs)
            self._actions.append(actions)
            self._rewards.append(rewards)
            self._dones.append(dones)
        else:
            self._obs[self._index] = obs
            self._actions[self._index] = actions
            self._rewards[self._index] = rewards
            self._dones[self._index] = dones

        self._index = (self._index + 1) % self._capacity
    
    def sample(self, batch_size):
        indices = np.random.randint(len(self._actions), size=batch_size)
        obs_batch = [self._obs[idx] for idx in indices]
        action_batch = [self._actions[idx] for idx in indices]
        reward_batch = [self._rewards[idx] for idx in indices]
        done_batch = [self._dones[idx] for idx in indices]

        seq_lens = [len(seq) for seq in obs_batch]
        obs_batch = nn.utils.rnn.pad_sequence(obs_batch)
        action_batch = nn.utils.rnn.pad_sequence(action_batch)
        reward_batch = nn.utils.rnn.pad_sequence(reward_batch)
        done_batch = nn.utils.rnn.pad_sequence(done_batch, padding_value=1)

        return obs_batch, action_batch, reward_batch, done_batch, seq_lens


class R2D2:

    def __init__(self, 
                env, 
                num_episodes=8, 
                buffer_size=128,
                batch_size=8,
                num_batches=8,
                sync_interval=4,
                epsilon=0.15,
                gamma=0.95,
                beta=0.5,
                lr=0.001,
                hidden_size=32,
                hidden_layers=1,
                deuling=True):
        self._env = env
        self._num_episodes = num_episodes
        self._batch_size = batch_size
        self._num_batches = num_batches
        self._epsilon = epsilon
        self._gamma = gamma
        self._beta = beta
        
        self._replay_buffer = ReplayBuffer(env.action_space, buffer_size)

        self._online_network = LSTMNet(env.observation_space, env.action_space, hidden_size, hidden_layers, deuling)
        self._target_network = LSTMNet(env.observation_space, env.action_space, hidden_size, hidden_layers, deuling)
        
        self._optimizer = Adam(self._online_network.parameters(), lr=lr)

    def _loss(self, obs_batch, action_batch, reward_batch, done_batch, seq_lens):
        obs_batch = nn.utils.rnn.pack_padded_sequence(obs_batch[:-1], seq_lens[:-1], enforce_sorted=False)
        next_obs_batch = nn.utils.rnn.pack_padded_sequence(obs_batch[1:], seq_lens[1:], enforce_sorted=False)
        
        online_q, _ = self._online_network(obs_batch)
        target_q, _ = self._target_network(next_obs_batch)

        online_q = nn.utils.rnn.pad_packed_sequence(online_q)
        target_q = nn.utils.rnn.pad_packed_sequence(target_q)
        
        q_targets = reward_batch + self._gamma * (1 - done_batch) * target_q.max(-1)
        online_q = (action_batch * online_q).sum(-1)

        return nn.functional.smooth_l1_loss(online_q, q_targets.detach(), beta=self._beta)

    def _act(self):
        pass

    def act(self, obs, history):
        pass

    def train(self):
        pass


class MemoryGame(gym.Env):
    '''An instance of the memory game with noisy observations'''

    def __init__(self, length=5, num_cues=2, noise=0.1):
        self.observation_space = Box(0, 2, shape=(num_cues + 2,))
        self.action_space = Discrete(num_cues)
        self._length = length
        self._num_cues = num_cues 
        self._noise = noise       
        self._current_step = 0
        self._current_cue = 0

    def _obs(self):
        obs = np.random.uniform(0, self._noise, self.observation_space.shape)
        if 0 == self._current_step:
            obs[-2] += 1
            obs[self._current_cue] += 1
        elif self._length == self._current_step:
            obs[-1] += 1
        return obs

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

    def expert(self):
        if self._current_step < self._length:
            return self.action_space.sample()
        else:
            return self._current_cue