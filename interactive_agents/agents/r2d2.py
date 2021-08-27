import gym
from gym.spaces import Discrete, Box
import numpy as np
import torch
from torch._C import TensorType
from torch.functional import Tensor
import torch.nn as nn
from torch.optim import Adam


class LSTMNet(nn.Module):
    
    def __init__(self, observation_space, action_space, hidden_size=32, hidden_layers=1, deuling=False):
        super(LSTMNet, self).__init__()
        self._hidden_size = hidden_size
        self._hidden_layers = hidden_layers
        self._deuling = deuling

        self._lstm = nn.LSTM(np.prod(observation_space.shape), hidden_size, hidden_layers)
        self._q_function = nn.Linear(hidden_size, action_space.n)

        if deuling:
            self._value_function = nn.Linear(hidden_size, 1)

    def forward(self, obs, hidden):
        # obs = obs.flatten(2)
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
        actions = nn.functional.one_hot(actions, self._action_space.n)
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
        
        obs_batch = [self._obs[idx][:-1] for idx in indices]
        next_obs_batch = [self._obs[idx][1:] for idx in indices]
        
        action_batch = [self._actions[idx] for idx in indices]
        reward_batch = [self._rewards[idx] for idx in indices]
        done_batch = [self._dones[idx] for idx in indices]
        mask = [torch.ones_like(self._dones[idx]) for idx in indices]

        obs_batch = nn.utils.rnn.pad_sequence(obs_batch)
        next_obs_batch = nn.utils.rnn.pad_sequence(next_obs_batch)
        action_batch = nn.utils.rnn.pad_sequence(action_batch)
        reward_batch = nn.utils.rnn.pad_sequence(reward_batch)
        done_batch = nn.utils.rnn.pad_sequence(done_batch)
        mask = nn.utils.rnn.pad_sequence(mask)

        return obs_batch, next_obs_batch, action_batch, reward_batch, done_batch, mask


class R2D2:  # TODO: This should implement some generic agent-interface

    def __init__(self, 
                env, 
                num_episodes=8, 
                buffer_size=2048,
                batch_size=64,
                num_batches=4,
                sync_interval=4,
                epsilon=0.05,
                gamma=0.99,
                beta=0.5,
                lr=0.01,
                hidden_size=64,
                hidden_layers=1,
                deuling=True):  # TODO: Replace these parameters with a config dict
        self._env = env
        self._num_episodes = num_episodes
        self._batch_size = batch_size
        self._num_batches = num_batches
        self._sync_interval = sync_interval
        self._epsilon = epsilon
        self._gamma = gamma
        self._beta = beta
        
        self._replay_buffer = ReplayBuffer(env.action_space, buffer_size)

        self._online_network = LSTMNet(env.observation_space, env.action_space, hidden_size, hidden_layers, deuling)
        self._target_network = LSTMNet(env.observation_space, env.action_space, hidden_size, hidden_layers, deuling)
        
        self._optimizer = Adam(self._online_network.parameters(), lr=lr)
        self._iterations = 0

        self._state = None

    def _loss(self, obs_batch, next_obs_batch, action_batch, reward_batch, done_batch, mask):
        h0 = self._online_network.get_h0(obs_batch.shape[1])
        online_q, _ = self._online_network(obs_batch, h0)  # Need batched history
        target_q, _ = self._target_network(next_obs_batch, h0)

        q_targets = reward_batch + self._gamma * (1 - done_batch) * target_q.max(-1).values
        online_q = (action_batch * online_q).sum(-1)

        errors = nn.functional.smooth_l1_loss(online_q, q_targets.detach(), beta=self._beta, reduction='none')
        return torch.mean(mask * errors)

    def reset(self, batch_size=1):
        self._state = self._online_network.get_h0(batch_size)

    def act(self, obs, explore=True):
        q_values, self._state = self._online_network(obs.reshape([1,1,-1]), self._state)

        if explore and np.random.random() <= self._epsilon:
            return self._env.action_space.sample()

        return q_values.reshape([-1]).argmax()

    def train(self):
        self._iterations += 1
        if self._iterations % self._sync_interval == 0:
            parameters = self._online_network.state_dict()
            self._target_network.load_state_dict(parameters)

        for _ in range(self._num_episodes):
            observations = []
            actions = []
            rewards = []
            dones = []

            self.reset()
            obs = self._env.reset()
            observations.append(obs)
            done = False

            while not done:
                action = self.act(torch.as_tensor(obs, dtype=torch.float32))
                obs, reward, done, _ = self._env.step(action)

                observations.append(obs)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)

            self._replay_buffer.add(observations, actions, rewards, dones)

        for _ in range(self._num_batches):
            batch = self._replay_buffer.sample(self._num_batches)
            self._optimizer.zero_grad()
            loss = self._loss(*batch).mean()
            loss.backward()
            self._optimizer.step()