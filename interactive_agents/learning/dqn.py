from collections import defaultdict

import gym
import numpy as np
import ray

import torch
import torch.nn as nn
from torch.optim import Adam


class QNet(nn.Module):

    def __init__(self, obs_space, action_space, hidden_sizes, deuling):
        super(QNet, self).__init__()
        self._deuling = deuling

        layers = []
        last_size = np.prod(obs_space.shape)
        for size in hidden_sizes:
            layers.append(nn.Linear(last_size, size))
            last_size = size

        self._hidden = nn.Sequential(*layers)
        self._q_function = nn.Linear(last_size, action_space.n)

        if deuling:
            self._value_function = nn.Linear(last_size, 1)

    def forward(self, obs):
        obs = torch.flatten(obs, start_dim=1, end_dim=-1)
        output = self._hidden(obs)
        Q = self._q_function(output)

        if self._deuling:
            V = self._value_function(output)
            Q += V - Q.mean(1, keepdims=True)

        return Q


class ReplayBuffer:

    def __init__(self, capacity=128):
        self._capacity = capacity

        self._index = 0
        self._obs = []
        self._actions = []
        self._rewards = []
        self._dones = []

    def add(self, obs, actions, rewards, dones):
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
        obs_batch = np.concatenate([self._obs[idx][:-1] for idx in indices])
        next_obs_batch = np.concatenate([self._obs[idx][1:] for idx in indices])
        action_batch = np.concatenate([self._actions[idx] for idx in indices])
        reward_batch = np.concatenate([self._rewards[idx] for idx in indices])
        done_batch = np.concatenate([self._dones[idx] for idx in indices])

        return obs_batch, next_obs_batch, action_batch, reward_batch, done_batch


class DQNAgent:

    def __init__(self, policy):
        self._policy = policy

    def act(self, obs):
        return self._policy.act(obs)


class DQNPolicy:

    def __init__(self, observation_space, action_space, hidden_sizes, dueling, epsilon):
        self._action_space = action_space
        self._epsilon = epsilon
        self._q_network = QNet(observation_space.shape[0], action_space.n, hidden_sizes, dueling)
    
    def act(self, obs):
        if np.random.random() <= self._epsilon:
            return self._action_space.sample()

        obs = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self._q_network(obs.unsqueeze(0))
        return q_values.reshape([-1]).argmax().item()

    def make_agent(self):
        return DQNAgent(self)

    def update(self, data):
        self._q_network.load_state_dict(data)


class DQN:

    def __init__(self, observation_space, action_space, config):
        self._observation_space = observation_space
        self._action_space = action_space
        self._batch_size = config.get("batch_size", 4)
        self._num_batches = config.get("num_batches", 4)
        self._sync_interval = config.get("sync_interval", 4)
        self._epsilon = config.get("epsilon", 0.05)
        self._gamma = config.get("gamma", 0.99)
        self._beta = config.get("beta", 0.5)
        self._hidden_sizes = config.get("hiddens", [64])
        self._dueling = config.get("dueling", True)

        self._replay_buffer = ReplayBuffer(config.get("buffer_size", 2048))

        self._online_network = QNet(observation_space.shape[0], action_space.n, self._hidden_sizes, self._dueling)
        self._target_network = QNet(observation_space.shape[0], action_space.n, self._hidden_sizes, self._dueling)

        self._optimizer = Adam(self._online_network.parameters(), lr=config.get("lr", 0.01))
        self._iterations = 0

    def _loss(self, obs_batch, next_obs_batch, action_batch, reward_batch, done_batch):
        obs_batch = torch.as_tensor(obs_batch, dtype=torch.float32)
        next_obs_batch = torch.as_tensor(next_obs_batch, dtype=torch.float32)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
        done_batch = torch.tensor(done_batch, dtype=torch.float32)

        action_batch = torch.as_tensor(action_batch, dtype=torch.int64)
        action_batch = nn.functional.one_hot(action_batch, self._action_space.n)

        online_q = self._online_network(obs_batch)
        target_q = self._target_network(next_obs_batch)

        q_targets = reward_batch + self._gamma * (1 - done_batch) * target_q.max(-1).values
        online_q = (action_batch * online_q).sum(-1)

        errors = nn.functional.smooth_l1_loss(online_q, q_targets.detach(), beta=self._beta, reduction='none')
        return torch.mean(errors)

    def learn(self):
        self._iterations += 1
        if self._iterations % self._sync_interval == 0:
            parameters = self._online_network.state_dict()
            self._target_network.load_state_dict(parameters)

        for _ in range(self._num_batches):
            batch = self._replay_buffer.sample(self._batch_size)
            self._optimizer.zero_grad()
            loss = self._loss(*batch).mean()
            loss.backward()
            self._optimizer.step()
        
        return {}  # TODO: Add statistics


    def add_batch(self, batch):
        for trajectory in batch:
            self._replay_buffer.add(*trajectory)

    def make_policy(self, eval=False):
        return DQNPolicy(self._observation_space, self._action_space, 
            self._hidden_sizes, self._dueling, 0 if eval else self._epsilon)

    def get_policy_update(self, eval=False):
        return self._online_network.state_dict()
