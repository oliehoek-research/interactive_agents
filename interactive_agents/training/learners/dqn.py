"""Simple Torch implementation of DQN"""
from math import ceil
import numpy as np
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam

from interactive_agents.sampling import MultiBatch


class QNet(nn.Module):
    """Dense Q-Network with optional deuling architecture"""

    def __init__(self, obs_space, action_space, hidden_sizes, deuling):
        super(QNet, self).__init__()
        self._deuling = deuling

        # NOTE: Separate variables needed for Torchscript
        input_size = obs_space.shape[0]
        output_size = action_space.n

        layers = []
        last_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(last_size, size))
            layers.append(nn.ReLU())
            last_size = size

        self._hidden = nn.Sequential(*layers)
        self._q_function = nn.Linear(last_size, output_size)

        if deuling:
            self._value_function = nn.Linear(last_size, 1)

    def forward(self, obs):
        output = self._hidden(obs)
        Q = self._q_function(output)

        if self._deuling:
            V = self._value_function(output)
            Q += V - Q.mean(-1, keepdim=True)

        return Q


class QPolicy(nn.Module):
    """Torchscript policy wrapper for Q networks"""

    def __init__(self, model):
        super(QPolicy, self).__init__()
        self._model = model

    def forward(self, obs, state: Optional[torch.Tensor]):
        return self._model(obs).argmax(-1), state
    
    @torch.jit.export
    def initial_state(self, batch_size: int=1):
        return torch.empty((batch_size, 0)) # NOTE: Return empty tensor for Torchscript


class ReplayBuffer:
    """Replay buffer which samples batches of episode rather than steps"""

    def __init__(self, capacity=128):
        self._buffer = []
        self._capacity = capacity
        self._index = 0

    def add_episodes(self, episodes):
        for episode in episodes:
            if len(self._buffer) < self._capacity:
                self._buffer.append(episode)
            else:
                self._buffer[self._index] = episode
            
            self._index = (self._index + 1) % self._capacity

    def sample(self, batch_size):
        obs_batch = []
        next_obs_batch = []
        action_batch = []
        reward_batch = []
        done_batch = []

        indices = np.random.randint(len(self._buffer), size=batch_size)
        for idx in indices:
            episode = self._buffer[idx]
            obs_batch.append(episode[MultiBatch.OBS][:-1])
            next_obs_batch.append(episode[MultiBatch.OBS][1:])
            action_batch.append(episode[MultiBatch.ACTION])
            reward_batch.append(episode[MultiBatch.REWARD])
            done_batch.append(episode[MultiBatch.DONE])

        obs_batch = np.concatenate(obs_batch)
        next_obs_batch = np.concatenate(next_obs_batch)
        action_batch = np.concatenate(action_batch)
        reward_batch = np.concatenate(reward_batch)
        done_batch = np.concatenate(done_batch)

        return obs_batch, next_obs_batch, action_batch, reward_batch, done_batch


class DQNAgent:

    def __init__(self, policy):
        self._policy = policy
    
    def act(self, obs):
        return self._policy.act(obs), {}


class DQNPolicy:

    def __init__(self, observation_space, action_space, hidden_sizes, dueling, epsilon):
        self._action_space = action_space
        self._epsilon = epsilon
        self._q_network = QNet(observation_space, action_space, hidden_sizes, dueling)

    def make_agent(self):
        return DQNAgent(self)

    def act(self, obs):
        if np.random.random() <= self._epsilon:
            return self._action_space.sample()

        obs = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self._q_network(obs.unsqueeze(0))

        return q_values.reshape([-1]).argmax().item()

    def update(self, state):
        self._q_network.load_state_dict(state)


class DQN:

    def __init__(self, observation_space, action_space, config):
        self._observation_space = observation_space
        self._action_space = action_space
        self._batch_size = config.get("batch_size", 4)
        self._batches_per_episode = config.get("batches_per_episode", 1)
        self._sync_interval = config.get("sync_interval", 100)
        self._epsilon = config.get("epsilon", 0.05)
        self._gamma = config.get("gamma", 0.99)
        self._beta = config.get("beta", 0.5)
        self._hidden_sizes = config.get("hiddens", [64])
        self._dueling = config.get("dueling", True)
        self._compile = config.get("compile", True)

        self._replay_buffer = ReplayBuffer(config.get("buffer_size", 1024))

        self._online_network = QNet(observation_space, action_space, self._hidden_sizes, self._dueling)
        self._target_network = QNet(observation_space, action_space, self._hidden_sizes, self._dueling)

        if self._compile:
            self._online_network = torch.jit.script(self._online_network)
            self._target_network = torch.jit.script(self._target_network)

        self._optimizer = Adam(self._online_network.parameters(), lr=config.get("lr", 0.01))
        self._episodes_since_sync = 0

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

    def learn(self, episodes):
        num_episodes = len(episodes)
        self._replay_buffer.add_episodes(episodes)

        self._episodes_since_sync += num_episodes
        if self._episodes_since_sync >= self._sync_interval:
            parameters = self._online_network.state_dict()
            self._target_network.load_state_dict(parameters)
            self._episodes_since_sync = 0

        num_batches = ceil(self._batches_per_episode * num_episodes)
        for _ in range(num_batches):
            batch = self._replay_buffer.sample(self._batch_size)
            self._optimizer.zero_grad()
            loss = self._loss(*batch).mean()
            loss.backward()
            self._optimizer.step()
        
        return {}  # TODO: Add statistics

    def make_policy(self, eval=False):
        return DQNPolicy(self._observation_space, self._action_space, 
            self._hidden_sizes, self._dueling, 0 if eval else self._epsilon)

    def get_update(self, eval=False):
        return self._online_network.state_dict()

    def export_policy(self):
        policy = torch.jit.script(QPolicy(self._online_network))
        policy.eval()  # NOTE: Need to explicitly switch to eval mode

        return torch.jit.freeze(policy, ["initial_state"])
