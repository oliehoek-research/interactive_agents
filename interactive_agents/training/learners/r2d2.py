"""Simple Torch implementation of R2D2"""
from math import ceil
import numpy as np
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam

from interactive_agents.sampling import MultiBatch


class LSTMNet(nn.Module):
    """LSTM-based Q-Network with optional deuling architecture"""
    
    def __init__(self, obs_space, action_space, hidden_size, hidden_layers, dueling):
        super(LSTMNet, self).__init__()
        self._hidden_size = hidden_size
        self._hidden_layers = hidden_layers
        self._dueling = dueling

        # NOTE: Separate variables needed for Torchscript
        input_size = obs_space.shape[0]
        output_size = action_space.n

        self._lstm = nn.LSTM(input_size, hidden_size, hidden_layers)
        self._q_function = nn.Linear(hidden_size, output_size)

        if dueling:
            self._value_function = nn.Linear(hidden_size, 1)

    def forward(self, 
            obs, 
            state: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        outputs, state = self._lstm(obs, state)
        Q = self._q_function(outputs)

        if self._dueling:
            V = self._value_function(outputs)
            Q += V - Q.mean(2, keepdim=True)

        return Q, state

    @torch.jit.export
    def initial_state(self, batch_size: int=1):
        hidden = torch.zeros((self._hidden_layers, 
            batch_size, self._hidden_size), dtype=torch.float32)
        cell = torch.zeros((self._hidden_layers, 
            batch_size, self._hidden_size), dtype=torch.float32)
        return hidden, cell


class LSTMPolicy(nn.Module):
    """Torchscript policy wrapper for LSTM-based Q networks"""

    def __init__(self, model):
        super(LSTMPolicy, self).__init__()
        self._model = model

    def forward(self, 
            obs, 
            state: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        Q, state = self._model(obs, state)
        return Q.argmax(-1), state
    
    @torch.jit.export
    def initial_state(self, batch_size: int=1):
        return self._model.initial_state(batch_size=batch_size)


class ReplayBuffer:
    """Replay buffer which samples batches of episode rather than steps"""

    def __init__(self, capacity):
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
        action_batch = []
        reward_batch = []
        done_batch = []

        indices = np.random.randint(len(self._buffer), size=batch_size)
        for idx in indices:
            episode = self._buffer[idx]
            obs_batch.append(episode[MultiBatch.OBS])
            action_batch.append(episode[MultiBatch.ACTION])
            reward_batch.append(episode[MultiBatch.REWARD])
            done_batch.append(episode[MultiBatch.DONE])

        return obs_batch, action_batch, reward_batch, done_batch


class R2D2Agent:

    def __init__(self, policy, state):
        self._policy = policy
        self._state = state

    def act(self, obs):
        action, self._state = self._policy.act(obs, self._state)
        return action, {}


class R2D2Policy:

    def __init__(self, 
            observation_space=None, 
            action_space=None, 
            hidden_size=64, 
            hidden_layers=1, 
            dueling=True, 
            epsilon=0):
        assert observation_space is not None, "Must define an observation space"
        assert action_space is not None, "Must define an action space"
        self._action_space = action_space
        self._epsilon = epsilon
        self._q_network = LSTMNet(observation_space, 
            action_space, hidden_size, hidden_layers, dueling)
    
    def act(self, obs, state):
        if np.random.random() <= self._epsilon:
            return self._action_space.sample(), state

        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        q_values, state  = self._q_network(obs.unsqueeze(0), state)

        return q_values.reshape([-1]).argmax().item(), state

    def make_agent(self):
        return R2D2Agent(self, self._q_network.initial_state())

    def update(self, state):
        self._q_network.load_state_dict(state)


class R2D2:

    def __init__(self, observation_space, action_space, config):
        self._observation_space = observation_space
        self._action_space = action_space
        self._batch_size = config.get("batch_size", 4)
        self._batches_per_episode = config.get("batches_per_episode", 1)
        self._sync_interval = config.get("sync_interval", 100)
        self._epsilon = config.get("epsilon", 0.05)
        self._gamma = config.get("gamma", 0.99)
        self._beta = config.get("beta", 0.5)
        self._hidden_size = config.get("hidden_size", 64)
        self._hidden_layers = config.get("hidden_layers", 1)
        self._dueling = config.get("dueling", True)
        self._compile = config.get("compile", True)
        
        self._replay_buffer = ReplayBuffer(config.get("buffer_size", 1024))

        self._online_network = LSTMNet(observation_space, action_space, self._hidden_size, self._hidden_layers, self._dueling)
        self._target_network = LSTMNet(observation_space, action_space, self._hidden_size, self._hidden_layers, self._dueling)
        
        if self._compile:
            self._online_network = torch.jit.script(self._online_network)
            self._target_network = torch.jit.script(self._target_network)

        self._optimizer = Adam(self._online_network.parameters(), lr=config.get("lr", 0.01))
        self._episodes_since_sync = 0

    def _loss(self, obs_batch, action_batch, reward_batch, done_batch):
        obs_batch = [torch.tensor(obs, dtype=torch.float32) for obs in obs_batch]
        reward_batch = [torch.tensor(rewards, dtype=torch.float32) for rewards in reward_batch]
        done_batch = [torch.tensor(dones, dtype=torch.float32) for dones in done_batch]

        action_batch = [torch.tensor(actions, dtype=torch.int64) for actions in action_batch]
        action_batch = [nn.functional.one_hot(actions, self._action_space.n) for actions in action_batch]
        
        seq_mask = [torch.ones_like(rewards) for rewards in reward_batch]

        obs_batch = nn.utils.rnn.pad_sequence(obs_batch)
        action_batch = nn.utils.rnn.pad_sequence(action_batch)
        reward_batch = nn.utils.rnn.pad_sequence(reward_batch)
        done_batch = nn.utils.rnn.pad_sequence(done_batch)
        seq_mask = nn.utils.rnn.pad_sequence(seq_mask)

        next_obs_batch = obs_batch[1:]
        obs_batch = obs_batch[:-1]

        state = self._online_network.initial_state(obs_batch.shape[1])
        online_q, _ = self._online_network(obs_batch, state)
        target_q, _ = self._target_network(next_obs_batch, state)

        q_targets = reward_batch + self._gamma * (1 - done_batch) * target_q.max(-1).values
        online_q = (action_batch * online_q).sum(-1)

        errors = nn.functional.smooth_l1_loss(online_q, q_targets.detach(), beta=self._beta, reduction='none')
        return torch.mean(seq_mask * errors)

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
        return R2D2Policy(self._observation_space, self._action_space, 
            self._hidden_size, self._hidden_layers, self._dueling, 0 if eval else self._epsilon)
    
    def get_update(self, eval=False):
        return self._online_network.state_dict()

    def export_policy(self):
        policy = torch.jit.script(LSTMPolicy(self._online_network))
        policy.eval()  # NOTE: Need to explicitly switch to eval mode

        return torch.jit.freeze(policy, ["initial_state"])
