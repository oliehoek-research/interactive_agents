import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam


class LSTMNet(nn.Module):
    
    def __init__(self, obs_space, action_space, hidden_size, hidden_layers, deuling):
        super(LSTMNet, self).__init__()
        self._hidden_size = hidden_size
        self._hidden_layers = hidden_layers
        self._deuling = deuling

        self._lstm = nn.LSTM(np.prod(obs_space.shape), hidden_size, hidden_layers)
        self._q_function = nn.Linear(hidden_size, action_space.n)

        if deuling:
            self._value_function = nn.Linear(hidden_size, 1)

    def forward(self, obs, hidden):
        obs = torch.flatten(start_dim=2, end_dim=-1)
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

        obs_batch = [self._obs[idx][:-1] for idx in indices]
        action_batch = [self._actions[idx] for idx in indices]
        reward_batch = [self._rewards[idx] for idx in indices]
        done_batch = [self._dones[idx] for idx in indices]

        return obs_batch, action_batch, reward_batch, done_batch


class R2D2Agent:

    def __init__(self, policy, state):
        self._policy = policy
        self._state = state

    def act(self, obs):
        action, self._state = self._policy.act(obs, self._state)
        return action


class R2D2Policy:

    def __init__(self, observation_space, action_space, hidden_size, hidden_layers, dueling, epsilon):
        self._action_space = action_space
        self._epsilon = epsilon
        self._q_network = LSTMNet(observation_space, action_space, hidden_size, hidden_layers, dueling)
    
    def act(self, obs, state):
        if np.random.random() <= self._epsilon:
            return self._action_space.sample()

        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        q_values = self._q_network(obs.unsqueeze(0), state)

        return q_values.reshape([-1]).argmax().item()

    def make_agent(self):
        return R2D2Agent(self, self._q_network.get_h0())

    def update(self, data):
        self._q_network.load_state_dict(data)


class R2D2:

    def __init__(self, observation_space, action_space, config):
        self._observation_space = observation_space
        self._action_space = action_space
        self._batch_size = config.get("batch_size", 4)
        self._num_batches = config.get("num_batches", 4)
        self._sync_interval = config.get("sync_interval", 4)
        self._epsilon = config.get("epsilon", 0.05)
        self._gamma = config.get("gamma", 0.99)
        self._beta = config.get("beta", 0.5)
        self._hidden_size = config.get("hidden_size", 64)
        self._hidden_layers = config.get("hidden_layers", 1)
        self._dueling = config.get("dueling", True)
        
        self._replay_buffer = ReplayBuffer(action_space, config.get("buffer_size", 2048))

        self._online_network = LSTMNet(observation_space, action_space, self._hidden_size, self._hidden_layers, self._dueling)
        self._target_network = LSTMNet(observation_space, action_space, self._hidden_size, self._hidden_layers, self._dueling)
        
        self._optimizer = Adam(self._online_network.parameters(), lr=config.get("lr", 0.01))
        self._iterations = 0

    def _loss(self, obs_batch, next_obs_batch, action_batch, reward_batch, done_batch, mask):
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

        next_obs_batch = obs_batch[:,:-1]
        obs_batch = obs_batch[:,1:]

        h0 = self._online_network.get_h0(obs_batch.shape[1])
        online_q, _ = self._online_network(obs_batch, h0)
        target_q, _ = self._target_network(next_obs_batch, h0)

        q_targets = reward_batch + self._gamma * (1 - done_batch) * target_q.max(-1).values
        online_q = (action_batch * online_q).sum(-1)

        errors = nn.functional.smooth_l1_loss(online_q, q_targets.detach(), beta=self._beta, reduction='none')
        return torch.mean(seq_mask * errors)

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
        return R2D2Policy(self._observation_space, self._action_space, 
            self._hidden_size, self._hidden_layers, self._dueling, 0 if eval else self._epsilon)

    def get_policy_update(self, eval=False):
        return self._online_network.state_dict()
