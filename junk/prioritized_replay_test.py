# A simplified implementation of the R2D2 architecture, implementing prioretized experience replay
from collections import defaultdict
import gym
from gym.spaces import Discrete, Box
import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Optional, Tuple


class Stopwatch:

    def __init__(self):
        self._started = None
        self._elapsed = 0

    def start(self):
        if self._started is None:
            self._started = time.time()

    def restart(self):
        self._elapsed = 0
        self._started = time.time()

    def stop(self):
        stopped = time.time()
        if self._started is not None:
            self._elapsed += stopped - self._started
            self._started = None

    def reset(self):
        self._elapsed = 0
        self._started = None

    def elapsed(self):
        return self._elapsed


class LSTMNet(nn.Module):
    
    def __init__(self, observation_space, action_space, hidden_size=32, hidden_layers=1, deuling=False):
        super(LSTMNet, self).__init__()
        self._hidden_size = hidden_size
        self._hidden_layers = hidden_layers
        self._deuling = deuling

        input_size = int(np.prod(observation_space.shape))  # NOTE: Need to cast for TorchScript to work
        self._lstm = nn.LSTM(input_size, hidden_size, hidden_layers)
        self._q_function = nn.Linear(hidden_size, action_space.n)

        if deuling:
            self._value_function = nn.Linear(hidden_size, 1)

    def forward(self, 
            obs: torch.Tensor, 
            hidden: Optional[Tuple[torch.Tensor, torch.Tensor]]=None):
        outputs, hidden = self._lstm(obs, hidden)
        Q = self._q_function(outputs)

        if self._deuling:
            V = self._value_function(outputs)
            Q += V - Q.mean(2, keepdim=True)

        return Q, hidden

    @torch.jit.export
    def get_h0(self, batch_size: int=1):
        shape = [self._hidden_layers, batch_size, self._hidden_size]  # NOTE: Shape must be a list for TorchScript serialization to work

        hidden = torch.zeros(shape, dtype=torch.float32)
        cell = torch.zeros(shape, dtype=torch.float32)
        
        return hidden, cell


class PriorityTree:

    def __init__(self, capacity):
        self._capacity = 1
        self._depth = 0

        while self._capacity < capacity:
            self._capacity *= 2
            self._depth += 1

        size = self._capacity * 2 - 1
        self._sums = np.full(size, 0.0)
        self._mins = np.full(size, np.inf)

        self._next_index = 0

    def set(self, indices, priorities):
        priorities = np.asarray(priorities)
        indices = np.asarray(indices, dtype=np.int64)
        indices += self._capacity - 1

        self._sums[indices] = priorities
        self._mins[indices] = priorities

        for _ in range(self._depth):
            indices //= 2
            left = indices * 2
            right = left + 1
            self._sums[indices] = self._sums[left] + self._sums[right]
            self._mins[indices] = np.minimum(self._mins[left], self._mins[right])

    def get(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        return self._sums[indices + self._capacity - 1]

    def min(self):
        return self._mins[0]

    def sum(self):
        return self._sums[0]

    def prefix_index(self, prefix):
        idx = 0
        for _ in range(self._depth):
            next_idx = idx * 2
            if prefix < self._sums[next_idx]:
                idx = next_idx
            else:
                prefix -= self._sums[next_idx]
                idx = next_idx + 1
        
        return idx - self._capacity + 1


def test_priority_tree():
    tree = PriorityTree(7)
    tree.update([1, 2, 5], [1, 3, 2])
    assert tree.min() == 1
    assert tree.sum() == 6
    assert tree.prefix_index(5) == 5


OBS = "obs"
ACTION = "action"
REWARD = "reward"
NEXT_OBS = "next_obs"
DONE = "done"


class ReplayBuffer:
    
    def __init__(self, capacity=128, alpha=0.0):
        self._capacity = capacity
        self._alpha = alpha

        self._next_index = 0
        self._samples = []

        if 0.0 == alpha:
            self._priorities = None
        else:
            self._priorities = PriorityTree(capacity)

    def add(self, sample, priority):
        if len(self._samples) < self._capacity:
            self._samples.append(sample)
        else:
            self._samples[self._next_index] = sample
        
        if self._priorities is not None:
            self._priorities.set(self._next_index, priority ** self._alpha)
        
        self._next_index = (self._next_index + 1) % self._capacity

    def update_priorities(self, indices, priorities):
        if self._priorities is not None:
            priorities = np.asarray(priorities, dtype=np.float32)
            self._priorities.set(indices, priorities ** self._alpha)
    
    def _sample_priority(self, batch_size, beta):
        masses = np.random.random(batch_size) * self._priorities.sum()
        indices = [self._priorities.prefix_index(m) for m in masses]
        
        priorities = self._priorities.get(indices)
        weights = (len(self._samples) * priorities) ** (-beta)

        p_min = self._priorities.min() / self._priorities.sum()
        max_weight = (len(self._samples) * p_min) ** (-beta)

        return indices, weights / max_weight

    def sample(self, batch_size, beta):
        if self._priorities is None:
            weights = np.full(batch_size, 1.0)
            indices = np.random.randint(0, len(self._samples), batch_size)
        else:
            indices, weights = self._sample_priority(batch_size, beta)

        batch = {}
        for key in self._samples[0].keys():
            tensors = []
            for idx in indices:
                tensors.append(self._samples[idx][key])
            
            batch[key] = np.stack(tensors)
        
        return batch, weights
    
    # TODO: Move this logic to the R2D2 agent itself
    def add(self, obs, actions, rewards, dones):
        obs = torch.tensor(obs, dtype=torch.float32, device=self._device)  # NOTE: No need to pre-tensorize as far as I can tell
        actions = torch.tensor(actions, dtype=torch.int64, device=self._device)
        actions = nn.functional.one_hot(actions, self._action_space.n)  # NOTE: should probably do one-hot encoding outside the buffer
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self._device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self._device)

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


class Policy:  # NOTE: What do we use this for?

    def __init__(self, model):
        self._model = model
        self._state = None

    def reset(self, batch_size=1):
        self._state = self._model.get_h0(batch_size)

    def act(self, obs, explore=False):
        q_values, self._state = self._model(obs.reshape([1,1,-1]), self._state)

        return q_values.reshape([-1]).argmax()


class R2D2:

    def __init__(self, 
                env, 
                num_episodes=8, 
                buffer_size=2048,
                batch_size=4,
                num_batches=4,
                sync_interval=4,
                epsilon=0.05,
                gamma=0.99,
                beta=0.5,
                lr=0.01,
                hidden_size=64,
                hidden_layers=1,
                deuling=True,
                device='cpu'):
        self._env = env
        self._num_episodes = num_episodes
        self._batch_size = batch_size
        self._num_batches = num_batches
        self._sync_interval = sync_interval
        self._epsilon = epsilon
        self._gamma = gamma
        self._beta = beta
        self._device = device

        self._replay_buffer = ReplayBuffer(env.action_space, buffer_size, device)

        self._online_network = LSTMNet(env.observation_space, env.action_space, hidden_size, hidden_layers, deuling)
        self._target_network = LSTMNet(env.observation_space, env.action_space, hidden_size, hidden_layers, deuling)
        
        self._online_network = torch.jit.script(self._online_network)
        self._target_network = torch.jit.script(self._target_network)

        # Optional: move models to GPU
        self._online_network.to(device)
        self._target_network.to(device)
        
        self._optimizer = Adam(self._online_network.parameters(), lr=lr)
        self._iterations = 0

        self._state = None

    def _loss(self, obs_batch, next_obs_batch, action_batch, reward_batch, done_batch, mask):
        h0 = self._online_network.get_h0(obs_batch.shape[1])
        h0 = (h0[0].to(self._device), h0[1].to(self._device))
        online_q, _ = self._online_network(obs_batch, h0)  # Need batched history
        target_q, _ = self._target_network(next_obs_batch, h0)

        q_targets = reward_batch + self._gamma * (1 - done_batch) * target_q.max(-1).values
        online_q = (action_batch * online_q).sum(-1)

        errors = nn.functional.smooth_l1_loss(online_q, q_targets.detach(), beta=self._beta, reduction='none')
        return torch.mean(mask * errors)

    def reset(self, batch_size=1):
        self._state = self._online_network.get_h0(batch_size)
        self._state = (self._state[0].to(self._device), self._state[1].to(self._device))

    def act(self, obs, explore=True):
        q_values, self._state = self._online_network(obs.reshape([1,1,-1]), self._state)

        if explore and np.random.random() <= self._epsilon:
            return self._env.action_space.sample()

        return q_values.reshape([-1]).argmax()

    def train(self):  # NOTE: For this script, we just do sampling within the 'train' method
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
                action = self.act(torch.as_tensor(obs, dtype=torch.float32, device=self._device))
                obs, reward, done, _ = self._env.step(action)

                observations.append(obs)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)

            self._replay_buffer.add(observations, actions, rewards, dones)

        for _ in range(self._num_batches):
            batch = self._replay_buffer.sample(self._batch_size)
            self._optimizer.zero_grad()
            loss = self._loss(*batch).mean()
            loss.backward()
            self._optimizer.step()
    
    def save(self, path):
        torch.jit.save(self._online_network, path)


def evaluate(env, policy, num_episodes=30, device='cpu'):
    total_reward = 0
    total_successes = 0

    for _ in range(num_episodes):
        policy.reset()
        obs = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = policy.act(torch.as_tensor(obs, dtype=torch.float32, device=device), explore=False)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        
        total_reward += episode_reward
        if episode_reward > 0:
            total_successes += 1

    return (total_reward / num_episodes), (total_successes / num_episodes)


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


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    stopwatch = Stopwatch()
    stopwatch.start()

    training_epochs = 500

    env = MemoryGame(10, 4)
    # env = CoordinationGame(20, 16, ['fixed'])
    agent = R2D2(env, device=device)

    print("\n===== Training =====")

    for epoch in range(training_epochs):
        agent.train()
        mean_reward, success_rate = evaluate(env, agent, device=device)

        print(f"\n----- Epoch {epoch + 1} -----")
        print(f"    mean return: {mean_reward}")
        print(f"    success rate: {success_rate * 100}%")
    
    # agent.save("torch_r2d2.pt")

    # model = torch.jit.load("torch_r2d2.pt")
    # policy = Policy(model)

    # mean_reward, success_rate = evaluate(env, policy)

    # print(f"\n----- Serialized Model -----")
    # print(f"    mean return: {mean_reward}")
    # print(f"    success rate: {success_rate * 100}%")

    stopwatch.stop()
    print(f"\nElapsed Time: {stopwatch.elapsed()}s")
