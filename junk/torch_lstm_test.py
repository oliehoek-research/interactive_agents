# Applies behavioral cloning with an LSTM network to the simple memory game
from collections import namedtuple
import gym
from gym.spaces import Discrete, Box
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

Step = namedtuple("Step", ["obs", "action"])


class MemoryGame(gym.Env):
    '''An instance of the memory game with noisy observations'''

    def __init__(self, length=5, num_cues=2, noise=0.1):
        self.observation_space = Box(0, 2, shape=(num_cues + 2,))
        self.action_space = Discrete(num_cues)
        # TODO: Test if "length" is a list and randomize length if it is
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


def generateDemos(env, episodes):
    demonstrations = []
    for _ in range(episodes):
        current_demo = []
        obs = env.reset()
        done = False

        while not done:
            action = env.expert()
            current_demo.append(Step(obs, action))
            obs, _, done, _ = env.step(action)
    
        demonstrations.append(current_demo)

    return demonstrations


def evaluate(env, model, episodes):
    total_reward = 0
    total_successes = 0

    for _ in range(episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        hidden = model.initial_hidden()

        while not done:
            logits, hidden = model(torch.as_tensor(obs, dtype=torch.float32).reshape(1,1, -1), hidden)
            action = np.argmax(logits.detach().numpy()[0,0])
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        
        total_reward += episode_reward
        if episode_reward > 0:
            total_successes += 1

    return (total_reward / episodes), (total_successes / episodes)


class ReplayBuffer:
    
    def __init__(self, num_actions, capacity=128):
        self._num_actions = num_actions
        self._capacity = capacity

        self._index = 0
        self._obs = []
        self._actions = []
    
    def add(self, episode):
        obs = []
        actions = []

        for step in episode:
            obs.append(step.obs)
            actions.append(step.action)
        
        obs = torch.tensor(obs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        actions = nn.functional.one_hot(actions, self._num_actions)

        if len(obs) < self._capacity:
            self._obs.append(obs)
            self._actions.append(actions)
        else:
            self._obs[self._index] = obs
            self._actions[self._index] = actions

        self._index = (self._index + 1) % self._capacity
    
    def sample(self, batch_size):
        indices = np.random.randint(len(self._obs), size=batch_size)
        obs_batch = [self._obs[idx] for idx in indices]
        action_batch = [self._actions[idx] for idx in indices]

        seq_lens = torch.tensor([len(seq) for seq in obs_batch], dtype=torch.int64)
        obs_batch = nn.utils.rnn.pad_sequence(obs_batch)
        action_batch = nn.utils.rnn.pad_sequence(action_batch)

        return obs_batch, action_batch, seq_lens


class LSTMNet(nn.Module):
    '''Simple recurrent network using LSTMs'''
    
    def __init__(self, input_size, output_size, lstm_size):
        super(LSTMNet, self).__init__()
        self._lstm = nn.LSTM(input_size, lstm_size)
        self._linear = nn.Linear(lstm_size, output_size)
        self._lstm_size = lstm_size

    def forward(self, obs, hidden, seq_lens):
        if seq_lens is None:
            out, hidden = self._lstm(obs, hidden)
        else:
            obs = nn.utils.rnn.pack_padded_sequence(obs, seq_lens, enforce_sorted=False)
            out, hidden = self._lstm(obs, hidden)
            out = nn.utils.rnn.pad_packed_sequence(out)

        out = self._linear(out)

        return out, hidden

    def initial_hidden(self, batch_size=1):
        hidden = torch.zeros((1, batch_size, self._lstm_size), dtype=torch.float32)
        cell = torch.zeros((1, batch_size, self._lstm_size), dtype=torch.float32)
        return hidden, cell


class GRUNet(nn.Module):
    '''Simple recurrent network using GRUs'''
    
    def __init__(self, input_size, output_size, lstm_size):
        super(GRUNet, self).__init__()
        self._lstm = nn.GRU(input_size, lstm_size)
        self._linear = nn.Linear(lstm_size, output_size)
        self._lstm_size = lstm_size

    def forward(self, obs, hidden):
        out, hidden = self._lstm(obs, hidden)
        out = self._linear(out)

        return out, hidden

    def initial_hidden(self, batch_size=1):
        hidden = torch.zeros((1, batch_size, self._lstm_size), dtype=torch.float32)
        return hidden


if __name__ == "__main__":
    env = MemoryGame(15, 4)
    num_demonstrations = 1024
    batch_size = 32
    hidden_size = 10
    training_epochs = 5000
    eval_episodes = 100
    eval_interval = 100
    
    data = generateDemos(env, num_demonstrations)
    buffer = ReplayBuffer(env.action_space.n, capacity=num_demonstrations)

    for episode in data:
        buffer.add(episode)
    
    # model = LSTMNet(env.observation_space.shape[0], env.action_space.n, hidden_size)
    model = GRUNet(env.observation_space.shape[0], env.action_space.n, hidden_size)

    mean_reward, success_rate = evaluate(env, model, eval_episodes)
    print("----- Untrained Model -----")
    print(f"    mean return: {mean_reward}")
    print(f"    success rate: {success_rate * 100}%")

    print("\n===== Training =====")
    optimizer = Adam(model.parameters(), lr=0.001)
    initial_hidden = model.initial_hidden(batch_size)

    for epoch in range(training_epochs):
        obs_batch, action_batch, _ = buffer.sample(batch_size)
        optimizer.zero_grad()
        logits, _ = model(obs_batch, initial_hidden, seq_lens)
        dist = nn.functional.softmax(logits, -1)
        loss = -torch.mean(dist * action_batch)
        loss.backward()
        optimizer.step()

        if 0 == (epoch + 1) % eval_interval:
            mean_reward, success_rate = evaluate(env, model, eval_episodes)
            print(f"\n----- Epoch {epoch + 1} -----")
            print(f"    mean return: {mean_reward}")
            print(f"    success rate: {success_rate * 100}%")
