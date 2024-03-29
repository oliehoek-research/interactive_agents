# NOTE: Modify this to use Torchscript Tracing, which is more 
# stable.  Simpler question first, does this still work?
'''Test the us of TorchScript to export and import of models with graph structure'''
from collections import namedtuple
import gym
from gym.spaces import Discrete, Box
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Union, Tuple, Optional

Sample = namedtuple("Step", ["obs", "action"])


# NOTE: This is the same memory game we have used in a lot of places (need to reduce duplicate code)
class MemoryGame(gym.Env):
    '''The n-step memory game with noisy observations'''

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


# NOTE: Generates roll-outs in a given environment
def generate_data(env, episodes):
    data = []
    for _ in range(episodes):
        current_seq = []
        obs = env.reset()
        done = False

        while not done:
            action = env.expert()
            current_seq.append(Sample(obs, action))
            obs, _, done, _ = env.step(action)
    
        data.append(current_seq)

    return data


# NOTE: Runs evluation episodes using a policy model (doesn't seem to use the "FrozenPolicy" class)
def evaluate(env, model, episodes):
    total_reward = 0
    total_successes = 0

    for _ in range(episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        hidden = model.get_h0()

        while not done:
            # TODO: Switching to ONNX may change how the policy needs to be evaluated
            logits, hidden = model(torch.as_tensor(obs, dtype=torch.float32).reshape(1,1, -1), hidden)
            action = np.argmax(logits.detach().numpy()[0,0])
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        
        total_reward += episode_reward
        if episode_reward > 0:
            total_successes += 1

    return (total_reward / episodes), (total_successes / episodes)


# NOTE: Simple replay buffer - why do we need this since we don't seem to be doing RL?
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

        seq_mask = [torch.ones(len(seq), dtype=torch.float32) for seq in obs_batch]
        seq_mask = nn.utils.rnn.pad_sequence(seq_mask)
       
        obs_batch = nn.utils.rnn.pad_sequence(obs_batch)
        action_batch = nn.utils.rnn.pad_sequence(action_batch)

        return obs_batch, action_batch, seq_mask


# NOTE: As the comment says, a simple LSTM network class
class LSTMNet(nn.Module):
    '''Simple LSTM network class'''

    def __init__(self, input_size, output_size, lstm_size):
        super(LSTMNet, self).__init__()
        self._lstm = nn.LSTM(input_size, lstm_size)
        self._linear = nn.Linear(lstm_size, output_size)
        self._lstm_size = lstm_size

    def forward(self, obs, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        out, hidden = self._lstm(obs, hidden)
        out = self._linear(out)

        return out, hidden

    @torch.jit.export
    def get_h0(self, batch_size: int=1):
        hidden = torch.zeros((1, batch_size, self._lstm_size), dtype=torch.float32)
        cell = torch.zeros((1, batch_size, self._lstm_size), dtype=torch.float32)
        return hidden, cell


if __name__ == "__main__":

    '''
    seq = torch.tensor([[[1,2,3,4],[5,6,7,8]]], dtype=torch.float32)
    h0 = (torch.zeros((1,2,32)), torch.zeros((1,2,32)))

    class RNN(nn.Module):

        def __init__(self):
            super(RNN, self).__init__()
            self.rnn = nn.LSTM(4,32)
            # self.rnn = torch.jit.script(nn.LSTM(4,32))  # Doesn't fix anything
        
        def forward(self, input, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]]):  # Note: Cannot pass two arguments to LSTM, seems to throw off torchscript
            return self.rnn(input, hidden)
    
    lstm = torch.jit.script(RNN())

    # lstm = nn.LSTM(4,32)
    # lstm = torch.jit.script(lstm)

    print(lstm(seq, h0))
    exit()
    '''

    # Configuration
    env = MemoryGame(10, 4)
    num_demonstrations = 1024
    batch_size = 32
    hidden_size = 10
    training_epochs = 3000
    eval_interval = 100
    eval_episodes = 128
    
    # Generate Data
    data = generate_data(env, num_demonstrations)
    buffer = ReplayBuffer(env.action_space.n, capacity=num_demonstrations)

    for episode in data:
        buffer.add(episode)
    
    # Initialize model
    model = LSTMNet(env.observation_space.shape[0], env.action_space.n, hidden_size)

    # NOTE: Need to add support for tracing rather than scripting
    model = torch.jit.script(model)  # NOTE: Is there a real performance advantage to doing this in advance?

    # Train pytoch model
    print("\n===== Training Model =====")
    optimizer = Adam(model.parameters(), lr=0.001)
    initial_hidden = model.get_h0(batch_size)  # NOTE: One of the problems with tracing is how we implement initial states

    # NOTE: Actually doing behvioral cloning, and using the replay buffer to generate batches
    for epoch in range(training_epochs):
        obs_batch, action_batch, seq_mask = buffer.sample(batch_size)
        optimizer.zero_grad()
        logits, _ = model(obs_batch, initial_hidden)
        
        # likelihoods = nn.functional.softmax(logits, -1)
        likelihoods = nn.functional.log_softmax(logits, -1)
        
        likelihoods = torch.sum(action_batch * likelihoods, -1)
        loss = -torch.mean(seq_mask * likelihoods)
        loss.backward()
        optimizer.step()

        if 0 == (epoch + 1) % eval_interval:
            mean_reward, success_rate = evaluate(env, model, eval_episodes)
            print(f"\n----- Epoch {epoch + 1} -----")
            print(f"    mean return: {mean_reward}")
            print(f"    success rate: {success_rate * 100}%")

    # Export model to .pt file
    torch.jit.save(model, "torch_lstm.pt")

    # Import model from .pt file
    model = torch.jit.load("torch_lstm.pt")

    mean_reward, success_rate = evaluate(env, model, eval_episodes)
    print(f"\n----- Serialized Model -----")
    print(f"    mean return: {mean_reward}")
    print(f"    success rate: {success_rate * 100}%")

    # Copy model
    model.eval()  # NOTE: Are models serielized in training mode?
    model = torch.jit.freeze(model, ["get_h0"])  # NOTE: Why do we need this?
    mean_reward, success_rate = evaluate(env, model, eval_episodes)
    print(f"\n----- Frozen Model -----")
    print(f"    mean return: {mean_reward}")
    print(f"    success rate: {success_rate * 100}%")
