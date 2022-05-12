"""
A simple, self-contained implementation of the R2D2 algorithm, including prioritized experience replay.
"""
import argparse
from collections import defaultdict
import gym
from gym.spaces import Discrete, Box
from numbers import Number
import numpy as np
import os
import os.path
from tensorboardX import SummaryWriter
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Optional, Tuple
import yaml

class Stopwatch:

    def __init__(self):
        self._started = None
        self._elapsed = 0

    def start(self):
        if self._started is None:
            self._started = time.time()

    def stop(self):
        stopped = time.time()
        if self._started is not None:
            self._elapsed += stopped - self._started
            self._started = None

    def elapsed(self):
        return self._elapsed


def make_unique_dir(path, tag=None):
    if tag is None:
        tag = ""
    else:
        tag = tag + "_"

    sub_path = os.path.join(path, tag + str(0))
    idx = 0

    while os.path.exists(sub_path):
        idx += 1
        sub_path = os.path.join(path, tag + str(idx))
    
    os.makedirs(sub_path)
    return sub_path


def conv_net(input_shape, filters):
    input_channels = input_shape[0]
    image_shape = list(input_shape)[1:]

    layers = []
    for l, (channels, kernel, stride) in enumerate(filters):
        kernel = list(kernel)
        padding = []
        for d in range(len(kernel)):
            kernel[d] += (kernel[d] + 1) % 2
            padding.append(kernel[d] // 2)

        layers.append(nn.Conv2d(
            input_channels,
            channels,
            kernel,
            stride=stride,
            padding=padding
        ))
        input_channels = channels
        
        if isinstance(stride, Number):
            stride = [stride] * len(image_shape)

        for d, s in enumerate(stride):
            image_shape[d] = -(-image_shape[d] // s) 

        if l < len(filters) - 1:
            layers.append(nn.ReLU())
    
    network = nn.Sequential(*layers)
    num_features = int(input_channels * np.prod(image_shape))
    
    return network, num_features


class LSTMNet(nn.Module):
    
    def __init__(self, observation_space, action_space, deuling=False, network_config={}):
        super(LSTMNet, self).__init__()
        self._deuling = deuling

        if len(observation_space.shape) == 1:
            input_size = int(observation_space.shape[0])  # NOTE: Need to cast for TorchScript to work
            self._conv = None
        elif len(observation_space.shape) == 3:
            conv_filters = network_config.get("conv-filters", [
                (8, (5, 5), 2), 
                (16, (5, 5), 2), 
                (16, (5, 5), 2)])
            self._conv, input_size = conv_net(observation_space.shape, conv_filters)
        
        self._hidden_size = network_config.get("hidden_size", 32)
        self._hidden_layers = network_config.get("hidden_layers", 1)

        self._lstm = nn.LSTM(input_size, self._hidden_size, self._hidden_layers)
        self._q_function = nn.Linear(self._hidden_size, action_space.n)

        if deuling:
            self._value_function = nn.Linear(self._hidden_size, 1)

        self._aux_value_function = nn.Linear(self._hidden_size, 1)
        self._aux_value = torch.zeros((1,))

    def forward(self, 
            obs: torch.Tensor, 
            hidden: Optional[Tuple[torch.Tensor, torch.Tensor]]=None):

        if self._conv is not None:
            base_shape = list(obs.shape)  # Should look like [Time x Batch x Channels x W x H]
            obs = obs.reshape([base_shape[0] * base_shape[1]] + base_shape[2:])  # Combine time and batch into a single dim (conv2d can't handle time)
            obs = self._conv(obs) # Should return a new sequence of images [(Time x Batch) x New_Channels x New_W x New_H]
            # obs = torch.flatten(obs, start_dim=1)  # Flatten the images (may not be necessary)
            obs = obs.reshape((base_shape[0], base_shape[1], -1)) # Separate time and batch dimensions
        
        features, hidden = self._lstm(obs, hidden)
        Q = self._q_function(features)

        if self._deuling:
            V = self._value_function(features)
            Q += V - Q.mean(2, keepdim=True)

        self._aux_value = self._aux_value_function(features)

        return Q, hidden

    @torch.jit.export
    def aux_value(self):
        return self._aux_value.reshape([-1])

    @torch.jit.export
    def get_h0(self, batch_size: int=1, device: str="cpu"):
        shape = [self._hidden_layers, batch_size, self._hidden_size]  # NOTE: Shape must be a list for TorchScript serialization to work

        hidden = torch.zeros(shape, dtype=torch.float32)
        cell = torch.zeros(shape, dtype=torch.float32)
        
        return hidden.to(device), cell.to(device)


class GRUNet(nn.Module):
    
    def __init__(self, observation_space, action_space, hidden_size=32, hidden_layers=1, deuling=False):
        super(GRUNet, self).__init__()
        self._hidden_size = hidden_size
        self._hidden_layers = hidden_layers
        self._deuling = deuling

        input_size = int(np.prod(observation_space.shape))  # NOTE: Need to cast for TorchScript to work
        self._lstm = nn.GRU(input_size, hidden_size, hidden_layers)
        self._q_function = nn.Linear(hidden_size, action_space.n)

        if deuling:
            self._value_function = nn.Linear(hidden_size, 1)

    def forward(self, 
            obs: torch.Tensor, 
            hidden: Optional[torch.Tensor]=None):
        outputs, hidden = self._lstm(obs, hidden)
        Q = self._q_function(outputs)

        if self._deuling:
            V = self._value_function(outputs)
            Q += V - Q.mean(2, keepdim=True)

        return Q, hidden

    @torch.jit.export
    def get_h0(self, batch_size: int=1, device: str="cpu"):
        shape = [self._hidden_layers, batch_size, self._hidden_size]  # NOTE: Shape must be a list for TorchScript serialization to work
        cell = torch.zeros(shape, dtype=torch.float32)
        
        return cell.to(device)


class PriorityTree:

    def __init__(self, capacity):
        self._capacity = 1
        self._depth = 0

        while self._capacity < capacity:
            self._capacity *= 2
            self._depth += 1

        size = self._capacity * 2
        self._sums = np.full(size, 0.0)
        self._mins = np.full(size, np.inf)

        self._next_index = 0

    def set(self, indices, priorities):
        priorities = np.asarray(priorities)
        indices = np.asarray(indices, dtype=np.int64)
        indices += self._capacity

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
        return self._sums[indices + self._capacity]

    def min(self):
        return self._mins[1]

    def sum(self):
        return self._sums[1]

    def prefix_index(self, prefix):
        idx = 1
        for _ in range(self._depth):
            next_idx = idx * 2
            if prefix < self._sums[next_idx]:
                idx = next_idx
            else:
                prefix -= self._sums[next_idx]
                idx = next_idx + 1
        
        return idx - self._capacity


def test_priority_tree():  # TODO: Need more tests, this is an easy class to screw up
    tree = PriorityTree(7)
    tree.set([1, 2, 5], [1, 3, 2])
    assert tree.min() == 1, f"tree minimum {tree.min()}"
    assert tree.sum() == 6, f"tree minimum {tree.sum()}"
    assert tree.prefix_index(5) == 5, f"prefix index for sum 5 is {tree.prefix_index(5)}"


OBS = "obs"
ACTION = "action"
REWARD = "reward"
NEXT_OBS = "next_obs"
DONE = "done"


class ReplayBuffer:
    
    def __init__(self, capacity, prioritize=True, device="cpu"):
        self._capacity = capacity
        self._device = device

        self._next_index = 0
        self._samples = []

        if prioritize:
            self._priorities = PriorityTree(capacity)
        else:
            self._priorities = None

    def add(self, samples, priorities):
        indices = []
        for sample in samples:
            for key in sample:
                sample[key] = torch.as_tensor(sample[key], device=self._device)

            if len(self._samples) < self._capacity:
                self._samples.append(sample)
            else:
                self._samples[self._next_index] = sample
            
            indices.append(self._next_index)
            self._next_index = (self._next_index + 1) % self._capacity
        
        if self._priorities is not None:
            priorities = np.asarray(priorities, dtype=np.float32)
            self._priorities.set(indices, priorities)

    def update_priorities(self, indices, priorities):
        if self._priorities is not None:
            priorities = np.asarray(priorities, dtype=np.float32)
            self._priorities.set(indices, priorities)
    
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

        batch = defaultdict(list)
        for idx in indices:
            for key, value in self._samples[idx].items():
                batch[key].append(value)

        seq_lens = [len(reward) for reward in batch[REWARD]]

        for key in batch.keys():
            batch[key] = nn.utils.rnn.pad_sequence(batch[key])

        return batch, weights, seq_lens, indices


class R2D2:

    def __init__(self, env, config={}, device='cpu'):
        self._env = env
        self._eval_iterations = config.get("eval_iterations", 10)
        self._eval_episodes = config.get("eval_episodes", 32)
        self._iteration_episodes = config.get("iteration_episodes", 128)
        self._num_batches = config.get("num_batches", 16)
        self._batch_size = config.get("batch_size", 16)  # NOTE: Need to look at R2D2 configs from stable-baselines, RLLib
        self._sync_iterations = config.get("sync_iterations", 1)
        self._learning_starts = config.get("learning_starts", 10)
        self._gamma = config.get("gamma", 0.99)
        self._beta = config.get("beta", 0.5)
        self._double_q = config.get("double_q", True)
        self._device = device

        # Epsilon-greedy exploration
        self._epsilon = config.get("epsilon_initial", 1)
        self._epsilon_iterations = config.get("epsilon_iterations", 1000)
        self._epsilon_decay = self._epsilon - config.get("epsilon_final", 0.01)
        self._epsilon_decay /= self._epsilon_iterations

        # Replay buffer
        self._replay_alpha = config.get("replay_alpha", 0.6)
        self._replay_epsilon = config.get("replay_epsilon", 0.01)
        self._replay_eta = config.get("replay_eta", 0.5)
        self._replay_beta = 0.0
        self._replay_beta_step = 1.0 / config.get("replay_beta_iterations", 1000)
        self._replay_buffer = ReplayBuffer(config.get("buffer_size", 2048), 0.0 != self._replay_alpha)

        # Q-Networks
        net_cls = GRUNet if config.get("model", "lstm") == "gru" else LSTMNet
        network_config = config.get("model_config", {})
        dueling = config.get("dueling", True)

        self._online_network = net_cls(env.observation_space, env.action_space, dueling, network_config)
        self._target_network = net_cls(env.observation_space, env.action_space, dueling, network_config)
        
        self._online_network = torch.jit.script(self._online_network)
        self._target_network = torch.jit.script(self._target_network)

        self._online_network.to(device)
        self._target_network.to(device)

        self._input_shape = [1, 1] + list(env.observation_space.shape)

        # Optimizer
        self._optimizer = Adam(self._online_network.parameters(), lr=config.get("lr", 0.01))

        # Statistics and timers
        self._global_timer = Stopwatch()
        self._sampling_timer = Stopwatch()
        self._learning_timer = Stopwatch()

        self._timesteps_total = 0
        self._episodes_total = 0

        self._current_iteration = 0

    def _initial_state(self, batch_size=1):
        return self._online_network.get_h0(batch_size, self._device)

    def _q_values(self, obs, state):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        q_values, state = self._online_network(obs.reshape(self._input_shape), state)
        return q_values.detach().reshape([-1]).numpy(), state
    
    def _priority(self, td_errors):
        abs_td = np.abs(td_errors)
        max_error = abs_td.max(axis=0)
        mean_error = abs_td.mean(axis=0)

        priority = self._replay_eta * max_error + (1 - self._replay_eta) * mean_error
        return (priority + self._replay_epsilon) ** self._replay_alpha

    def _loss(self, batch, weights, seq_lens):
        h0 = self._initial_state(len(weights))

        mask = [torch.ones(l) for l in seq_lens]
        mask = nn.utils.rnn.pad_sequence(mask)

        weights = torch.as_tensor(weights, dtype=torch.float32)

        online_q, _ = self._online_network(batch[OBS], h0)
        target_q, _ = self._target_network(batch[NEXT_OBS], h0)

        if self._double_q:
            max_actions = online_q.argmax(-1).unsqueeze(-1)
            target_q = torch.gather(target_q, -1, max_actions).squeeze(-1)
        else:
            target_q, _ = target_q.max(-1)

        online_q = torch.gather(online_q, -1, batch[ACTION].unsqueeze(-1)).squeeze(-1)

        q_targets = batch[REWARD] + self._gamma * (1 - batch[DONE]) * target_q
        q_targets = q_targets.detach()

        errors = nn.functional.smooth_l1_loss(online_q, q_targets, beta=self._beta, reduction='none')
        loss = torch.mean(weights * torch.mean(mask * errors, 0))

        td_errors = torch.abs(online_q - q_targets)
        return loss, td_errors

    def _rollout(self, num_episodes, explore=True, fetch_q=True):
        episodes = []
        r_mean = 0
        r_max = -np.inf
        r_min = np.inf
        timesteps = 0
        
        for _ in range(num_episodes):
            state = self._initial_state()
            obs = self._env.reset()
            done = False

            obs_t = [obs]
            action_t = []
            reward_t = []
            done_t = []

            if fetch_q:
                q_values_t = []
                action_q_t = []

            while not done:
                q_values, state = self._q_values(obs, state)

                if explore and np.random.random() <= self._epsilon:
                    action = self._env.action_space.sample()
                else:
                    action = q_values.argmax()
                
                obs, reward, done, _ = self._env.step(action)

                obs_t.append(obs)
                action_t.append(action)
                reward_t.append(reward)
                done_t.append(done)

                if fetch_q:
                    q_values_t.append(q_values)
                    action_q_t.append(q_values[action])

                timesteps += 1

            episode = {}
            episode[ACTION] = np.asarray(action_t, dtype=np.int64)
            episode[REWARD] = np.asarray(reward_t, dtype=np.float32)
            episode[DONE] = np.asarray(done_t, np.float32)

            obs_t = np.stack(obs_t).astype(np.float32)
            episode[OBS] = obs_t[:-1]
            episode[NEXT_OBS] = obs_t[1:]
            
            if fetch_q:
                episode["q_values"] = np.stack(q_values_t)
                episode["action_q"] = np.asarray(action_q_t, dtype=np.float32)

            episodes.append(episode)

            total_reward = episode[REWARD].sum()
            r_mean += total_reward
            r_max = max(r_max, total_reward)
            r_min = min(r_min, total_reward)

        return episodes, {
            "reward_mean": r_mean / num_episodes,
            "reward_max": r_max,
            "reward_min": r_min,
            "episodes": num_episodes,
            "timesteps": timesteps,
        }

    def _learn(self):
        outputs = defaultdict(list)

        for _ in range(self._num_batches):   

            # Sample batch 
            batch, weights, seq_lengths, indices = \
                self._replay_buffer.sample(self._batch_size, self._beta)
            
            outputs["replay_weight"].append(weights)

            # Compute loss and do gradient update
            self._optimizer.zero_grad()
            loss, td_errors = self._loss(batch, weights, seq_lengths)
            loss.backward()
            self._optimizer.step()

            td_errors = td_errors.detach().numpy()
            outputs["td_error"].append(td_errors)
            outputs["loss"].append(loss.detach().numpy())

            # Update replay priorities
            priorities = self._priority(td_errors)
            self._replay_buffer.update_priorities(indices, priorities)

            outputs["priorities"].append(priorities)
        
        stats = {}
        for key, value in outputs.items():
            value = np.stack(value)
            stats[key + "_mean"] = value.mean()
            stats[key + "_max"] = value.max()
            stats[key + "_min"] = value.min()
        
        return stats

    def train(self):
        self._global_timer.start()
        stats = {}

        # Sync online and target networks at fixed intervals
        if self._current_iteration % self._sync_iterations == 0:
            parameters = self._online_network.state_dict()
            self._target_network.load_state_dict(parameters)

        # Generate training samples
        self._sampling_timer.start()
        if 0.0 != self._replay_alpha:
            samples, sample_stats = self._rollout(self._iteration_episodes, fetch_q=True)

            # Compute sample priorities
            priorities = []
            for sample in samples:
                max_q = sample["q_values"][1:].max(-1)
                q_targets = sample[REWARD].copy()
                q_targets[:-1] += self._gamma * sample[DONE][:-1] * max_q
                
                priorities.append(self._priority(sample["action_q"] - q_targets))

                del sample["q_values"]
                del sample["action_q"]
        else:
            samples, sample_stats = self._rollout(self._iteration_episodes, fetch_q=False)
            priorities = None

        self._replay_buffer.add(samples, priorities)
        self._sampling_timer.stop()

        for key, value in sample_stats.items():
            stats["sampling/" + key] = value

        # Do training updates
        if self._current_iteration >= self._learning_starts:  # NOTE: Stupid stupid stupid, wasn't training anything!!!
            self._learning_timer.start()
            learning_stats = self._learn()
            self._learning_timer.stop()

            for key, value in learning_stats.items():
                stats["learning/" + key] = value  # NOTE: The fact that we don't record learning values for the first few iterations seems to cause problems

        # Update exploration rate
        if self._current_iteration < self._epsilon_iterations:
            self._epsilon -= self._epsilon_decay
        stats["global/epsilon"] = self._epsilon

        # Update replay beta
        self._replay_beta = min(1.0, self._replay_beta + self._replay_beta_step)
        stats["global/replay_beta"] = self._replay_beta

        # Increment iteration
        self._current_iteration += 1

        # Do evaluation if needed
        if self._current_iteration % self._eval_iterations == 0:
            _, eval_stats = self._rollout(self._eval_episodes, explore=False, fetch_q=False)

            for key, value in eval_stats.items():
                stats["eval/" + key] = value

        # Return statistics
        self._global_timer.stop()

        self._episodes_total += sample_stats["episodes"]
        self._timesteps_total += sample_stats["timesteps"]

        stats["global/episodes_total"] = self._episodes_total
        stats["global/timesteps_total"] = self._timesteps_total

        stats["global/total_time_s"] = self._global_timer.elapsed()
        stats["global/sampling_time_s"] = self._sampling_timer.elapsed()
        stats["global/learning_time_s"] = self._learning_timer.elapsed()

        return stats
    
    def save(self, path):
        torch.jit.save(self._online_network, path)


class MemoryGame(gym.Env):
    '''An instance of the memory game with noisy observations'''

    def __init__(self, config={}):
        self._length = config.get("length", 5)
        self._num_cues =config.get("num_cues", 2)
        self._noise = config.get("noise", 0.1)
        self._image = config.get("image", False)

        if self._image:
            self._image_size = config.get("image_size", 100)
            self.observation_space = Box(0, 2, shape=(1, self._image_size, self._image_size))
        else:
            self.observation_space = Box(0, 2, shape=(self._num_cues + 2,))
        
        self.action_space = Discrete(self._num_cues)

        self._current_step = 0
        self._current_cue = 0

    def _vector_obs(self):
        obs = np.random.uniform(0, self._noise, self.observation_space.shape)
        if 0 == self._current_step:
            obs[-2] += 1
            obs[self._current_cue] += 1
        elif self._length == self._current_step:
            obs[-1] += 1
        return obs

    def _image_obs(self):
        obs = np.random.uniform(0, self._noise, self.observation_space.shape)

        if 0 == self._current_step:
            slope = self._current_cue * (2.0 / (self._num_cues - 1)) - 1.0
            offset = self._image_size // 2

            for x in range(self._image_size):
                y = int((x - offset) * slope)
                y = max(0, min(self._image_size - 1, y + offset))
                obs[0, x, y] += 1.0
        
        return obs

    def _obs(self):
        if self._image:
            return self._image_obs()
        else:
            return self._vector_obs()

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


def parse_args():
    parser = argparse.ArgumentParser("Training script for R2D2 with prioritized replay, using the memory game")

    parser.add_argument("-f", "--config-file", type=str, default=None,
                        help="If specified, use config options from this file")
    parser.add_argument("-o", "--output-path", type=str, default="results/debug/R2D2",
                        help="directory in which we should save results")
    parser.add_argument("-i", "--iterations", type=int, default=1000,
                        help="number of training iterations to run experiment for")
    parser.add_argument("-g", "--gpu", action="store_true",
                        help="enable GPU if available")
    parser.add_argument("-t", "--tag", type=str, default="",
                        help="name for experiment")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Get torch device
    device = "cuda" if args.gpu else "cpu"
    print(f"\nTraining R2D2 on device '{device}'")

    # Load experiment config
    if args.config_file is None:  # Use default config
        print("using default config")
        config = {
            "env": {
                "length": 50,
                "num_cues": 2,
                "noise": 0.1,
                "image": True,
            },
            "eval_iterations": 10,
            "eval_episodes": 32,
            "iteration_episodes": 16,
            "num_batches": 8,
            "batch_size": 16,
            "sync_iterations": 5,
            "learning_starts": 20,
            "gamma": 0.99,
            "beta": 0.5,
            "double_q": False,
            "epsilon_initial": 0.5,
            "epsilon_iterations": 500,
            "epsilon_final": 0.01,
            "replay_alpha": 0.0,
            "replay_epsilon": 0.01,
            "replay_eta": 0.5,
            "replay_beta_iterations": 500,
            "buffer_size": 4096,
            "dueling": True,
            "model": "lstm",
            "model_config": {
                "hidden_size": 64,
                "hidden_layers": 1,
                "conv_filters": [
                    (8, (7, 7), 2), 
                    (8, (7, 7), 2), 
                    (8, (7, 7), 2)
                ],
            },
            "lr": 0.001,
        }
    else:
        print(f"using config file: {args.config_file}")
        with open(args.config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Create results directory
    path = make_unique_dir(args.output_path, args.tag)
    print(f"saving results to: {path}")

    # Save config (with torch device and iteration count)
    config["device"] = device
    config["iterations"] = args.iterations
    with open(os.path.join(path, "config.yaml"), 'w') as config_file:
        yaml.dump(config, config_file)

    # Initialize Memory Environemnt
    env = MemoryGame(config.get("env", {}))

    # Initialize R2D2
    learner = R2D2(env, config)  # NOTE: Could initialize environment within learner

     # Start TensorboardX
    with SummaryWriter(path) as writer:
        for iteration in range(args.iterations):
            stats = learner.train()

            for key, value in stats.items():
                writer.add_scalar(key, value, iteration)
            
            if "eval/reward_mean" in stats:
                print(f"\nIteration {iteration + 1}")
                print(f"mean eval reward: {stats['eval/reward_mean']}")
                print(f"episodes: {stats['global/episodes_total']}")
                print(f"time: {stats['global/total_time_s']}s")

    # Save policy
    learner.save(os.path.join(path, "policy.pt"))

    # Load policy
    model = torch.jit.load(os.path.join(path, "policy.pt"))

    # Test policy inference
    obs = env.reset()
    obs = torch.as_tensor(obs, dtype=torch.float32)
    obs = obs.unsqueeze(0)
    obs = obs.unsqueeze(0)

    hidden = model.get_h0()
    q_values, _ = model(obs, hidden)
    aux_value = model

    print(f"Q values of loaded model:\n{q_values.detach()}")
    print(f"Auc value: {aux_value.aux_value().detach()}")
