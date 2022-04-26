"""
A simple, self-contained implementation of the R2D2 algorithm, including prioritized experience replay.
"""
import argparse
from collections import defaultdict
import gym
from gym.spaces import Discrete, Box
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


def make_unique_dir(path):
    sub_path = os.path.join(path, str(0))
    idx = 0

    while os.path.exists(sub_path):
        idx += 1
        sub_path = os.path.join(path, str(idx))
    
    os.makedirs(sub_path)
    return sub_path


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
    def get_h0(self, batch_size: int=1, device: str="cpu"):
        shape = [self._hidden_layers, batch_size, self._hidden_size]  # NOTE: Shape must be a list for TorchScript serialization to work

        hidden = torch.zeros(shape, dtype=torch.float32)
        cell = torch.zeros(shape, dtype=torch.float32)
        
        return hidden.to(self._device), cell.to(self._device)


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
            self._priorities.set(indices, priorities ** self._alpha)

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

        batch = defaultdict(list)
        for idx in indices:
            for key, value in self._samples[idx]:
                batch[key].append(value)

        seq_lens = [len(reward) for reward in batch[REWARD]]

        for key in batch.keys():
            batch[key] = nn.utils.rnn.pad_sequence(batch[key], batch_first=True)

        return batch, weights, seq_lens, indices


class R2D2:

    def __init__(self, env, config={}, device='cpu'):
        self._env = env
        self._eval_iterations = config.get("eval_iterations", 10)
        self._eval_episodes = config.get("eval_episodes", 16)
        self._iteration_episodes = config.get("iteration_episodes", 16)
        self._num_batches = config.get("num_batches", 4)
        self._batch_size = config.get("batch_size", 4)  # NOTE: Need to look at R2D2 configs from stable-baselines, RLLib
        self._sync_iterations = config.get("sync_iterations", 1)
        self._learning_starts = config.get("learning_starts", 100)
        self._gamma = config.get("gamma", 0.99)
        self._beta = config.get("beta", 0.5)
        self._double_q = config.get("double_q", True)
        self._device = device

        # Epsilon-greedy exploration
        self._epsilon = config.get("epsilon_initial", 0.1)
        self._epsilon_iterations = config.get("epsilon_iterations", 1000)
        self._epsilon_decay = self._epsilon - config.get("epsilon_final", 0.01)
        self._epsilon_decay /= self._epsilon_iterations

        # Replay buffer
        self._replay_alpha = config.get("replay_alpha", 0.6)
        self._replay_epsilon = config.get("replay_epsilon", 0.01)
        self._replay_eta = config.get("replay_eta", 0.0)
        self._replay_beta = 0.0
        self._replay_beta_step = 1.0 / config.get("replay_beta_iterations", 1000)
        self._replay_buffer = ReplayBuffer(config.get("buffer_size", 2048), 0.0 != self._replay_alpha)

        # Q-Networks
        dueling = config.get("dueling", True)
        hidden_size = config.get("hidden_size", 64)
        hidden_layers = config.get("hidden_layers", 1)

        self._online_network = LSTMNet(env.observation_space, env.action_space, hidden_size, hidden_layers, dueling)
        self._target_network = LSTMNet(env.observation_space, env.action_space, hidden_size, hidden_layers, dueling)
        
        self._online_network = torch.jit.script(self._online_network)
        self._target_network = torch.jit.script(self._target_network)

        self._online_network.to(device)
        self._target_network.to(device)

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
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        q_values, state = self._online_network(obs.reshape([1,1,-1]), state)
        return q_values.reshape([-1]).numpy(), state
    
    def _priority(self, td_errors):
        abs_td = td_errors.abs()
        max_error = abs_td.max(dim=-1)
        mean_error = abs_td.mean(dim=-1)

        priority = self._replay_eta * max_error + (1 - self._replay_eta) * mean_error
        return (priority + self._replay_epsilon) ** self._replay_alpha

    def _loss(self, batch, weights, seq_lens):
        h0 = self._initial_state(len(weights))

        mask = [torch.ones(l) for l in seq_lens]
        mask = nn.utils.rnn.pad_sequence(mask, batch_first=True)

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
        loss = torch.mean(weights * torch.mean(mask * errors, -1))

        td_errors = online_q.detach() - q_targets
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

            episode = defaultdict(list)
            episode[OBS].append(obs)

            while not done:
                q_values = self._q_values(obs, state)

                if explore and np.random.random() <= self._epsilon:
                    action = self._env.action_space.sample()
                else:
                    action = q_values.argmax()
                
                obs, reward, done, _ = self._env.step(action)

                episode[OBS].append(obs)
                episode[ACTION].append(action)
                episode[REWARD].append(reward)
                episode[DONE].append(done)

                if fetch_q:
                    episode["q_values"].append(q_values)
                    episode["action_q"].append(q_values[action])

                timesteps += 1

            for key in episode.keys():
                episode[key] = np.stack(episode[key])
            
            episode[NEXT_OBS] = episodes[OBS][1:]
            episode[OBS] = episodes[OBS][:-1]

            episodes[ACTION] = np.eye(self._env.action_space.n)[episodes[ACTION]]

            episodes.append(episode)

            total_reward = sum(episode[REWARD])
            r_mean += total_reward
            r_max = max(r_max, total_reward)
            r_min = min(r_min, total_reward)

        return episodes, {
            "reward_mean": r_mean / num_episodes,
            "reward_max": r_max,
            "reward_mean": r_min,
            "episodes": num_episodes,
            "timesteps": timesteps,
        }

    def train(self):
        self._global_timer.start()

        # Sync online and target networks at fixed intervals
        if self._iterations % self._sync_interval == 0:
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

        # Do training updates
        self._learning_timer.start()
        for _ in range(self._num_batches):
            batch, weights, seq_lengths, indices = self._replay_buffer.sample(self._batch_size)
            self._optimizer.zero_grad()
            loss, td_errors = self._loss(batch, weights, seq_lengths)
            loss.backward()
            self._optimizer.step()
            ReplayBuffer.update_priorities(self._priority(td_errors))
        
        self._learning_timer.stop()

        # Update exploration rate
        if self._current_iteration < self._epsilon_iterations:
            self._epsilon -= self._epsilon_decay

        # Update replay beta
        self._replay_beta = min(1.0, self._replay_beta + self._replay_beta_step)

        # Increment iteration
        self._current_iteration += 1

        # Do evaluation if needed
        if self._current_iteration % self._eval_iterations == 0:
            _, eval_stats = self._rollout(self._eval_episodes, explore=False, fetch_q=False)

        # Return statistics
        self._global_timer.stop()
        stats = {}

        for key, value in sample_stats.items():
            stats["sampling/" + key] = value

        for key, value in eval_stats.items():
            stats["eval/" + key] = value

        self._episodes_total += sample_stats["episodes"]
        self._timesteps_total += sample_stats["timesteps"]

        stats["episodes_total"] = self._episodes_total
        stats["timesteps_total"] = self._timesteps_total

        stats["global_time_s"] = self._global_timer.elapsed()
        stats["sampling_time_s"] = self._sampling_timer.elapsed()
        stats["learning_time_s"] = self._learning_timer.elapsed()

        return stats
    
    def save(self, path):
        torch.jit.save(self._online_network, path)


class MemoryGame(gym.Env):
    '''An instance of the memory game with noisy observations'''

    def __init__(self, config={}):
        self._length = config.get("length", 5)
        self._num_cues =config.get("num_cues", 2)
        self._noise = config.get("noise", 0.1)

        self.observation_space = Box(0, 2, shape=(self._num_cues + 2,))
        self.action_space = Discrete(self._num_cues)

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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Get torch device
    device = "cuda" if args.gpu else "cpu"
    print("\nTraining R2D2 on device '{device}'")

    # Load experiment config
    if args.config_file is None:  # Use default config
        print("using default config")
        config = {}
    else:
        print(f"using config file: {args.config_file}")
        with open(args.config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Create results directory
    path = make_unique_dir(args.output_path)
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
                print(f"\nIteration {iteration}")
                print(f"mean eval reward: {stats['eval/reward_mean']}")
                print(f"episodes: {stats['episodes_total']}")
                print(f"time: {stats['global_time_s']}s")

    # Save policy
    learner.save(os.path.join(path, "policy.pt"))
