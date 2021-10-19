from collections import defaultdict
from copy import copy, deepcopy

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


class R2D2:

    def __init__(self, 
                observation_space,
                action_space,
                learn_interval=1,
                sync_interval=32,
                buffer_size=2048,
                batch_size=32,
                epsilon=0.05,
                gamma=0.99,
                beta=0.5,
                lr=0.01,
                hidden_size=64,
                hidden_layers=1,
                deuling=True):
        self._action_space = action_space
        self._learn_interval = learn_interval
        self._sync_interval = sync_interval
        self._batch_size = batch_size
        self._epsilon = epsilon
        self._gamma = gamma
        self._beta = beta

        self._replay_buffer = ReplayBuffer(action_space, buffer_size)

        self._online_network = LSTMNet(observation_space, action_space, hidden_size, hidden_layers, deuling)
        self._target_network = LSTMNet(observation_space, action_space, hidden_size, hidden_layers, deuling)
        
        self._optimizer = Adam(self._online_network.parameters(), lr=lr)

        self._episodes = 0
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
            return self._action_space.sample()

        return q_values.reshape([-1]).argmax()

    def learn(self, obs, actions, rewards, dones):

        # Add latest episode to replay buffer
        self._replay_buffer.add(obs, actions, rewards, dones)
        self._episodes += 1

        # Sync parameters
        if self._episodes % self._sync_interval == 0:
            parameters = self._online_network.state_dict()
            self._target_network.load_state_dict(parameters)

        # Train on batch
        if self._episodes % self._learn_interval == 0:
            batch = self._replay_buffer.sample(self._batch_size)
            self._optimizer.zero_grad()
            loss = self._loss(*batch).mean()
            loss.backward()
            self._optimizer.step()


class SimultaneousPlay:

    def __init__(self, env, learners, num_episodes=16):
        self._env = env
        self._learners = learners
        self._num_episodes = num_episodes

    def train(self):

        def append(item, to):
            for key, value in item.items():
                to[key].append(value)

        total_returns = defaultdict(lambda: 0)

        for _ in range(self._num_episodes):
            observations = defaultdict(list)
            actions = defaultdict(list)
            rewards = defaultdict(list)
            dones = defaultdict(list)

            for learner in self._learners.values():
                learner.reset()

            obs = self._env.reset()
            append(obs, observations)
            done = {0:False}

            while not all(done.values()):
                action = {}
                for id, learner in self._learners.items():
                    action[id] = learners[id].act(torch.as_tensor(obs[id], dtype=torch.float32))

                obs, reward, done, _ = self._env.step(action)
                append(obs, observations)
                append(action, actions)
                append(reward, rewards)
                append(done, dones)

            for id, returns in rewards.items():
                total_returns[id] += sum(returns)

            for id, learner in self._learners.items():
                learner.learn(observations[id], actions[id], rewards[id], dones[id])
        
        for id in total_returns.keys():
            total_returns[id] /= self._num_episodes

        return total_returns


class CoordinationGame(gym.Env):

    def __init__(self, config={}):
        self._num_rounds = config.get("rounds", 5)
        self._num_actions = config.get("actions", 8)
        self._num_players = config.get("players", 2)

        self._obs_size = self._num_actions * (self._num_players - 1)

        self.observation_space = {}
        self.action_space = {}

        for pid in range(self._num_players):
            self.observation_space[pid] = Box(0, 1, shape=(self._obs_size,))
            self.action_space[pid] = Discrete(self._num_actions)
        
        self._current_round = 0

    def reset(self):
        self._current_round = 0
        obs = {}

        for pid in range(self._num_players):
            obs[pid] = np.zeros(self._obs_size)

        return obs

    def step(self, actions):
        obs = {}

        for pid in range(self._num_players):
            p_obs = np.zeros(self._obs_size)
            index = 0

            for id, action in actions.items():
                if pid != id:
                    p_obs[index + action] = 1
                    index += self._num_actions

            obs[pid] = p_obs.reshape(self._obs_size)

        reward = 1 if all(a == actions[0] for a in actions.values()) else 0
        rewards = {pid:reward for pid in range(self._num_players)}

        self._current_round += 1
        done = self._num_rounds <= self._current_round
        dones = {pid:done for pid in range(self._num_players)}

        infos = {pid:None for pid in range(self._num_players)}

        return obs, rewards, dones, infos


class PermutationEnv:

    def __init__(self, env):
        self._env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._action_permutations = None
        self._observation_permutations = None

    def _permute_observation(self, obs):
        for key in obs.keys():
            if obs[key].max() > 0:  # Initial observation will be all zeros, so don't permute
                idx = np.argmax(obs[key])
                obs[key][idx] = 0
                idx = self._observation_permutations[key][idx]
                obs[key][idx] = 1
        
        return obs

    def _permute_action(self, action):
        for key, value in action.items():
            action[key] = self._action_permutations[key][value]
        
        return action

    def reset(self):
        obs = self._env.reset()

        self._action_permutations = {}
        self._observation_permutations = {}
        for id, action_space in self._env.action_space.items():
            forward = np.random.permutation(action_space.n)
            backwards = np.zeros(action_space.n, dtype=np.int64)

            for idx in range(action_space.n):
                backwards[forward[idx]] = idx

            self._action_permutations[id] = forward
            self._observation_permutations[id] = backwards

        return self._permute_observation(obs)
    
    def step(self, action):
        action = self._permute_action(action)
        obs, rewards, dones, infos = self._env.step(action)
        obs = self._permute_observation(obs)

        return obs, rewards, dones, infos


def cross_evaluate(env, populations, num_episodes=10):

    def permutations(num_agents, num_populations):
        num_permutations = num_populations ** num_agents
        for index in range(num_permutations):
            permutation = [0] * num_agents
            idx = index
            for id in range(num_agents):
                permutation[id] = idx % num_populations
                idx = idx // num_populations
            yield permutation

    agent_ids = list(env.observation_space.keys())
    num_agents = len(agent_ids)
    num_populations = len(populations)
    returns = np.zeros(tuple([num_populations] * num_agents))

    for permutation in permutations(num_agents, num_populations):
        print(f"permuation: {permutation}")
        policies = {}
        for id, p in enumerate(permutation):
            agent_id = agent_ids[id]
            policies[agent_id] = populations[p][agent_id]
        total_reward = defaultdict(lambda: 0)

        for _ in range(num_episodes):
            for policy in policies.values():
                policy.reset()

            obs = env.reset()
            done = {0:False}

            while not all(done.values()):
                action = {}
                for id, policy in policies.items():
                    action[id] = policies[id].act(torch.as_tensor(obs[id], dtype=torch.float32))

                obs, reward, done, _ = env.step(action)

                for id, rew in reward.items():
                    total_reward[id] += rew

        idx = tuple(permutation)
        for reward in total_reward.values():
            returns[idx] += reward / num_episodes
        
    return returns


if __name__ == "__main__":
    env = CoordinationGame(config={
        "rounds": 10,
        "actions": 10
    })

    train_env = env
    # train_env = PermutationEnv(train_env)

    training_epochs = 100
    num_populations = 2

    populations = []

    for population in range(num_populations):
        print(f"\n===== Training Population {population} =====")

        learners = {}
        for id in train_env.observation_space.keys():
            learners[id] = R2D2(train_env.observation_space[id], train_env.action_space[id])
        
        trainer = SimultaneousPlay(train_env, learners)

        for epoch in range(training_epochs):
            print(f"\n----- Epoch {epoch + 1} -----")
            mean_returns = trainer.train()

            for id, mean in mean_returns.items():
                print(f"    {id}, mean return: {mean}")
        
        populations.append(learners)
    
    jpc = cross_evaluate(env, populations)

    print(f"\n===== Joint Policy Correlation Matrix =====")
    print(jpc)
