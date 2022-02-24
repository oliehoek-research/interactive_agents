"""Test self-play algorithms in repeated matrix games"""
from collections import defaultdict

import gym
from gym.spaces import Discrete, Box
import numpy as np
import torch
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


class IndependentTraining:

    def __init__(self, learners, config={}):
        num_episodes = config.get("num_episodes", 16)
        self._policies = []

        for _ in range(num_episodes):
            self._policies.append(learners)
        
        self._trainable = set(learners.values())

    def next(self):
        return self._policies, self._trainable


class SelfPlay:

    def __init__(self, env, learners, metastrategy, config={}):
        self._env = env
        self._league = metastrategy(learners, config)

    def train(self):
        to_execute, trainable = self._league.next()  # NOTE: Only allows for blind meta-strategies

        def append(item, to):
            for key, value in item.items():
                to[key].append(value)

        total_returns = defaultdict(lambda: 0)

        for policies in to_execute:
            observations = defaultdict(list)
            actions = defaultdict(list)
            rewards = defaultdict(list)
            dones = defaultdict(list)

            for policy in policies.values():
                policy.reset()

            obs = self._env.reset()
            append(obs, observations)
            done = {None: False}

            while not all(done.values()):
                action = {}
                for id, policy in policies.items():
                    action[id] = policy.act(torch.as_tensor(obs[id], dtype=torch.float32))

                obs, reward, done, _ = self._env.step(action)
                append(obs, observations)
                append(action, actions)
                append(reward, rewards)
                append(done, dones)

            for id, returns in rewards.items():
                total_returns[id] += sum(returns)

            for id, policy in policies.items():
                if policy in trainable:
                    policy.learn(observations[id], actions[id], rewards[id], dones[id])
        
        for id in total_returns.keys():
            total_returns[id] /= len(to_execute)
        return total_returns


class RepeatedGame(gym.Env):

    def __init__(self, A, B=None, rounds=1, noise=0.0):
        self._rounds = rounds
        self._noise = noise
        
        if B is None:
            B = A

        assert A.shape == B.shape, "Payoff matrices must be the same shape"

        self._A = A
        self._B = B

        self._row_actions = A.shape[0]
        self._column_actions = A.shape[1]

        self.observation_space = {
            0: Box(0, 1, shape=(self._column_actions,)),
            1: Box(0, 1, shape=(self._row_actions,))
        }
        self.action_space = {
            0: Discrete(self._row_actions),
            1: Discrete(self._column_actions)
        }

        self._current_round = 0
    
    def reset(self):
        self._current_round = 0

        return {
            0: np.zeros(self._column_actions),
            1: np.zeros(self._row_actions)
        }

    def step(self, actions):
        obs = {
            0: np.zeros(self._column_actions),
            1: np.zeros(self._row_actions)
        }

        obs[0][actions[1]] = 1
        obs[1][actions[0]] = 1

        if self._noise > 0:
            noise = self._noise * np.random.normal()
        else:
            noise = 0

        rewards = {
            0: self._A[actions[0], actions[1]] + noise,
            1: self._B[actions[0], actions[1]] + noise
        }

        self._current_round += 1
        done = self._rounds <= self._current_round
        dones = {0: done, 1: done}

        return obs, rewards, dones, {}


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
    M = np.identity(5)
    env = RepeatedGame(M)

    training_epochs = 100
    num_populations = 2

    populations = []

    for population in range(num_populations):
        print(f"\n===== Training Population {population} =====")

        learners = {}
        for id in env.observation_space.keys():
            learners[id] = R2D2(env.observation_space[id], env.action_space[id])
        
        trainer = SelfPlay(env, learners, IndependentTraining)

        for epoch in range(training_epochs):
            print(f"\n----- Epoch {epoch + 1} -----")
            mean_returns = trainer.train()

            for id, mean in mean_returns.items():
                print(f"    {id}, mean return: {mean}")
        
        populations.append(learners)
    
    jpc = cross_evaluate(env, populations)

    print(f"\n===== Joint Policy Correlation Matrix =====")
    print(jpc)
