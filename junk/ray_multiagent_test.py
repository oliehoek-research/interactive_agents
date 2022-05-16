'''
Script to test using Ray for distributed sampling.
'''
from abc import ABC, abstractmethod
from collections import defaultdict

import gym
import numpy as np
import ray

import torch
import torch.nn as nn
from torch.optim import Adam


class SimpleAgent(ABC):

    @abstractmethod
    def act(self, obs):
        "Sample from the policy for the given observation, return metadata as well"


class SimplePolicy(ABC):

    @abstractmethod
    def make_agent(self):
        "Builds an agent that executes this policy"

    @abstractmethod
    def update(self, data):
        "Synchronize the policy using data from the central learner"


class SimpleLearner:

    @abstractmethod
    def make_policy(self, evaluation=False):
        "Builds a policy that is compatible with this learner"

    @abstractmethod
    def add_batch(self, batch):
        "Sends a new experience batch to the learner"

    @abstractmethod
    def get_update(self):
        "Gets data to send to this learner's actors"

    @abstractmethod
    def learn(self):
        "Runs a training iteration"


class QNet(nn.Module):

    def __init__(self, obs_size, num_actions, hidden_sizes, deuling=False):
        super(QNet, self).__init__()
        self._deuling = deuling

        layers = []
        last_size = obs_size
        for size in hidden_sizes:
            layers.append(nn.Linear(last_size, size))
            last_size = size

        self._hidden = nn.Sequential(*layers)
        self._q_function = nn.Linear(last_size, num_actions)

        if deuling:
            self._value_function = nn.Linear(last_size, 1)

    def forward(self, obs):
        output = self._hidden(obs)
        Q = self._q_function(output)

        if self._deuling:
            V = self._value_function(output)
            Q += V - Q.mean(1, keepdims=True)

        return Q


class ReplayBuffer:

    def __init__(self, action_space, capacity=128):
        self._action_space = action_space
        self._capacity = capacity

        self._index = 0
        self._obs = []
        self._actions = []
        self._rewards = []
        self._dones = []

    def add_trajectory(self, obs, actions, rewards, dones):
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
        obs_batch = np.concatenate([self._obs[idx][:-1] for idx in indices])
        next_obs_batch = np.concatenate([self._obs[idx][1:] for idx in indices])
        action_batch = np.concatenate([self._actions[idx] for idx in indices])
        reward_batch = np.concatenate([self._rewards[idx] for idx in indices])
        done_batch = np.concatenate([self._dones[idx] for idx in indices])

        return obs_batch, next_obs_batch, action_batch, reward_batch, done_batch


class DQNAgent(SimpleAgent):

    def __init__(self, policy):
        self._policy = policy

    def act(self, obs):
        return self._policy.act(obs)


class DQNPolicy(SimplePolicy):

    def __init__(self, observation_space, action_space, hidden_sizes, dueling, epsilon):
        self._action_space = action_space
        self._epsilon = epsilon
        self._q_network = QNet(observation_space.shape[0], action_space.n, hidden_sizes, dueling)
    
    def act(self, obs):
        if np.random.random() <= self._epsilon:
            return self._action_space.sample()

        obs = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self._q_network(obs.unsqueeze(0))  # Add batch dimension
        return q_values.reshape([-1]).argmax().item()

    def make_agent(self):
        return DQNAgent(self)

    def update(self, data):
        self._q_network.load_state_dict(data)


class DQNLearner:

    def __init__(self, observation_space, action_space, config):
        self._observation_space = observation_space
        self._action_space = action_space
        self._batch_size = config.get("batch_size", 4)
        self._num_batches = config.get("num_batches", 4)
        self._sync_interval = config.get("sync_interval", 4)
        self._epsilon = config.get("epsilon", 0.05)
        self._gamma = config.get("gamma", 0.99)
        self._beta = config.get("beta", 0.5)
        self._hidden_sizes = config.get("hiddens", [64])
        self._dueling = config.get("dueling", True)

        self._replay_buffer = ReplayBuffer(config.get("buffer_size", 2048))

        self._online_network = QNet(observation_space.shape[0], action_space.n, self._hidden_sizes, self._dueling)
        self._target_network = QNet(observation_space.shape[0], action_space.n, self._hidden_sizes, self._dueling)

        self._optimizer = Adam(self._online_network.parameters(), lr=config.get("lr", 0.01))
        self._iterations = 0

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

    def make_policy(self, evaluation=False):
        epsilon = 0 if evaluation else self._epsilon
        return DQNPolicy(self._observation_space, self._action_space, self._hidden_sizes, self._dueling, epsilon)

    def add_batch(self, batch):
        for trajectory in batch:
            self._replay_buffer.add_trajectory(*trajectory)

    def get_update(self):
        return self._online_network.state_dict()


class MultiagentSimulator:  # NOTE: This doesn't inherit from any "actor" interface

    def __init__(self, env, policies, policy_mapping_fn, max_steps=None):
        self._env = env
        self._policies = policies
        self._policy_mapping_fn = policy_mapping_fn
        self._max_steps = max_steps

    def sample(self, num_trajectories, max_steps=100):
        batches = defaultdict(list)

        for trajectory in range(num_trajectories):
            observations = defaultdict(list)
            actions = defaultdict(list)
            rewards = defaultdict(list)
            dones = defaultdict(list)

            obs = self._env.reset()
            agents = {}
            for id, ob in obs.items():
                observations[id].append(ob)
                pid = self._policy_mapping_fn(id)
                agents[id] = self._policies[pid].make_agent()

            step = 0
            done = {0: False}

            while step < self._max_steps and not all(done.values()):
                action = {}
                for id, ob in obs.items():                    
                    action[id] = agents[id].act(ob)
                
                obs, reward, done, _ = self._env.step(action)

                for id in obs.keys():
                    observations[id].append(obs[id])
                    actions[id].append(action[id])
                    rewards[id].append(reward[id])
                    dones[id].append(done[id])
                
                step += 1
            
            for id in observations.keys():
                obs = np.array(observations[id], dtype=np.float32)
                action = np.array(actions[id], dtype=np.int64)
                reward = np.array(rewards[id], dtype=np.float32)
                done = np.array(dones[id], dtype=np.float32)
                batches[id].append((obs, action, reward, done))

        return batches


class IndependentTrainer:  # NOTE: This doesn't inherit from any "actor" interface

    def __init__(self, env_fn, learner_fn, agent_ids, config):
        self._num_episodes = config.get("num_episodes", 100)
        self._num_workers = config.get("num_workers", 0)

        env = env_fn()
        learner_config = config.get("learner", {})
        self._learners = {}
        for id in agent_ids:
            obs_space = env.observation_space[id]
            action_space = env.action_space[id]
            self._learners[id] = learner_fn(obs_space, action_space, learner_config)

        max_steps = config.get("max_steps", 100)
        policy_mapping_fn = lambda agent_id: agent_id
        self._workers = []
        if self._num_workers > 0:
            worker_cls = ray.remote(num_cpus=1)(MultiagentSimulator)
            for _ in range(self._num_workers):
                env = env_fn()
                policies = {}
                for id, learner in self._learners.items():
                    policies[id] = learner.make_policy(evaluation=False)
                
                self._workers.append(worker_cls.remote(env, policies, policy_mapping_fn, max_steps))
        else:
            env = env_fn()
            policies = {}
            for id, learner in self._learners.items():
                policies[id] = learner.make_policy(evaluation=False)
            self._workers.append(MultiagentSimulator(env, policies, policy_mapping_fn, max_steps))

    def train(self):
        print("sampling...")
        if self._num_workers > 0:
            requests = [worker.sample.remote(self._num_episodes) for worker in self._workers]  # NOTE: No interface for initiating independent processes
            batches = []
            for request in requests:
                batches.append(ray.get(request))
        else:
            batches = [self._workers[0].sample(self._num_episodes)]
        
        print("learning...")
        for id, learner in self._learners.items():
            for batch in batches:
                learner.add_batch(batch[id])
            
            learner.learn()


class MultiagentWrapper:

    def __init__(self, env):
        self._env = env
        self.observation_space = {0: env.observation_space}
        self.action_space = {0: env.action_space}
    
    def reset(self):
        obs = self._env.reset()
        return {0: obs}
    
    def step(self, action):
        obs, rew, done, info = self._env.step(action[0])
        return {0: obs}, {0: rew}, {0: done}, info
        

if __name__ == "__main__":

    # Configuration
    num_epochs = 20
    config = {
        "num_workers": 1,
        "learner": {}
    }

    # Initialize Ray
    ray.init()  # NOTE: How do we check that Ray is shutting down properly?

    # Define environment
    env_fn = lambda: MultiagentWrapper(gym.make("MountainCar-v0"))  # NOTE: Needed to make the gym interface compatible with our API
    agent_ids = [0]

    # Set learner class
    learner_fn = DQNLearner

    # Initialize Trainer
    trainer = IndependentTrainer(env_fn, learner_fn, agent_ids, config)

    # Start sampling
    print("\n===== Training =====")
    for epoch in range(num_epochs):
        print(f"\nepoch {epoch}")
        trainer.train()
