"""Methods and utilities for sampling experiences"""
from collections import defaultdict
import numpy as np

import torch

class MultiBatch:
    """Represents a multi-agent experience batch"""

    OBS = "obs"
    ACTION = "actions"
    REWARD = "rewards"
    DONE = "dones"

    def __init__(self):
        self._policy_batches = defaultdict(list)
        self._agent_episodes = None
        self._policy_map = None

    def _store_episode(self):
        for agent_id, episode in self._agent_episodes.items():
            for key in episode.keys():
                episode[key] = np.array(episode[key])

            if self._policy_map is not None:
                batch = self._policy_batches[self._policy_map[agent_id]]
            else:
                batch = self._policy_batches[agent_id]

            batch.append(episode)

    def start_episode(self, policy_map=None):
        if self._agent_episodes is not None:
            self._store_episode()

        self._agent_episodes = defaultdict(lambda: defaultdict(list))
        self._policy_map = policy_map

    def end_episode(self):
        if self._agent_episodes is not None:
            self._store_episode()
        
        self._agent_episodes = None
        self._policy_map = None

    def append(self, field, values):
        for agent_id, value in values.items():
            self._agent_episodes[agent_id][field].append(value)

    def step(self, obs, actions, rewards, dones, fetches):
        for agent_id in obs.keys():
            episode = self._agent_episodes[agent_id]

            episode[MultiBatch.OBS].append(obs[agent_id])
            episode[MultiBatch.ACTION].append(actions[agent_id])
            episode[MultiBatch.REWARD].append(rewards[agent_id])
            episode[MultiBatch.DONE].append(dones[agent_id])
            
            for key, value in fetches[agent_id].items():                
                episode[key].append(value)

    def policy_batch(self, policy_id):
        return self._policy_batches[policy_id]

    def items(self):
        return self._policy_batches.items()


class FrozenPolicy:
    """Wrapper for frozen Torch policies"""

    def __init__(self, model):
        self._model = model

    def make_agent(self):

        class Agent:

            def __init__(self, model):
                self._state = model.initial_state(batch_size=1)
                self._model = model

            def act(self, obs):
                obs = torch.as_tensor(obs, dtype=torch.float32)
                obs = obs.unsqueeze(0)  # NOTE: Add batch dimension
                obs = obs.unsqueeze(0)  # NOTE: Add time dimension (for RNNs)

                action, self._state = self._model(obs, self._state)
                action = action.squeeze(0)  # NOTE: Remove time dimension
                action = action.squeeze(0)  # NOTE: Remove batch dimension
                
                return action.numpy(), {}

        return Agent(self._model)

    def update(self, update):
        """Dummy to make sure we are not trying to update frozen polices"""
        raise NotImplementedError("Frozen policies cannot be updated")
    
    @property
    def model(self):
        return self._model


def sample(env, policies, num_episodes, max_steps, policy_fn=None):
    """Generates a batch of episodes"""
    batch = MultiBatch()
    total_steps = 0
    
    for episode in range(num_episodes):
        obs = env.reset()
        step = 0

        agents = {}
        map = {}
        dones = {}
        for agent_id in obs.keys():
            if policy_fn is not None:
                policy_id = policy_fn(agent_id)
            else:
                policy_id = agent_id
            
            agents[agent_id] = policies[policy_id].make_agent()
            map[agent_id] = policy_id
            dones[agent_id] = False

        batch.start_episode(map)  # NOTE: Why do we need the policy map
        batch.append(MultiBatch.OBS, obs)

        while step < max_steps and not all(dones.values()):
            actions = {}
            fetches = {}
            for agent_id, ob in obs.items():
                actions[agent_id], fetches[agent_id] = agents[agent_id].act(ob)

            obs, rewards, dones, _ = env.step(actions)

            batch.step(obs, actions, rewards, dones, fetches)
            step += 1
    
        batch.end_episode()
        total_steps += step
    
    stats = {"mean_reward": 0}
    for policy_id, agent_batch in batch.items():
        mean_reward = 0
        for episode in agent_batch:
            mean_reward += np.sum(episode[MultiBatch.REWARD])

        mean_reward /= len(agent_batch)
        stats[str(policy_id) + "/mean_reward"] = mean_reward
        stats["mean_reward"] += mean_reward

    stats["episodes"] = num_episodes
    stats["samples"] = total_steps

    return batch, stats
