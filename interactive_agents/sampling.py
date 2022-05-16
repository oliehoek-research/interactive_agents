"""Methods and utilities for sampling experiences"""
from collections import defaultdict
import numpy as np

import torch

class Batch:
    """Represents a multi-agent experience batch"""

    OBS = "obs"
    NEXT_OBS = "next_obs"
    ACTION = "actions"
    REWARD = "rewards"
    DONE = "dones"

    def __init__(self):
        self._policy_batches = defaultdict(list)
        self._agent_episodes = None
        self._policy_map = None

    def _store_episode(self):
        for agent_id, episode in self._agent_episodes.items():
            d = {}
            d[Batch.ACTION] = np.asarray(episode.pop(Batch.ACTION), np.int64)
            d[Batch.REWARD] = np.asarray(episode.pop(Batch.REWARD), np.float32)
            d[Batch.DONE] = np.asarray(episode.pop(Batch.DONE), np.float32)

            obs_t = np.asarray(episode.pop(Batch.OBS), np.float32)
            d[Batch.OBS] = obs_t[:-1]
            d[Batch.NEXT_OBS] = obs_t[1:]

            for key, value in episode.items():
                d[key] = np.asarray(value, np.float32)

            if self._policy_map is not None:
                self._policy_batches[self._policy_map[agent_id]].append(d)
            else:
                self._policy_batches[agent_id].append(d)

    def start_episode(self, initial_obs, policy_map=None):
        if self._agent_episodes is not None:
            self._store_episode()

        self._agent_episodes = defaultdict(lambda: defaultdict(list))
        self._policy_map = policy_map

        for agent_id, obs in initial_obs.items():
            self._agent_episodes[agent_id][Batch.OBS].append(obs)

    def end_episode(self):
        if self._agent_episodes is not None:
            self._store_episode()
        
        self._agent_episodes = None
        self._policy_map = None

    def step(self, obs, actions, rewards, dones, fetches):
        for agent_id in obs.keys():
            episode = self._agent_episodes[agent_id]

            episode[Batch.OBS].append(obs[agent_id])
            episode[Batch.ACTION].append(actions[agent_id])
            episode[Batch.REWARD].append(rewards[agent_id])
            episode[Batch.DONE].append(dones[agent_id])
            
            for key, value in fetches[agent_id].items():                
                episode[key].append(value)

    def policy_batch(self, policy_id):
        return self._policy_batches[policy_id]

    def items(self):
        return self._policy_batches.items()


class FrozenAgent:

    def __init__(self, model, device):
        self._state = model.initial_state(1, device)
        self._model = model
        self._device = device

    def act(self, obs):
        obs = torch.as_tensor(obs, 
            dtype=torch.float32, device=self._device)
        obs = obs.unsqueeze(0)  # Add batch dimension
        obs = obs.unsqueeze(0)  # Add time dimension (for RNNs)

        action, self._state = self._model(obs, self._state)
        action = action.squeeze(0)  # Remove time dimension
        action = action.squeeze(0)  # Remove batch dimension
        
        return action.numpy(), {}


class FrozenPolicy:
    """Wrapper for frozen Torch policies"""

    def __init__(self, model, device="cpu"):
        self._model = model.to(device)
        self._device = device

    def make_agent(self):
        return FrozenAgent(self._model, self._device)

    def update(self, update):
        """Dummy to make sure we are not trying to update frozen polices"""
        raise NotImplementedError("Frozen policies cannot be updated")
    
    @property
    def model(self):  # NOTE: Why do we need this?
        return self._model


# TODO: Enable support for multiple policies maintained by a single actor (needed for SAD, CC methods)
def sample(env, policies, num_episodes=128, max_steps=1e6, policy_fn=None):
    """Generates a batch of episodes using the given policies"""
    batch = Batch()
    total_steps = 0
    
    for _ in range(num_episodes):

        # Initialize episode and episode batch
        obs = env.reset()
        current_step = 0

        agents = {}
        policy_map = {}
        dones = {}
        for agent_id in obs.keys():
            if policy_fn is not None:
                policy_id = policy_fn(agent_id)
            else:
                policy_id = agent_id
            
            agents[agent_id] = policies[policy_id].make_agent()
            policy_map[agent_id] = policy_id
            dones[agent_id] = False

        batch.start_episode(obs, policy_map)

        # Rollout episode
        while current_step < max_steps and not all(dones.values()):
            actions = {}
            fetches = {}
            for agent_id, ob in obs.items():
                actions[agent_id], fetches[agent_id] = agents[agent_id].act(ob)

            obs, rewards, dones, _ = env.step(actions)

            batch.step(obs, actions, rewards, dones, fetches)
            current_step += 1
    
        # TODO: Allow actors to do additional postprocessing
        batch.end_episode()
        total_steps += current_step
    
    stats = {
        "reward_mean": 0,
        "reward_max": -np.inf,
        "reward_min": np.inf
    }

    for policy_id, agent_batch in batch.items():
        r_mean = 0
        r_max = -np.inf
        r_min = np.inf

        for episode in agent_batch:
            episode_reward = np.sum(episode[Batch.REWARD])
            r_mean += episode_reward
            r_max = max(r_max, episode_reward)
            r_min = min(r_min, episode_reward)

        r_mean /= len(agent_batch)
        stats[str(policy_id) + "/reward_mean"] = r_mean
        stats[str(policy_id) + "/reward_max"] = r_max
        stats[str(policy_id) + "/reward_min"] = r_min

        stats["reward_mean"] += r_mean
        stats["reward_max"] = max(r_max, stats["reward_max"])
        stats["reward_min"] = min(r_min, stats["reward_min"])

    stats["episodes"] = num_episodes
    stats["timesteps"] = total_steps

    return batch, stats
