"""Methods and utilities for sampling experiences"""
from collections import defaultdict
import numpy as np

import torch

class Batch(dict):  # NOTE: Seems to organize batches by the "policy" they are associated with, rather than the "agent"
    """A dictionary object representing a multi-agent experience batch"""

    OBS = "obs"
    NEXT_OBS = "next_obs"
    ACTION = "actions"
    REWARD = "rewards"
    TERMINATED = "terminated"
    TRUNCATED = "truncated"

    def __init__(self, batches={}, episodes=0, timesteps=0):
        super(Batch, self).__init__(batches)  # NOTE: Each item in the dictionary is a list of dictionaries, with each being a single episode for a single agent
        self._episodes = episodes
        self._timesteps = timesteps  # NOTE: What is this used for?

    @property
    def episodes(self):
        return self._episodes

    @property
    def timesteps(self):
        return self._timesteps

    def policy_batches(self, policy_ids):  # NOTE: Seems to return a batch associated with a single policy
        batch = {}
        for pid in policy_ids:
            if pid in self:
                batch[pid] = self[pid]
        
        return Batch(batch, self._episodes, self._timesteps)

    def policy_batch(self, policy_id):  # NOTE: Just a utility that returns a single policy
        return self.policy_batches([policy_id])

    def extend(self, batch):
        for policy_id, episodes in batch.items():
            if policy_id not in self:
                self[policy_id] = []
            
            self[policy_id].extend(episodes)
        
        self._episodes += batch.episodes
        self._timesteps += batch.timesteps

    def statistics(self, alt_names=None):  # NOTE: Statistics computed internally by the batch itself
        if alt_names is None:
            alt_names = {pid:pid for pid in self.keys()}
        
        stats = {
            "reward_mean": 0,
            "reward_max": -np.inf,
            "reward_min": np.inf
        }

        for policy_id, agent_batch in self.items():
            if policy_id in alt_names:
                logging_id = alt_names[policy_id]
                r_mean = 0
                r_max = -np.inf
                r_min = np.inf

                for episode in agent_batch:
                    episode_reward = np.sum(episode[Batch.REWARD])
                    r_mean += episode_reward
                    r_max = max(r_max, episode_reward)
                    r_min = min(r_min, episode_reward)

                r_mean /= len(agent_batch)
                stats[str(logging_id) + "/reward_mean"] = r_mean
                stats[str(logging_id) + "/reward_max"] = r_max
                stats[str(logging_id) + "/reward_min"] = r_min

                stats["reward_mean"] += r_mean
                stats["reward_max"] = max(r_max, stats["reward_max"])
                stats["reward_min"] = min(r_min, stats["reward_min"])

        stats["episodes"] = self._episodes
        stats["timesteps"] = self._timesteps

        return stats

# NOTE: This does not seem to correctly handle the policy mapping, find out 
# what is wrong. <- Need to look at what R2D2 itself expects.
#
# R2D2 seems to expect a list of dictionaries, where each dictionary corresponds 
# to an individual trajectory.  (There are mre performant ways of doing this)
#
class BatchBuilder:  # NOTE: Basically a batch with an internal state corresponding to the current episode
    """Used to record a multi-agent batch during sampling"""

    def __init__(self):
        self._policy_batches = defaultdict(list)  # NOTE: Keys are policy IDs, what do the lists consist of?
        self._episodes = 0
        self._timesteps = 0
        
        self._agent_episodes = None
        self._policy_map = None
        self._episode_steps = 0

    def _store_episode(self):
        for agent_id, episode in self._agent_episodes.items():
            d = {}
            d[Batch.ACTION] = np.asarray(episode.pop(Batch.ACTION), np.int64)
            d[Batch.REWARD] = np.asarray(episode.pop(Batch.REWARD), np.float32)
            d[Batch.TERMINATED] = np.asarray(episode.pop(Batch.TERMINATED), np.float32)
            d[Batch.TRUNCATED] = np.asarray(episode.pop(Batch.TRUNCATED), np.float32)

            obs_t = np.asarray(episode.pop(Batch.OBS), np.float32)
            d[Batch.OBS] = obs_t[:-1]
            d[Batch.NEXT_OBS] = obs_t[1:]

            for key, value in episode.items():  # NOTE: This seems to handle policy-specific outputs, such as internal state
                d[key] = np.asarray(value, np.float32)

            if self._policy_map is not None:
                self._policy_batches[self._policy_map[agent_id]].append(d)
            else:
                self._policy_batches[agent_id].append(d)

        self._episodes += 1
        self._timesteps += self._episode_steps

    def start_episode(self, initial_obs, policy_map=None):
        assert self._agent_episodes is None, "Must call 'end_episode()' first to end current episode"
        self._agent_episodes = defaultdict(lambda: defaultdict(list))
        self._policy_map = policy_map
        self._episode_steps = 0

        for agent_id, obs in initial_obs.items():
            self._agent_episodes[agent_id][Batch.OBS].append(obs)

    def end_episode(self):
        if self._episode_steps > 0:
            self._store_episode()
        
        self._agent_episodes = None

    def step(self, obs, actions, rewards, terminated, truncated, fetches):
        assert self._agent_episodes is not None, "Must call 'start_episode()' first to start new episode"
        for agent_id in obs.keys():
            episode = self._agent_episodes[agent_id]

            episode[Batch.OBS].append(obs[agent_id])
            episode[Batch.ACTION].append(actions[agent_id])
            episode[Batch.REWARD].append(rewards[agent_id])
            episode[Batch.TERMINATED].append(terminated[agent_id])
            episode[Batch.TRUNCATED].append(truncated[agent_id])
            
            for key, value in fetches[agent_id].items():                
                episode[key].append(value)
        
        self._episode_steps += 1

    def build(self):
        return Batch(self._policy_batches, self._episodes, self._timesteps)


# NOTE: Seems to be used to de-serialize an arbitrary policy
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
        
        return action.cpu().numpy(), {}


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


# NOTE: Need to update to the latest PettingZoo interface
# NOTE: This needs to change quite a bit, also needs to support vectored environments
# TODO: Enable support for multiple policies maintained by a single actor (needed for SAD, CC methods)
def sample(env, policies, num_episodes=128, max_steps=1e6, policy_fn=None):  # NOTE: Accepts a single environment, rather than a batch
    """Generates a batch of episodes using the given policies"""
    batch = BatchBuilder()
    
    for _ in range(num_episodes):

        # Initialize episode and episode batch
        obs = env.reset()  # NOTE: Need to fix random seeding
        current_step = 0
        done = False

        agents = {}
        policy_map = {}
        terminated = {}
        truncated = {}
        for agent_id in obs.keys():
            if policy_fn is not None:
                policy_id = policy_fn(agent_id)
            else:
                policy_id = agent_id
            
            agents[agent_id] = policies[policy_id].make_agent()
            policy_map[agent_id] = policy_id
            terminated[agent_id] = False
            truncated[agent_id] = False

        batch.start_episode(obs, policy_map)

        # Rollout episode
        while current_step < max_steps and not done:
            actions = {}
            fetches = {}
            for agent_id, ob in obs.items():
                actions[agent_id], fetches[agent_id] = agents[agent_id].act(ob)

            obs, rewards, terminated, truncated, _ = env.step(actions)
            done = all(terminated.values()) or all(truncated.values())

            batch.step(obs, actions, rewards, terminated, truncated, fetches)  # NOTE: All data added on a 'per-agent' basis
            current_step += 1
    
        # TODO: Allow actors to do additional postprocessing
        batch.end_episode()
    
    return batch.build()
