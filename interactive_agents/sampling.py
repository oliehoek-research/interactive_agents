from collections import defaultdict, namedtuple
import numpy as np

from interactive_agents.envs import get_env_class

Episode = namedtuple("Episode", ["obs", "actions", "rewards", "dones"])

class Sampler:

    def __init__(self, env_name, env_config, policies, policy_fn, max_steps):
        env_cls = get_env_class(env_name)
        self._env = env_cls(env_config)
        self._policies = policies
        self._policy_fn = policy_fn
        self._max_steps = max_steps

    def update_policies(self, updates):
        for id, update in updates.items():
            self._policies[id].update(update)

    def sample(self, num_trajectories):
        batches = defaultdict(list)
        total_samples = 0

        for trajectory in range(num_trajectories):
            observations = defaultdict(list)
            actions = defaultdict(list)
            rewards = defaultdict(list)
            dones = defaultdict(list)
            agents = {}

            obs = self._env.reset()
            for id, ob in obs.items():
                observations[id].append(ob)
                pid = self._policy_fn(id)
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
                batches[id].append(Episode(obs, action, reward, done))
            
            total_samples += step
        
        stats = {"mean_reward": 0}
        for id, batch in batches.items():
            mean_reward = 0
            for episode in batch:
                mean_reward += np.sum(episode.rewards)

            mean_reward /= num_trajectories
            stats[str(id) + "/mean_reward"] = mean_reward
            stats["mean_reward"] += mean_reward

        stats["episodes"] = num_trajectories
        stats["samples"] = total_samples

        return batches, stats
