from collections import defaultdict
import numpy as np

from .common import SyncEnv

class BatchedEnv(SyncEnv):

    active_agents = None
    _active_envs = None
    _current_step = None

    def __init__(self, env_fn, env_config, num_envs=1, max_steps=np.infty):
        self._max_steps = max_steps
        self._envs = [env_fn(env_config) for _ in range(num_envs)]

        self.observation_spaces = {}
        self.action_spaces = {}

        for agent_id in self._envs[0].possible_agents:
            self.observation_spaces[agent_id] = self._envs[0].observation_space(agent_id)
            self.action_spaces[agent_id] = self._envs[0].action_space(agent_id)

    def reset(self, seed=None):
        self.seed(seed)
        self._current_step = 0

        self.active_agents = defaultdict(list)
        self._active_envs = []

        observations = defaultdict(list)
        for env in self._envs:
            obs = env.reset(seed=self.rng.integers(np.iinfo(np.int32).max))
            self._active_envs.append(True)

            for agent_id in self.possible_agents:
                if agent_id in obs:
                    observations[agent_id].append(obs[agent_id])
                    self.active_agents[agent_id].append(True)
                else:
                    observations[agent_id].append(None)
                    self.active_agents[agent_id].append(False)
        
        return observations

    def step(self, actions):
        observations = defaultdict(list)
        rewards = defaultdict(list)
        terminated = defaultdict(list)
        truncated = defaultdict(list)
        infos = defaultdict(list)

        self._current_step += 1

        for index, env in enumerate(self._envs):
            if self._active_envs[index]:
                action = { id:act[index] for id, act in actions.items() }
                obs, reward, term, trunc, info = env.step(action)

            for agent_id in self.possible_agents:
                if self.active_agents[agent_id][index]:
                    if self._current_step >= self._max_steps:
                        trunc[agent_id] = not term[agent_id]

                    if term[agent_id] or trunc[agent_id]:
                        self.active_agents[agent_id][index] = False

                        if not any([self.active_agents[id][index] for id in self.possible_agents]):
                            self._active_envs[index] = False

                    observations[agent_id].append(obs[agent_id])
                    rewards[agent_id].append(reward[agent_id])
                    terminated[agent_id].append(term[agent_id])
                    truncated[agent_id].append(trunc[agent_id])

                    if info is not None and agent_id in info:
                        infos[agent_id].append(info[agent_id])
                    else:
                        infos[agent_id].append(None)
                else:
                    observations[agent_id].append(None)
                    rewards[agent_id].append(None)
                    terminated[agent_id].append(None)
                    truncated[agent_id].append(None)
                    infos[agent_id].append(None)
        
        return observations, rewards, terminated, truncated, infos
