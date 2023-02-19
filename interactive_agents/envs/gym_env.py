import gymnasium as gym
from gymnasium.wrappers import RecordVideo, TimeLimit
import numpy as np
import time

from .common import SyncEnv

class GymEnv(SyncEnv):
    """Multi-agent wrapper for single-agent OpenAI Gym environments."""

    def __init__(self, config={}):  # TODO: Allow config to specify preprocessor stack
        assert "name" in config, "must specify name of gym environment"
        self._env = gym.make(config.get("name"))

        self._agent_id = config.get("agent_id", "agent")

        self.observation_spaces = {self._agent_id: self._env.observation_space}
        self.action_spaces = {self._agent_id: self._env.action_space}
    
    def reset(self, seed=None):
        obs, _ = self._env.reset(seed=seed)
        return {self._agent_id: obs}
    
    def step(self, action):
        obs, rew, term, trunc, info = self._env.step(action[self._agent_id])
        obs = {self._agent_id: obs}
        rew = {self._agent_id: rew}
        term = {self._agent_id: term}
        trunc = {self._agent_id: trunc}
        return obs, rew, term, trunc, info
    
    def visualize(self, 
                  policies={},
                  policy_fn=None,
                  max_episodes=None, 
                  max_steps=None,
                  speed=1,
                  record_path=None,
                  headless=False,
                  **kwargs):  # TODO: Need to think more about the visualization pipeline
        env = self._env  # NOTE: We will apply multiple wrappers around the base environment

        # Enforce step limit
        if max_steps is not None:
            env =TimeLimit(env, max_episode_steps=max_steps)  # NOTE: We don't need to use a wrapper for this

        # Enable video recording if requested
        if record_path is not None:

            # NOTE: Need to provide the episode_trigger to record every episode
            env = RecordVideo(env, record_path, episode_trigger=lambda id: True)  # TODO: How do we record for PettingZoo environments?

        # Begin visualization
        if max_episodes is None:
            max_episodes = np.inf

        if max_steps is None:
            max_steps = np.inf

        step_interval = 50.0 / speed

        episodes = 0
        while episodes < max_episodes:
            obs = env.reset()
            step = 0

            # Render if not in headless mode
            if not headless:
                env.render(mode="human")
                time.sleep(step_interval)

            # Build agents for this episode, to keep track of internal states
            agents = {}
            dones = {}
            for agent_id in obs.keys():
                if policy_fn is not None:
                    policy_id = policy_fn(agent_id)
                else:
                    policy_id = agent_id
            
                agents[agent_id] = policies[policy_id].make_agent()
                dones[agent_id] = False

            # Run episode
            while step < max_steps and not all(dones.values()):
                actions = {}
                for agent_id, ob in obs.items():
                    actions[agent_id], _ = agents[agent_id].act(ob)  # NOTE: May return additional info
                
                obs, rewards, dones, info = env.step(actions)
                step += 1

                # Render if not in headless mode
                if not headless:
                    env.render(mode="human")
                    time.sleep(step_interval)
