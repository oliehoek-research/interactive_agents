"""Additional utilities for multi-agent environments"""
import gym
from gym.wrappers import RecordVideo, TimeLimit
import numpy as np
import time

class VisualizeGym(gym.core.Wrapper):
    
    def __init__(self, env):
        super(VisualizeGym, self).__init__(env)
    
    def visualize(self, 
                  policies={},
                  policy_fn=None,
                  max_episodes=None, 
                  max_steps=None,
                  speed=50,
                  record_path=None,
                  headless=False,
                  **kwargs):

        # NOTE: We will have multiple wrappers around the base environment (performance isn't an issue for visualization)
        env = self.env

        # Enforce step limit
        if max_steps is not None:
            env =TimeLimit(env, max_episode_steps=max_steps)

        # Enable video recording if requested
        if record_path is not None:

            # NOTE: Need to provide the episode_trigger to record every episode
            env = RecordVideo(env, record_path, episode_trigger=lambda id: True)

        # Begin visualization
        if max_episodes is None:
            max_episodes = np.inf

        if max_steps is None:
            max_steps = np.inf

        step_interval = 1.0 / speed

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
