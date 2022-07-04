"""
Common utilities and base-classes for our multi-agent environments
"""
from pettingzoo import ParallelEnv

class MultiagentEnv(ParallelEnv):
    """
    A simple extension of the PettingZoo parallel API that serves as the base
    class for all of our environments.

    Assumes the set of agents is fixed, and provides the 'agents' and 'possible_agents'
    properties, which just return the list of observation space keys. Also provides a 
    dummy 'visualize()' method which environments can optionally implement.
    """

    @property
    def agents(self):
        return list(self.observation_spaces.keys())

    @property
    def possible_agents(self):
        return list(self.observation_spaces.keys())  

    def visualize(self, **kwargs):
        raise NotImplementedError("Environment does not support visualization")
