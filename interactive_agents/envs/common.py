"""
Common utilities and base-classes for our multi-agent environments
"""
from pettingzoo import ParallelEnv

class MultiagentEnv(ParallelEnv):
    """
    A simple extension of the PettingZoo parallel API that serves as the base
    class for all of our environments.

    Assumes the set of agents is fixed, and provides the 'agents' property that
    returns a list of agent IDs.  Also provides a dummy 'visualize()' method 
    which environments can optionally implement.
    """

    def agents(self):
        return list(self.observation_space.keys())
    
    def visualize(self, **kwargs):
        raise NotImplementedError("Environment does not support visualization")
