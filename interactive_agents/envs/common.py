from pettingzoo import ParallelEnv

class SyncEnv(ParallelEnv):
    """
    A convenience class that makes it easier to implement the
    PettingZoo ParallelEnv interface.  Assumes that the set of
    agents is always the same for each episode and time step.
    """

    @property
    def agents(self):
        return list(self.observation_spaces.keys())

    @property
    def possible_agents(self):  # NOTE: Will the ParallelEnv class do this for us?
        return list(self.observation_spaces.keys())  
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
