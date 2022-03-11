import gym

class GymEnv(gym.Wrapper):
    '''Multi-agent wrapper for single-agent OpenAI Gym environments'''

    def __init__(self, config, spec_only=False):
        assert "name" in config, "must specify name of gym environment"
        super(GymEnv, self).__init__(gym.make(config.get("name")))

        self._agent_id = config.get("agent_id", 0)

        self.observation_space = {self._agent_id: self.env.observation_space}
        self.action_space = {self._agent_id: self.env.action_space}
    
    def reset(self):
        obs = self.env.reset()
        return {self._agent_id: obs}
    
    def step(self, action):
        obs, rew, done, info = self.env.step(action[self._agent_id])
        obs = {self._agent_id: obs}
        rew = {self._agent_id: rew}
        done = {self._agent_id: done}
        return obs, rew, done, info
