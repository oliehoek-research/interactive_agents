import importlib
import numpy as np

from supersuit import dtype_v0

# NOTE: PettingZoo interface has changed significantly, may want to use it as our base interface in the future
class PettingZooMPE:

    def __init__(self, env_config, spec_only=False):
        env_config = env_config.copy()
        env_name = env_config.pop("scenario")

        # Load appropriate PettingZoo class
        env_module = importlib.import_module("pettingzoo.mpe." + env_name)

        # Build PettingZoo environment
        env = env_module.parallel_env(**env_config)  # NOTE: The parallel env API is closer to the gym api that PettingZoo's default API
        self.env = dtype_v0(env, dtype=np.float32)  # NOTE: Not sure if this typecasting is still necessary

    @property
    def observation_space(self):
        return self.env.observation_spaces # TODO: Double-check that these are actually dictionaries

    @property
    def action_space(self):
        return self.env.action_spaces

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        return self.env.step(actions)

    def render(self, mode="human"):
        return self.render(mode)
