import importlib
import numpy as np

from supersuit import dtype_v0

# TODO: We will eventually want to provide visualization support for the MPE, but not needed yet
def petting_zoo_mpe(env_config, spec_only=False):
    assert "scenario" in env_config, "Must specify 'scenario' for particle env"
    env_config = env_config.copy()
    env_name = env_config.pop("scenario")

    # Load appropriate PettingZoo class
    env_module = importlib.import_module("pettingzoo.mpe." + env_name)

    # Build PettingZoo environment
    env = env_module.parallel_env(**env_config)  # NOTE: The parallel env API is closer to the gym api than PettingZoo's AEC API
    return dtype_v0(env, dtype=np.float32)  # NOTE: Make sure the obs tensors are the right type
