import importlib
import numpy as np

from supersuit import dtype_v0

def petting_zoo_mpe(config={}):
    """
    Imports and returns an instance of the appropriate PettingZoo MPE environment. For
    right now this ignores the random generator, as the PettingZoo MPE provides no way
    to directly set the seed (use `np.random.seed()`).
    """
    assert "scenario" in config, "Must specify 'scenario' for particle env"
    config = config.copy()
    env_name = config.pop("scenario")

    # Load appropriate PettingZoo class
    env_module = importlib.import_module("pettingzoo.mpe." + env_name)

    # Build PettingZoo environment
    env = env_module.parallel_env(**config)  # NOTE: The parallel env API is closer to the gym api than PettingZoo's AEC API
    return dtype_v0(env, dtype=np.float32)  # NOTE: Make sure the obs tensors are the right type
