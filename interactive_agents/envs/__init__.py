from .common import MultiagentEnv

def get_coordination_game():
    from interactive_agents.envs.coordination import CoordinationGame
    return CoordinationGame


def get_linguistic_game():
    from interactive_agents.envs.linguistic_game import LinguisticGame
    return LinguisticGame


def get_gym_env():
    from interactive_agents.envs.gym_env import GymEnv
    return GymEnv


def get_memory_game():
    from interactive_agents.envs.memory_game import MemoryGame
    return MemoryGame


def get_repeated_game():
    from interactive_agents.envs.repeated_game import RepeatedGame
    return RepeatedGame


def get_pettingzoo_mpe():
    from interactive_agents.envs.pettingzoo_mpe import petting_zoo_mpe
    return petting_zoo_mpe


ENVS = {
    "coordination": get_coordination_game,
    "linguistic": get_linguistic_game,
    "gym": get_gym_env,
    "memory": get_memory_game,
    "repeated": get_repeated_game,
    "mpe": get_pettingzoo_mpe,
}

def get_env_class(name):
    if name not in ENVS:
        raise ValueError(f"Environment '{name}' is not defined")
    
    return ENVS[name]()
