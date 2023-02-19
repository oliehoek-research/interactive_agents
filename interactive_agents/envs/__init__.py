from .common import SyncEnv

# TODO: At some point it might be cleaner to use importlib to do this
def get_memory():
    from interactive_agents.envs.memory import Memory
    return Memory


def get_coordination():
    from interactive_agents.envs.coordination import Coordination
    return Coordination


def get_listener():
    from interactive_agents.envs.listener import Listener
    return Listener


def get_speaker_listener():
    from interactive_agents.envs.speaker_listener import SpeakerListener
    return SpeakerListener


def get_gym_env():
    from interactive_agents.envs.gym_env import GymEnv
    return GymEnv


def get_pettingzoo_mpe():
    from interactive_agents.envs.pettingzoo_mpe import petting_zoo_mpe
    return petting_zoo_mpe


# def get_linguistic_game():
#     from interactive_agents.envs.linguistic_game import LinguisticCoordination
#     return LinguisticCoordination


# def get_repeated_game():
#     from interactive_agents.envs.repeated_game import RepeatedGame
#     return RepeatedGame


ENVS = {
    "memory": get_memory,
    "coordination": get_coordination,
    "listener": get_listener,
    "speaker_listener": get_speaker_listener,
    "gym": get_gym_env,
    "mpe": get_pettingzoo_mpe,
    # "linguistic": get_linguistic_game,
    # "repeated": get_repeated_game,
}

def get_env_class(name):
    if name not in ENVS:
        raise ValueError(f"Environment '{name}' is not defined")
    
    return ENVS[name]()
