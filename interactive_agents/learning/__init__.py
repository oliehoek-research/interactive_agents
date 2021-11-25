
def get_dqn():
    from interactive_agents.learning.dqn import DQN
    return DQN


def get_r2d2():
    from interactive_agents.learning.r2d2 import R2D2
    return R2D2


LEARNERS = {
    "DQN": get_dqn,
    "R2D2": get_r2d2,
}

def get_learner_class(name):
    if name not in LEARNERS:
        raise ValueError(f"Learner '{name}' is not defined")
    
    return LEARNERS[name]()


def get_independent():
    from interactive_agents.learning.independent import IndependentTrainer
    return IndependentTrainer


TRAINERS = {
    "independent": get_independent,
}

def get_trainer_class(name):
    if name not in TRAINERS:
        raise ValueError(f"Trainer '{name}' is not defined")
    
    return TRAINERS[name]()
