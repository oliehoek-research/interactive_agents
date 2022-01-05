from interactive_agents.learning.dqn import DQN
from interactive_agents.learning.r2d2 import R2D2
from interactive_agents.learning.independent import IndependentTrainer

LEARNERS = {
    "DQN": DQN,
    "R2D2": R2D2,
}

def get_learner_class(name):
    if name not in LEARNERS:
        raise ValueError(f"Learner '{name}' is not defined")
    
    return LEARNERS[name]


TRAINERS = {
    "independent": IndependentTrainer,
}

def get_trainer_class(name):
    if name not in TRAINERS:
        raise ValueError(f"Trainer '{name}' is not defined")
    
    return TRAINERS[name]
