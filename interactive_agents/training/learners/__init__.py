# from interactive_agents.training.learners.dqn import DQN
from interactive_agents.training.learners.r2d2 import R2D2

LEARNERS = {
#     "DQN": DQN,
    "R2D2": R2D2,
}

def get_learner_class(name):
    if name not in LEARNERS:
        raise ValueError(f"Learner '{name}' is not defined")
    
    return LEARNERS[name]