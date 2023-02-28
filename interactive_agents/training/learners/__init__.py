# from interactive_agents.training.learners.dqn import DQN
from interactive_agents.training.learners.r2d2 import R2D2
from interactive_agents.training.learners.priority_tree import PriorityTree
from interactive_agents.training.learners.models import build_model

LEARNERS = {
    "R2D2": R2D2,
#    "DQN": DQN,
}

def get_learner_class(name):
    if name not in LEARNERS:
        raise ValueError(f"Learner '{name}' is not defined")
    
    return LEARNERS[name]


__all__ = []
