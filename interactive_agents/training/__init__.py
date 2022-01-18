from interactive_agents.training.independent import IndependentTrainer
from interactive_agents.training.regret_game import RegretGameTrainer
from interactive_agents.training.self_play import SelfPlayTrainer

TRAINERS = {
    "independent": IndependentTrainer,
    "regret_game": RegretGameTrainer,
    "self_play": SelfPlayTrainer,
}

def get_trainer_class(name):
    if name not in TRAINERS:
        raise ValueError(f"Trainer '{name}' is not defined")
    
    return TRAINERS[name]
