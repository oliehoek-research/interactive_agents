from interactive_agents.training.independent import IndependentTrainer
from interactive_agents.training.regret_game import RegretGameTrainer

TRAINERS = {
    "independent": IndependentTrainer,
    "regret_game": RegretGameTrainer,
}

def get_trainer_class(name):
    if name not in TRAINERS:
        raise ValueError(f"Trainer '{name}' is not defined")
    
    return TRAINERS[name]
