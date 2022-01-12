from interactive_agents.training.independent import IndependentTrainer

TRAINERS = {
    "independent": IndependentTrainer,
}

def get_trainer_class(name):
    if name not in TRAINERS:
        raise ValueError(f"Trainer '{name}' is not defined")
    
    return TRAINERS[name]
