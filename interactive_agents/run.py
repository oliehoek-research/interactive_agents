from collections import defaultdict
import numpy as np
import os
import os.path
import pandas
from tensorboardX import SummaryWriter
import torch

from interactive_agents.experiments import save_metadata
from interactive_agents.training import get_trainer_class

# Parses the run config to determine when to stop training
def get_stop_conditions(stop):
    max_iterations = stop.pop("iterations", np.infty)
    
    # Flatten termination conditions
    def flatten(d):
        flattened = {}
        for key, value in d.items():
            if isinstance(value, dict):
                value = flatten(value)

                for sub_key, sub_value in value.item():
                    flattened[key + "/" + sub_key] = sub_value
            else:
                flattened[key] = value
        
        return flattened
    
    termination = flatten(stop)

    # Add top-level keys to the "global" namespace
    for key, value in stop.items():
        if not isinstance(value, (dict, list)):
            termination["global/" + key] = value

    return max_iterations, termination


# TODO: Add validation to ensure all seeds within an experiment have same configuration
def run_trial(trial, device='cpu', verbose=False):
    print(f"running: {trial.name} - seed {trial.seed}")

    # Make output directory for trial
    path = os.path.join(trial.path, f"seed_{trial.seed}")
    assert not os.path.exists(path), f"found existing path '{path}', aborting trial"
    os.makedirs(path)

    # Save metadata
    save_metadata(path)

    # Extract termination conditions
    stop = trial.config.pop("stop", {})
    max_iterations, stop = get_stop_conditions(stop)

    # Build trainer
    trainer_cls = get_trainer_class(trial.config.get("trainer", "independent"))
    trainer = trainer_cls(trial.config.get("config", {}),
        seed=trial.seed, device=device, verbose=verbose)


    # Run trainer with TensorboardX logging
    stat_values = defaultdict(list)
    stat_indices = defaultdict(list)
    iteration = 0
    complete = False
    
    with SummaryWriter(path) as writer:
        while not complete:
            stats = trainer.train()

            # Write statistics to tensorboard and append to data series
            for key, value in stats.items():
                writer.add_scalar(key, value, iteration)
                stat_values[key].append(value)
                stat_indices[key].append(iteration)

            # Check termination conditions
            iteration += 1
            if iteration >= max_iterations:
                complete = True

            for key, value in stop.items():
                if key in stats and stats[key] >= value:
                    complete = True

    # Build and save data frame
    series = {}
    for key, values in stat_values.items():
        series[key] = pandas.Series(np.asarray(values), np.asarray(stat_indices[key]))

    dataframe = pandas.DataFrame(series)
    dataframe.to_csv(os.path.join(path, "results.csv"))
    
    # Export policies
    path = os.path.join(path, "policies")
    os.makedirs(path)

    # NOTE: Should we let the trainer itself handle serialization?
    policies = trainer.export_policies()
    for id, policy in policies.items():
        torch.jit.save(policy, os.path.join(path, f"{id}.pt"))
