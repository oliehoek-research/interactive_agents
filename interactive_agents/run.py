from collections import defaultdict
from datetime import datetime
from multiprocessing.sharedctypes import Value
from git import Repo
from multiprocessing import Pool
import numpy as np
import os
import os.path
import pandas
from tensorboardX import SummaryWriter
import torch
import traceback
import yaml

from interactive_agents.grid_search import grid_search
from interactive_agents.training import get_trainer_class

def make_unique_dir(path, tag):
    sub_path = os.path.join(path, tag)
    idx = 0

    while os.path.exists(sub_path):
        idx += 1
        sub_path = os.path.join(path, tag + "_" + str(idx))
    
    os.makedirs(sub_path)
    return sub_path


def print_error(error):
    traceback.print_exception(type(error), error, error.__traceback__, limit=5)


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


def run_trail(base_path, config, seed, device, verbose):
    path = os.path.join(base_path, f"seed_{seed}")  
    os.makedirs(path)

    # Extract termination conditions
    stop = config.pop("stop", {})
    max_iterations, stop = get_stop_conditions(stop)

    # Build trainer
    trainer_cls = get_trainer_class(config.get("trainer", "independent"))
    trainer = trainer_cls(config.get("config", {}),
        seed=seed, device=device, verbose=verbose)
    
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

    policies = trainer.export_policies()  # NOTE: Should we let the trainer itself handle these processes?
    for id, policy in policies.items():
        torch.jit.save(policy, os.path.join(path, f"{id}.pt"))


def launch_experiment(path, name, config, pool, device, verbose):
    path = make_unique_dir(path, name)

    # Save experiment configuration
    with open(os.path.join(path, "config.yaml"), 'w') as config_file:
        yaml.dump({name: config}, config_file)

    # Save experiment metadata
    metadata = {
        "timestamp": datetime.utcnow().isoformat()
    }

    try:
        repo = Repo(search_parent_directories=True)
        metadata["git_commit"] = str(repo.active_branch.commit)
    except:
        print("NOTICE: Could not determine current git commit")

    with open(os.path.join(path, "metadata.yaml"), 'w') as metadata_file:
        yaml.dump(metadata, metadata_file)

    # Get random seeds
    num_seeds = config.get("num_seeds", 1)
    seeds = config.get("seeds", list(range(num_seeds)))

    # Launch trials
    trials = []
    for seed in seeds:
        print(f"launching: {name} - seed: {seed}")
        trials.append(pool.apply_async(run_trail, 
            (path, config, seed, device, verbose), error_callback=print_error))
    
    return trials


def run_experiments(experiments, base_path, num_cpus=1, device="cpu", verbose=False):

    # Limit CPU paralellism globally
    torch.set_num_threads(num_cpus)

    # Uses the built-in multiprocessing pool to schedule experiments
    pool = Pool(num_cpus)

    # Generate hyperparameter variations and queue all trials
    trials = []

    for name, config in experiments.items():
        variations = grid_search(name, config)

        if variations is None:
            trials += launch_experiment(base_path, name, config, pool, device, verbose)
        else:
            exp_path = make_unique_dir(base_path, name)

            # Save base tuning configuration for reference
            with open(os.path.join(exp_path, "config.yaml"), 'w') as config_file:
                yaml.dump({name: config}, config_file)

            for var_name, var_config in variations.items():
                trials += launch_experiment(exp_path, var_name, var_config, pool, device, verbose)

    # Wait for trails to complete before returning
    for trial in trials:
        trial.wait()


def load_configs(config_files):
    """
    Simple utility for loading a list of config files and combining them into
    a single dictionary. Assumes each file provides a unique experiment name.
    """
    experiments = {}

    for path in config_files:
        with open(path) as f:
            experiments.update(yaml.load(f, Loader=yaml.FullLoader))
    
    return experiments
