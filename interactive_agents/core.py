from collections import defaultdict
from datetime import datetime
from git import Repo
from multiprocessing import Pool
import numpy as np
import os
import os.path
import pandas
import random
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


def run_trail(path, trainer_cls, config, stop, seed):
    path = os.path.join(path, f"seed_{seed}")  
    os.makedirs(path)

    # Reseed python, numpy, and pytorch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Initialize trainer
    trainer = trainer_cls(config)
    
    # Run trainer
    results = defaultdict(list)
    complete = False
    
    while not complete:
        statistics = trainer.train()

        for key, value in statistics.items():
            results[key].append(value)

        for key, value in stop.items():
            if statistics[key] >= value:
                complete = True

    # Build and save data frame
    results_file = os.path.join(path, "results.csv") 

    dataframe = pandas.DataFrame(results)
    dataframe.to_csv(results_file)
    
    # Export policies
    path = os.path.join(path, "policies")
    os.makedirs(path)

    policies = trainer.export_policies()
    for id, policy in policies.items():
        torch.jit.save(policy, os.path.join(path, f"{id}.pt"))


def run_experiment(path, name, config, pool):

    # Make new results directory
    path = make_unique_dir(path, name)

    # Save experiment configuration
    with open(os.path.join(path, "config.yaml"), 'w') as config_file:
        yaml.dump({name: config}, config_file)

    # Save experiment metadata
    metadata = {}

    d = datetime.utcnow()
    metadata["timestamp"] = d.isoformat()

    try:
        repo = Repo(search_parent_directories=True)
        metadata["git_commit"] = str(repo.active_branch.commit)
    except:
        pass

    with open(os.path.join(path, "metadata.yaml"), 'w') as metadata_file:
        yaml.dump(metadata, metadata_file)

    # Get trainer class and config
    trainer_cls = get_trainer_class(config.get("trainer", "independent"))
    trainer_config = config.get("config", {})

    # Get termination conditions
    stop = config.get("stop", {})

    # Launch trials
    trials = []
    for seed in range(config.get("num_seeds", 1)):
        print(f"{name} - seed: {seed}")
        trials.append(pool.apply_async(run_trail, 
            (path, trainer_cls, trainer_config, stop, seed), error_callback=print_error))
    
    return trials


def run_experiments(experiments, path, num_cpus=1):  # NOTE: Right now we don't even support GPU training explicitly - codebase is a mess

    # Limit CPU paralellism
    torch.set_num_threads(num_cpus)

    pool = Pool(num_cpus)
    trials = []

    for name, experiment in experiments.items():
        variations = grid_search(name, experiment)

        if variations is None:
            trials += run_experiment(path, name, experiment, pool)
        else:
            exp_path = make_unique_dir(path, name)

            with open(os.path.join(exp_path, "config.yaml"), 'w') as config_file:
                yaml.dump({name: experiment}, config_file)  # NOTE: Save base tuning configuration

            for var_name, var_experiment in variations.items():
                trials += run_experiment(exp_path, var_name, var_experiment, pool)

    for trial in trials:
        trial.wait()


def load_configs(config_files):
    experiments = {}

    for path in config_files:
        with open(path) as f:
            experiments.update(yaml.load(f, Loader=yaml.FullLoader))
    
    return experiments
