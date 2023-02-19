from copy import deepcopy
from datetime import datetime
from git import Repo
import os
import os.path
import yaml

from interactive_agents.grid_search import grid_search

class Trial:

    def __init__(self, path, name, config, seed):
        self.path = path  # NOTE: When we run a trial, do we always create a new seed path?
        self.name = name  # NOTE: What do we use the name for if we already have the path?
        self.config = config
        self.seed = seed


def get_directory(base_path, name, use_existing=False, index_digits=3):
    path = os.path.join(base_path, name)
    
    idx = 0
    while not use_existing and os.path.exists(path):
        idx += 1
        path = os.path.join(base_path, name + "_" + str(idx).zfill(index_digits))
    
    os.makedirs(path, exist_ok=True)
    return path

# NOTE: Saves the following meta-data in a .yaml file:
# - timestamp at which the experiment folder was created (or recreated)
# - current git commit when the experiment folder was created (or recreated)
#
# This meta-data may be overwritten if we launch new seeds in the same directory
def save_metadata(path, use_existing=False):
    metadata = {"timestamp": datetime.utcnow().isoformat()}

    try:
        repo = Repo(search_parent_directories=True)
        metadata["git_commit"] = str(repo.active_branch.commit)
    except:
        print("NOTICE: Could not determine current git commit")

    metadata_path = os.path.join(path, "metadata.yaml")
    if not os.path.exists(metadata_path) or not use_existing:
        with open(metadata_path, 'w') as metadata_file:
            yaml.dump(metadata, metadata_file)


def save_config(path, name, config, use_existing=False):
    config_path = os.path.join(path, "config.yaml")
    if not os.path.isfile(config_path) or not use_existing:
        with open(config_path, 'w') as config_file:
            yaml.dump({name: config}, config_file)


def load_configs(config_files):
    """
    Simple utility for loading a list of config files and combining them into
    a single dictionary. Assumes each file provides a unique experiment name.
    """
    if isinstance(config_files, str):
        config_files = [config_files]

    experiments = {}
    for path in config_files:
        with open(path) as f:
            experiments.update(yaml.load(f, Loader=yaml.FullLoader))
    
    return experiments

# NOTE: Where is this used?
# TODO: We may eventually want to be able to continue an individual trial, rather than restarting it
def load_trial(path):
    with open(os.path.join(path, "config.yaml")) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Get experiment name and config dictionary
    name, config = next(iter(config.items()))

    # Get the random seed for this trial
    seed = config["seeds"][0]

    return Trial(path, name, config, seed)


def setup_trial(output_path, name, config, seed):
    config = deepcopy(config)

    # Set a single seed for trial configuration
    del config["num_seeds"]
    config["seeds"] = [seed]

    # Create directory for this trial
    path = os.path.join(output_path, f"seed_{seed}")
    assert not os.path.exists(path), f"found existing path '{path}', aborting trial"
    os.makedirs(path)

    # Save the configuration for this trial
    save_config(path, name, config, use_existing=False)

    return Trial(path, name, config, seed)


# NOTE: If "use_existing=True", then we will not create a new directory for this 
# experiment if one already exists, though we may create directories for new seeds.
def setup_experiment(output_path, name, config, use_existing):
    path = get_directory(output_path, name, use_existing)

    # Save experiment configuration
    save_config(path, name, config, use_existing)  # NOTE: If "use_existing=True", this may overwrite an existing config (without checking that it is the same)

    # Save experiment-wide metadata
    save_metadata(path, use_existing)  # NOTE: What metadata gets saved?

    # NOTE: This gets overridden by the command-line options
    # Get random seeds
    num_seeds = config.get("num_seeds", 1)
    seeds = config.get("seeds", list(range(num_seeds)))

    # Construct individual trials
    return [setup_trial(path, name, config, seed) for seed in seeds]


# NOTE: This is the main method for configuring experiment
def setup_experiments(experiments, output_path, use_existing=False):  # NOTE: What does "use_existing" do?
    trials = []

    for name, config in experiments.items():  # NOTE: Accepts multiple config files 
        variations = grid_search(name, config)  # NOTE: Check for "grid_search" keys for hyperparameter sweeps
    
        if variations is None:  # NOTE: Only running a sinlge configuration
            trials += setup_experiment(output_path, name, config, use_existing)  # NOTE: Sets-up and individual experiment for multiple seeds
        else:  # NOTE: Hyperparameter sweep
            path = get_directory(output_path, name, use_existing)  # NOTE: What does this do?

            for var_name, var_config in variations.items():  # NOTE: Iterate over hyperparameter configurations
                trials += setup_experiment(path, var_name, var_config, use_existing=True)

    return trials
