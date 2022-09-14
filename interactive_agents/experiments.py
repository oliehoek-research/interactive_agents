from datetime import datetime
from git import Repo
import os
import os.path
import yaml

from interactive_agents.grid_search import grid_search

class Trial:

    def __init__(self, path, name, config, seed):
        self.path = path
        self.name = name
        self.config = config
        self.seed = seed


def get_directory(base_path, name, use_existing=False):
    path = os.path.join(base_path, name)
    
    idx = 0
    while not use_existing and os.path.exists(path):
        idx += 1
        path = os.path.join(base_path, name + "_" + str(idx))
    
    os.makedirs(path, exist_ok=True)
    return path


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


# NOTE: Sets up an individual experiments (with a given set of random seeds)
def setup_experiment(output_path, name, config, use_existing):
    path = get_directory(output_path, name, use_existing)

    # Save experiment configuration
    save_config(path, name, config, use_existing)

    # Save experiment-wide metadata
    save_metadata(path, use_existing)

    # Get random seeds
    num_seeds = config.get("num_seeds", 1)
    seeds = config.get("seeds", list(range(num_seeds)))

    # Construct trials
    return [Trial(path, name, config, seed) for seed in seeds]


def setup_experiments(experiments, output_path, use_existing=False):
    trials = []

    for name, config in experiments.items():
        variations = grid_search(name, config)
    
        if variations is None:
            trials += setup_experiment(output_path, name, config, use_existing)
        else:
            path = get_directory(output_path, name, use_existing)

            for var_name, var_config in variations.items():
                trials += setup_experiment(path, var_name, var_config, use_existing=True)

    return trials
