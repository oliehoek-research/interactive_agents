"""Helper methods for data logging."""
from datetime import datetime
from git import Repo
import os.path  
import yaml

# TODO: Combine all of these different methods into a single, clean interface

def make_experiment_dir(base_path, name, config, use_existing=False):
    path = os.path.join(base_path, name)

    if os.path.isfile(path):
        raise FileExistsError(f"'{path}' exists but is not a directory")

    if not use_existing:
        idx = 0
        while os.path.exists(path):
            idx += 1
            path = os.path.join(base_path, name + "_" + str(idx))
    
    if not os.path.exists(path):
        os.makedirs(path)

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

    return path


def load_configs(config_files):  # TODO: Add basic config validation
    """
    Simple utility for loading a list of config files and combining them into
    a single dictionary. Assumes each file provides a unique experiment name.
    """
    experiments = {}

    for path in config_files:
        with open(path) as f:
            experiments.update(yaml.load(f, Loader=yaml.FullLoader))
    
    return experiments
