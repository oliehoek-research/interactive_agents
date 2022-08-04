#!/usr/bin/env python3
"""Dummy training script for testing slurm/singularity configuration"""
import argparse
import numpy as np
import os
import os.path
import pandas
import yaml


def parse_args():
    parser = argparse.ArgumentParser("Dummy training script to test Slurm/Singularity configuration")

    parser.add_argument("-f", "--config-file", default=None, type=str, action="append",
                        help="if specified, use config options from this file.")
    parser.add_argument("-o", "--output-path", type=str, default="results/debug",
                        help="directory in which we should save results")
    
    parser.add_argument("-s", "--seed", type=int, default=0, 
                        help="random seed for the current variation")

    return parser.parse_args()


def get_dir(base_path, name, use_existing=True):
    path = os.path.join(base_path, name)

    if os.path.exists(path):
        if not use_existing:
            raise Exception(f"Directory '{path}' already exists")

        if os.path.isfile(path):
            raise Exception(f"'{path}' already exists and is a file")
        
        return path    
    
    os.makedirs(path)
    return path


def run_trial(base_path, config, seed):
    path = get_dir(base_path, f"seed_{seed}", use_existing=False)  

    series = {
        "integers": pandas.Series(np.arange(10), np.zeros(10))
    }

    dataframe = pandas.DataFrame(series)
    dataframe.to_csv(os.path.join(path, "results.csv"))


if __name__ == '__main__':
    args = parse_args()

    if args.config_file is None:
        raise Exception("Must specify at least one config file")

    experiments = {}
    for path in args.config_file:
        with open(path) as f:
            experiments.update(yaml.load(f, Loader=yaml.FullLoader))

    # Don't do grid-search yet, just figure out where things are loaded from and stored to
    for name, config in experiments.items():
        path = get_dir(args.output_path, name)

        # Save config
        config_path = os.path.join(path, "config.yaml")
        if not os.path.exists(config_path):
            with open(config_path, 'w') as config_file:
                yaml.dump({name: config}, config_file)
        
        # Run trail for given seed
        run_trial(path, config, args.seed)
