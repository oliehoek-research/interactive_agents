#!/usr/bin/env python3
"""Script to select the best configuration from a hyperparameter sweep"""
import argparse
import json
import numpy as np
import os
import os.path
import pandas
import yaml

from interactive_agents.grid_search import grid_search

# TODO: Examine how this script actually works so we don't break it

# NOTE: This struct seems to be used to store all the seeds associated with a specific config
class Configuration:

    def __init__(self, params):
        self.params = params
        self.runs = []


# TODO: This script doesn't support some types of tuning, like maximizing performance against a fixed self-play population
def parse_args():
    parser = argparse.ArgumentParser("Identifies the best hyperparameters settings from a tuning sweep")
    
    # NOTE: Assumes all tuning runs are stored in the the same top-level directory (Mert's code does not create this automatically)
    parser.add_argument("path", type=str, help="path to directory containing training results")

    # NOTE: Currently, we can only tuning on statistics collected during training
    parser.add_argument("-l", "--loss", type=str, default="eval/reward_mean", 
        help="key of the metric to minimize (or maximize)")

    # NOTE: Because these statistics are time-series, need to specify how they will be accumulated into a scalar value
    parser.add_argument("-a", "--accumulate", type=str, default="mean", 
        help="method for condensing time series into a scalar ['mean','max','min']")
    
    # NOTE: Some statistics are losses that we would want to minimize rather than maximize
    parser.add_argument("-m", "--mode", type=str, default="max",
        help="whether to maximize or minimize the given key ['max','min']")
    
    return parser.parse_args()


def load_runs(path, loss, accumulate):
    print(f"loading: {path}")
    runs = []  # NOTE: We iterate through seed directories and process runs one at a time

    # NOTE: We never check that the folder actually contains an experiment configuration .yaml file (this may be an artifact from the MSR code)
    if os.path.isdir(path):  # NOTE: Here "path" is going to be the path to the folder containing results for a specific configuration
        for obj in os.listdir(path):  # NOTE: Lists directories, which should represent different random seeds (we never actually check what these seeds are)
            results_path = os.path.join(path, obj)

            if os.path.isdir(results_path):
                results_file = os.path.join(results_path, "results.csv")  # NOTE: Checks that the results file actually exists

                if os.path.isfile(results_file):
                    results = pandas.read_csv(results_file)  # NOTE: The previous dataframe is dereferenced when a new one is loaded, so memory shouldn't be an issue

                    # Filter out empy data series
                    if len(results.index) > 0:  # NOTE: Ignores files that contain no data (failed runs)
                        result = results[loss]

                        if "max" == accumulate:
                            value = np.nanmax(result)
                        elif "max" == accumulate:
                            value = np.nanmin(result)
                        else:
                            value = np.nanmean(result)

                        runs.append(value)  # NOTE: Is there a risk that we store a reference to the full dataframe in this scalar value?

    return runs


def main(args):

    # NOTE: Just lets us know that we have the right data path set
    print(f"Path: {args.path}")
    print("Loading runs...")

    # NOTE: This script seems to require that the original grid-search config is itself stored with the tuning data (can't just load separate experiments)
    # Load variations
    with open(os.path.join(args.path, "config.yaml"), 'r') as config_file:
        experiments = yaml.load(config_file, Loader=yaml.FullLoader)

    variations = {}

    # NOTE: We rerun the grid search, which seems unnecessary, since all the individual configurations should have been stored
    for name, config in experiments.items():
        variations.update(grid_search(name, config))

    # Collect all runs for each config
    configs = dict()

    # NOTE: We accumulate data based on the "predicted" configurations, rather than what is actually available
    for name, config in variations.items():
        runs = load_runs(os.path.join(args.path, name), args.loss, args.accumulate)  # NOTE: This returns an empty list if the runs aren't defined
        config_str = json.dumps({
            "trainer": config["trainer"],
            "config": config["config"]
        }, sort_keys=True)

        # NOTE: Interestingly, if we have two identical configurations in two different folders, we will combine their runs, rather than treat them separately (not clear what scenario this was for)
        if config_str not in configs:
            configs[config_str] = Configuration(config["config"])

        configs[config_str].runs.extend(runs)

    # Identify best configuration
    if "min" == args.mode:
        best_mean = np.Infinity
    else:
        best_mean = -np.Infinity
    
    best_configs = []

    for config in configs.values():
        if len(config.runs) > 0:  # NOTE: Ignores configs for which no data was available
            mean = np.mean(config.runs)

            print("\n------------")  # NOTE: Prints results for every config, probably don't need this
            print(f"Mean: {mean}")
            print("Config:")
            print(yaml.dump(config.params, default_flow_style=False))

            if mean == best_mean:
                best_configs.append(config.params)  # NOTE: If two configurations are both optimal, we will return both 
            elif "min" == args.mode:
                if mean < best_mean:
                    best_mean = mean
                    best_configs = [config.params]
            else:
                if mean > best_mean:
                    best_mean = mean
                    best_configs = [config.params]
    
    # Return best config
    print(f"\nBest Value: {best_mean}")
    print("Best Configs:")

    for config in best_configs:
        print("\n----------\n")
        print(yaml.dump(config, default_flow_style=False))


if __name__ == "__main__":
    args = parse_args()
    main(args)
