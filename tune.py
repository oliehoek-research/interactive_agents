"""Selects the best configuration(s) from a hyperparameter sweep.

Given a directory containing multiple experiments with different configurations,
this script selects the configuration (or configurations if there is a tie) that
maximize (or minimize) the given statistic.

This script can optimize based on any statistic that is collected during training.
As these statistics are recorded as a time series over training iterations, the 
'--accumulate' option allows us to specify how a sequence of statistics should be
reduced to a single scalar (default is to find the maximum value).
"""
import argparse
import json
import numpy as np
import os
import os.path
import pandas
import yaml

class Configuration:

    def __init__(self, trainer, config):
        self.trainer = trainer
        self.config = config
        self.runs = []


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, 
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument("path", type=str, help="path to directory containing training results")
    parser.add_argument("-l", "--loss", type=str, default="eval/reward_mean", 
        help="key of the metric to minimize (or maximize)")
    parser.add_argument("-a", "--accumulate", type=str, default="max", 
        help="method for reducing the time series into a scalar ['mean','max','min']")
    parser.add_argument("-m", "--mode", type=str, default="max",
        help="whether to maximize or minimize the given key ['max','min']")
    parser.add_argument("-v", "--verbose", action="store_true",
        help="print results for every configuration")
    
    return parser.parse_args()


def load_runs(path, loss, accumulate):
    print(f"loading: {path}")
    runs = []

    if os.path.isdir(path):
        for obj in os.listdir(path):
            results_path = os.path.join(path, obj)

            if os.path.isdir(results_path):
                results_file = os.path.join(results_path, "results.csv")

                if os.path.isfile(results_file):

                    # Load final results from CSV file (faster than tensorboard logs)
                    results = pandas.read_csv(results_file)

                    # Filter out empy data series
                    if len(results.index) > 0:
                        result = results[loss]

                        if "max" == accumulate:
                            value = np.nanmax(result)
                        elif "max" == accumulate:
                            value = np.nanmin(result)
                        else:
                            value = np.nanmean(result)

                        runs.append(value)  # NOTE: Is there a risk that we store a reference to the full dataframe in this scalar value?

    return runs


if __name__ == "__main__":
    args = parse_args()

    # Check that data path exists
    print(f"Path: {args.path}")
    assert os.path.isdir(args.path), f"Data path '{args.path}' does not exist or is not a directory"

    # Load all training runs for all configurations
    print("Loading runs...")

    configs = dict()
    for obj in os.listdir(args.path):
        experiment_path = os.path.join(args.path, obj)

        if os.path.isdir(experiment_path):
            config_path = os.path.join(experiment_path, "config.yaml")

            if os.path.isfile(config_path):
                with open(config_path, 'r') as config_file:
                    config = yaml.load(config_file, Loader=yaml.FullLoader)
                
                if "trainer" not in config:
                    config = list(config.values)[0]

                trainer = config["trainer"]
                trainer_config = config["config"]
                
                runs = load_runs(experiment_path, args.loss, args.accumulate)
                config_str = json.dumps({
                    "trainer": trainer,
                    "config": trainer_config
                }, sort_keys=True)

                # NOTE: If multiple experiment directories contain the same configuration, they will be combined
                if config_str not in configs:
                    configs[config_str] = Configuration(trainer, trainer_config)

                configs[config_str].runs.extend(runs)

    # Identify best configuration(s)
    if "min" == args.mode:
        best_mean = np.Infinity
    else:
        best_mean = -np.Infinity
    
    best_configs = []

    for config in configs.values():
        if len(config.runs) > 0:  # NOTE: Ignores configs for which no data was available
            mean = np.mean(config.runs)

            if mean == best_mean:
                best_configs.append(config)  # NOTE: If two configurations are both optimal, we will return both 
            elif "min" == args.mode:
                if mean < best_mean:
                    best_mean = mean
                    best_configs = [config]
            else:
                if mean > best_mean:
                    best_mean = mean
                    best_configs = [config]
            
            if args.verbose:
                print("\n------------")
                print(f"Mean: {mean}")
                print(f"Trainer: {config.trainer}")
                print("Config:")
                print(yaml.dump(config.config, default_flow_style=False))
    
    # Return best config
    print(f"\nBest Value: {best_mean}")
    print("Best Configs:")

    for config in best_configs:
        print("\n----------\n")
        print(f"Trainer: {config.trainer}")
        print("Config:")
        print(yaml.dump(config.config, default_flow_style=False))
