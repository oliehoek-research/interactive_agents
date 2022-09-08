#!/usr/bin/env python3
import argparse

from interactive_agents.run import load_configs
from interactive_agents.grid_search import generate_config_files 

# NOTE: How does this script work?

# NOTE: Seems to use the same configuration options as the base "train.py" script
def parse_args():
    parser = argparse.ArgumentParser("Training script for finite, constant-sum extensive-form games")

    parser.add_argument("-f", "--config-file", default=None, type=str, action="append",
                        help="if specified, use config options from this file.")  # NOTE: Can change this to a single positional argment
    parser.add_argument("-o", "--output-path", type=str, default="results/debug",
                        help="directory in which we should save results")
    parser.add_argument("-n", "--num-cpus", type=int, default=2,
                        help="the number of parallel worker processes to launch")  # NOTE: I don't think we need this here
    parser.add_argument("-g", "--gpu", action="store_true",
                        help="enable GPU if available")  # NOTE: I don't think we need this
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print data for every training iteration")  # NOTE: I don't think we need this either
    parser.add_argument("--num-seeds", type=int,
                        help="the number of random seeds to run, overrides values from the config file")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="the list of random seeds to run, overrides values from the config file")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # NOTE: Seems like we could remove this if we change the parser to only accept a single config
    assert args.config_file is not None, "There must be a config file to generate experiments!"
    
    experiments = load_configs(args.config_file)

    # NOTE: Doesn't work if we upload more than one file, so probably don't need this
    assert len(experiments) == 1, "This works only with one experiment per config file! You have more!" 
    
    # NOTE: Basically just gets the name and config for the single experiment
    name, config = next(iter(experiments.items()))

    # NOTE: This is where all the work gets done
    generate_config_files(name, config)
    print("Files have been generated.")
