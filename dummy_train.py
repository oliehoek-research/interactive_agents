#!/usr/bin/env python3
"""Simple script for launching experiments"""
import argparse

from interactive_agents.run import load_configs, run_experiments, run_experiments_triton


def parse_args():
    parser = argparse.ArgumentParser("Training script for finite, constant-sum extensive-form games")

    parser.add_argument("-f", "--config-file", default=None, type=str, action="append",
                        help="if specified, use config options from this file.")
    parser.add_argument("-o", "--output-path", type=str, default="results/debug",
                        help="directory in which we should save results")
    
    parser.add_argument("--variation", type=int, default=0, 
                        help="the index of the config variation to run (for hyperparameter tuning)")
    parser.add_argument("--seed", type=int, default=0, help="")

    parser.add_argument("-r", "--resources", nargs="+",
                        help="a list of key-value pairs representing file resources (policies, datasets, etc.)")
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.config_file is not None:
        experiments = load_configs(args.config_file)
    else:
        raise Exception("Must specify at least one")

    device = "cuda" if args.gpu else "cpu"
    print(f"Training with Torch device '{device}'")

    if args.triton is True:
        print("Experiments are running on Triton.")
        run_experiments_triton(experiments, args.output_path, 
            args.num_cpus, device, args.verbose, args.num_seeds, args.seeds)
    else:
        print("Experiments NOT running on triton. Use --triton if you want to run on triton!")
        run_experiments(experiments, args.output_path, 
        args.num_cpus, device, args.verbose, args.num_seeds, args.seeds, args.resources)
