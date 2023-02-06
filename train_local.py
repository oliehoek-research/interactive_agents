# TODO: Check that this still runs
"""Use this script to launch experiments locally on your own machine.

For a complete configuration file (with no "grid_search" keys), this
script launches a separate process for each random seed specified with
the "--num-seeds" or "--seeds" arguments, or the seeds specified in the
config file itself if these arguments are not provided.

For hyperparameter tuning configs (with "grid_search" keys), this script
launches a separate process for each possible configuration in the grid
search, and each random seed.

For trainers that accept additional command-line arguments, this script
will pass any unrecognized arguments to each trainer instance.
"""
import argparse
import traceback

import torch
from torch.multiprocessing import Pool

from interactive_agents import load_configs, run_trial, setup_experiments

def print_error(error):
    traceback.print_exception(type(error), error, error.__traceback__, limit=5)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, 
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("config_files", type=str, nargs="+",
                        help="provide one or more experiment config files")
    parser.add_argument("-o", "--output-path", type=str, default="results/debug",
                        help="directory in which we should save results")

    parser.add_argument("--num-seeds", type=int,
                        help="the number of random seeds to run, overrides values from the config file")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="a list of random seeds to run, overrides values from the config file")

    parser.add_argument("-n", "--num-cpus", type=int, default=1,
                        help="the number of parallel experiments to launch")
    parser.add_argument("-g", "--gpu", action="store_true",
                        help="enable GPU acceleration if available")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print data for every training iteration")
    parser.add_argument("--flush-secs", type=int, default=200,
                        help="number of seconds after which we should flush the training longs (default 200)")

    return parser.parse_known_args()


if __name__ == '__main__':
    args, unknown = parse_args()

    # Load configuration files
    experiments = load_configs(args.config_files)

    # Select torch device  # NOTE: How would this handle multi-GPU machines?
    device = "cuda" if args.gpu else "cpu"
    print(f"Training with Torch device '{device}'")

    # Override config if random seeds are provided
    for name, config in experiments.items():
        if args.num_seeds is not None:
            config["num_seeds"] = args.num_seeds

        if args.seeds is not None:
            config["seeds"] = args.seeds
            
        # Add custom arguments to config
        config["arguments"] = unknown

    # Setup experiment
    trial_configs = setup_experiments(experiments, args.output_path, use_existing=False)

    # Limit Torch CPU parallelism  # NOTE: This must be set BEFORE we initialize the process pool
    torch.set_num_interop_threads(1)
    torch.set_num_threads(1)

    # Launch experiments
    with Pool(args.num_cpus) as pool:
        trials = []
        for trial in trial_configs:
            trials.append(pool.apply_async(run_trial, (trial,), {
                    "device": device, 
                    "verbose": args.verbose, 
                    "flush_secs": args.flush_secs
                }, error_callback=print_error))
            
        # Wait for all trails to complete before returning
        for trial in trials:
            trial.wait()
