"""Do not run this script manually.

This script is run by the sbatch scripts
    
    'train_slurm_local.sh'
    'train_slurm_delft.sh' 

to launch experiments on SLURM clusters.  When run with the
'--setup' flag, this script just sets up the required directory
structure, and prints the number of individual trials to run.
"""
import argparse
import os
import torch

from interactive_agents import load_configs, run_trial, setup_experiments


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, 
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("config_files", type=str, nargs="+",
                        help="provide one or more experiment config files")
    parser.add_argument("-o", "--output-path", type=str, default="results/debug",
                        help="directory in which we should save results")
    parser.add_argument("-g", "--gpu", action="store_true",
                        help="enable GPU acceleration if available")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print data for every training iteration")
    parser.add_argument("--num-seeds", type=int,
                        help="the number of random seeds to run, overrides values from the config file")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="a list of random seeds to run, overrides values from the config file")

    parser.add_argument("--setup", action="store_true",
                        help="just setup the directory structure, run no experiments")
    
    return parser.parse_known_args()


if __name__ == '__main__':
    args, unknown = parse_args()

    # Load configuration files
    experiments = load_configs(args.config_files)

    # Override config if random seeds are provided
    for name, config in experiments.items():
        if args.num_seeds is not None:
            config["num_seeds"] = args.num_seeds

        if args.seeds is not None:
            config["seeds"] = args.seeds
            
        # Add custom arguments to config
        config["arguments"] = unknown

    # Setup experiment
    trial_configs = setup_experiments(experiments, args.output_path, use_existing=True)

    if args.setup:
        print(len(trial_configs))
    else:

        # Select torch device  # NOTE: How would this handle multi-GPU machines?
        device = "cuda" if args.gpu else "cpu"
        print(f"Training with Torch device '{device}'")

        # Limit Torch CPU parallelism
        torch.set_num_interop_threads(1)
        torch.set_num_threads(1)

        # Get target trial config from SLURM task ID variable
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)
        try:
            trial = trial_configs[int(task_id)]
        except ValueError:
            raise ValueError(f"Invalid SLURM task ID '{task_id}'")

        # Run Trial
        run_trial(trial, device=device, verbose=args.verbose)
