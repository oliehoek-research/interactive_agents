"""Do not run this script manually.

This script is intended to be called by the "train_slurm.py" script 
to launch experiments on SLURM clusters.  This script parses a set
of experiment configuration files, and sets up the corresponding
directory structure.  It then prints the resulting trial directories
to the command line, one path per line.
"""
import argparse

from interactive_agents import load_configs, setup_experiments

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, 
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # NOTE: How should we change the command line arguments?
    parser.add_argument("config_files", type=str, nargs="+",
                        help="provide one or more experiment config files")
    parser.add_argument("-o", "--output-path", type=str, default="./results/debug",
                        help="directory in which we should save results")  # NOTE: Potential issue, we may end up with relative, rather than absolute paths
    
    parser.add_argument("--num-seeds", type=int,
                        help="the number of random seeds to run, overrides values from the config file")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="a list of random seeds to run, overrides values from the config file")
    
    return parser.parse_known_args()


if __name__ == '__main__':
    args, unknown = parse_args()

    # Load configuration files
    experiments = load_configs(args.config_files)

    # Override config if random seeds are provided
    for config in experiments.values():
        if args.num_seeds is not None:
            config["num_seeds"] = args.num_seeds

        if args.seeds is not None:
            config["seeds"] = args.seeds
            
        # Add custom arguments to config
        config["arguments"] = unknown

    # Setup experiment  # NOTE: On SLURM, default to 
    trials = setup_experiments(experiments, args.output_path, use_existing=True)

    # Print trial paths
    for trial in trials:
        print(trial.path)
