# NOTE: This is a pretty clumsy setup, has to be a more elegant/reliable way of doing this

"""Do not run this script manually.

This script is intended to be called by the "train_slurm.py" script 
to launch experiments on SLURM clusters.  This script assumes that it
has been launched as part of a SLURM job array, and requires that the
'SLURM_ARRAY_TASK_ID' has been set.  The script accepts a list of 
paths to trial directories, and loads and runs one of these paths depending
on its own SLURM task ID.
"""
import argparse
import os
import torch

from interactive_agents import load_trial, run_trial

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, 
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("paths", type=str, nargs="+",
                        help="paths to all trials (each instance of this script will only run one)")

    # NOTE: These are no longer needed here
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print data for every training iteration")
    parser.add_argument("--flush-secs", type=int, default=200,
                        help="number of seconds after which we should flush the training longs (default 200)")
    
    return parser.parse_args()


if __name__ == '__main__':
    print("Launched on SLURM")
    args = parse_args()

    device = "cpu"  # NOTE: If we want to use a GPU, we need to allocate in in SLURM
    print(f"Training with Torch device '{device}'")

    # Limit Torch CPU parallelism
    torch.set_num_interop_threads(1)
    torch.set_num_threads(1)

    # Get target trial config from SLURM task ID variable
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)
    try:
        trial_path = args.paths[int(task_id)]
    except ValueError:
        raise ValueError(f"Invalid SLURM task ID '{task_id}'")

    # Load trial config
    trial = load_trial(trial_path)

    # Run Trial
    run_trial(trial, device=device, verbose=args.verbose, flush_secs=args.flush_secs)
