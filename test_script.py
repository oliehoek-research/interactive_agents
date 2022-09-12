"""Test script for sbatch to call"""
import argparse
import numpy as np
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", action="store_true",
                        help="If provided, just setup the experiment and print the number of trials to run")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.setup:
        print(2)
    else:
        print(f"running task {os.environ['SLURM_ARRAY_TASK_ID']}")
