"""Test script that just imports our library"""
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--job-id", default=None, type=int,
                        help="Job ID for SLURM job arrays.")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Current working directory: {os.getcwd()}")

    if args.job_id is not None:
        print(f"Job array index: {args.job_id}")
