#!/usr/bin/env python3

'''
Generic RLLib training script.
'''

import argparse
import yaml

import ray
from ray.tune import run_experiments


def parse_args():
    parser = argparse.ArgumentParser("Generic training script for any registered single agent environment, logs intrinsic reward stats.")

    parser.add_argument("-f", "--config-file", default=None, type=str, action="append",
                        help="If specified, use config options from these files.")
    parser.add_argument("--local-dir", type=str, default="../../results/debug",
                        help="path to save training results")
    parser.add_argument("--num-cpus", type=float, default=4,
                        help="the maximum number of CPUs ray is allowed to us, useful for running locally")
    parser.add_argument("--num-gpus", type=float, default=0,
                        help="the number of GPUs to allocate for each trainer")
    parser.add_argument("--total-gpus", type=float, default=0,
                        help="the maximum number of GPUs ray is allowed to use")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="the number of parallel workers per experiment")
    
    return parser.parse_args()


def main(args):
    if args.config_file:
        EXPERIMENTS = dict()

        for config_file in args.config_file:
            with open(config_file) as f:
                EXPERIMENTS.update(yaml.load(f, Loader=yaml.FullLoader))
    else:
        EXPERIMENTS = {
            # "PPO_cartpole": {
            "PPO_mountaincar": {
                "run": "PPO",
                "env": "MountainCar-v0",
                "stop": {
                    "timesteps_total": 100000,
                },
                "checkpoint_freq": 10,
                "local_dir": "../ray_results",
                "num_samples": 2,
                "config": {
                    "gamma": 0.99,
                    "lambda": 0.95,
                    "entropy_coeff": 0.001,
                    "clip_param": 0.1,
                    "lr": 0.001,
                    "num_sgd_iter": 4,
                },
            },
        }

    for experiment in EXPERIMENTS.values():

        # Set local directory for checkpoints
        experiment["local_dir"] = str(args.local_dir)

        # Set num workers
        experiment["config"]["num_workers"] = args.num_workers

        # Set num GPUs per trainer
        experiment["config"]["num_gpus"] = args.num_gpus

        # Modify config to reduce TensorFlow thrashing
        experiment["config"]["tf_session_args"] = {
            "intra_op_parallelism_threads": 1,
            "inter_op_parallelism_threads": 1,
        }

        experiment["config"]["local_tf_session_args"] = {
            "intra_op_parallelism_threads": 1,
            "inter_op_parallelism_threads": 1,
        }

    ray.init(num_cpus=args.num_cpus, num_gpus=args.total_gpus)
    run_experiments(EXPERIMENTS, verbose=0)

if __name__ == '__main__':
    args = parse_args()
    main(args)
