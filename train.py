#!/usr/bin/env python3
'''Simple script for launching experiments'''
import argparse

from interactive_agents.core import load_configs, run_experiments


def parse_args():
    parser = argparse.ArgumentParser("Training script for finite, constant-sum extensive-form games")

    parser.add_argument("-f", "--config-file", default=None, type=str, action="append",
                        help="If specified, use config options from this file.")
    parser.add_argument("-o", "--output-path", type=str, default="results/debug",
                        help="directory in which we should save results")
    parser.add_argument("-n", "--num-cpus", type=int, default=2,
                        help="the number of parallel worker processes to launch")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if len(args.config_file) > 0:
        experiments = load_configs(args.config_file)
    else:
        experiments = {
            "DQN_debug": {
                "stop": {
                    "total_iterations": 10
                },
                "trainer": "independent",
                "num_seeds": 2,
                "config": {
                    "max_steps": 100,
                    "iteration_episodes": 100,
                    "eval_episodes": 10,
                    "env": "coordination",
                    "env_config": {
                        "stages": 5,
                        "actions": 4,
                        "players": 2
                    },
                    "learner": "R2D2",
                    "learner_config": {},
                }
            }
        }

    run_experiments(experiments, args.output_path, args.num_cpus)
