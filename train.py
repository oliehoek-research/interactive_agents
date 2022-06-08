#!/usr/bin/env python3
"""Simple script for launching experiments"""
import argparse

from interactive_agents.run import load_configs, run_experiments


def parse_args():
    parser = argparse.ArgumentParser("Training script for finite, constant-sum extensive-form games")

    parser.add_argument("-f", "--config-file", default=None, type=str, action="append",
                        help="if specified, use config options from this file.")
    parser.add_argument("-o", "--output-path", type=str, default="results/debug",
                        help="directory in which we should save results")
    parser.add_argument("-n", "--num-cpus", type=int, default=2,
                        help="the number of parallel worker processes to launch")
    parser.add_argument("-g", "--gpu", action="store_true",
                        help="enable GPU if available")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print data for every training iteration")
    parser.add_argument("--num-seeds", type=int,
                        help="the number of random seeds to run, overrides values from the config file")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="the list of random seeds to run, overrides values from the config file")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.config_file is not None:
        experiments = load_configs(args.config_file)
    else:
        experiments = {
            "R2D2_debug": {
                "stop": {
                    "iterations": 100
                },
                "trainer": "independent",
                "num_seeds": 1,
                "config": {
                    "eval_iterations": 10,
                    "eval_episodes": 32,
                    "iteration_episodes": 32,
                    "env": "memory",
                    "env_config": {
                        "length": 10,
                        "num_cues": 4,
                        "noise": 0.1,
                    },
                    "learner": "R2D2",
                    "learner_config": {
                        "num_batches": 16,
                        "batch_size": 16,
                        "sync_iterations": 5,
                        "learning_starts": 10,
                        "gamma": 0.99,
                        "beta": 0.5,
                        "double_q": True,
                        "epsilon_initial": 0.5,
                        "epsilon_iterations": 100,
                        "epsilon_final": 0.01,
                        "replay_alpha": 0.0,
                        "replay_epsilon": 0.01,
                        "replay_eta": 0.5,
                        "replay_beta_iterations": 100,
                        "buffer_size": 2048,
                        "dueling": True,
                        "model": "lstm",
                        "model_config": {
                            "hidden_size": 64,
                            "hidden_layers": 1,
                        },
                        "lr": 0.001,
                    },
                }
            }
        }

    device = "cuda" if args.gpu else "cpu"
    print(f"Training with Torch device '{device}'")

    run_experiments(experiments, args.output_path, 
        args.num_cpus, device, args.verbose, args.num_seeds, args.seeds)
