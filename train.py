# NOTE: This was set up to allow experiments to be run on the Aalto cluster.  Shouldn't need separate configs anymore.
"""Simple script for launching experiments"""
import argparse

from interactive_agents.run import load_configs, run_experiments, run_experiments_triton


def parse_args():
    parser = argparse.ArgumentParser("Training script for finite, constant-sum extensive-form games")

    parser.add_argument("-f", "--config-file", default=None, type=str, action="append",
                        help="if specified, use config options from this file.")
    parser.add_argument("-o", "--output-path", type=str, default="results/debug",
                        help="directory in which we should save results")
    
    # NOTE: Currently, parallelism is only allowed across seeds and configurations, none of our trainers support multiple worker threads
    parser.add_argument("-n", "--num-cpus", type=int, default=1,
                        help="the number of parallel worker processes to launch")

    # NOTE: We will need to figure out how to specify GPU resources available through SLURM
    parser.add_argument("-g", "--gpu", action="store_true",
                        help="enable GPU if available")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print data for every training iteration")
    parser.add_argument("--num-seeds", type=int,
                        help="the number of random seeds to run, overrides values from the config file")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="the list of random seeds to run, overrides values from the config file")
    parser.add_argument("-t", "--triton", action="store_true",
                        help="the training is done in triton. changes the train and run behaviour!!!")

    # NOTE: How this is used is trainer dependent
    parser.add_argument("-r", "--resources", nargs="+",
                        help="a list of key-value pairs representing file resources (policies, datasets, etc.)")  # NOTE: This is mainly used to load pre-trained partner strategies
    
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

    if args.triton is True:
        print("Experiments are running on Triton.")
        run_experiments_triton(experiments, args.output_path, 
            args.num_cpus, device, args.verbose, args.num_seeds, args.seeds)
    else:
        print("Experiments NOT running on triton. Use --triton if you want to run on triton!")
        run_experiments(experiments, args.output_path, 
        args.num_cpus, device, args.verbose, args.num_seeds, args.seeds, args.resources)
