#!/usr/bin/env python3
'''Computes the Joint Policy Correlation matrix for a set of trained policies'''
import argparse
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
import os
import os.path
import traceback
import yaml

import torch

from interactive_agents.envs import get_env_class
from interactive_agents.sampling import sample, FrozenPolicy

# TODO: Tensorflow models seem to hang when we try to run them in a distributed fashion, need to initialize models separately in each thread like pytorch does

def print_error(error):
    traceback.print_exception(type(error), error, error.__traceback__, limit=5)


def parse_args():
    parser = argparse.ArgumentParser("Computes the Joint Policy Correlation matrix for a set of trained policies")

    parser.add_argument("path", type=str, help="path to directory containing training results")
    parser.add_argument("-o", "--output-path", type=str, default=None,
                        help="directory in which we should save matrix (defaults to experiment directory)")
    parser.add_argument("-n", "--num-cpus", type=int, default=1,
                        help="the number of parallel worker processes to launch")
    parser.add_argument("-e", "--num-episodes", type=int, default=100,
                        help="the number of episodes to run for each policy combination")


    return parser.parse_args()


def load_populations(path):
    populations = defaultdict(dict)
    config_path = os.path.join(path, "config.yaml")
    
    if not os.path.isfile(config_path):
        raise ValueError(f"Config File: '{config_path}' not defined")

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if "trainer" not in config:  # NOTE: When would this be needed?
        config = list(config.values())[0]

    trainer_config = config.get("config", {})

    env_name = trainer_config.get("env")
    env_config = trainer_config.get("env_config", {})

    env_cls = get_env_class(env_name)
    env = env_cls(env_config, spec_only=True)

    for seed in range(config.get("num_seeds", 1)):
        sub_path = os.path.join(path, f"seed_{seed}/policies")

        if os.path.isdir(sub_path):
            print(f"\nloading path: {sub_path}")

            for policy_id in env.observation_space.keys():
                policy_path= os.path.join(sub_path, f"{policy_id}.pt")
                print(f"loading: {policy_path}")

                if os.path.isfile(policy_path):
                    model = torch.jit.load(policy_path)
                    populations[seed][policy_id] = FrozenPolicy(model)
    
    return populations, trainer_config


def evaluate(env_cls, env_config, policies, num_episodes, max_steps):
    env = env_cls(env_config)
    _, stats =sample(env, policies, num_episodes, max_steps)
    return stats


def permutations(num_agents, num_populations):
        num_permutations = num_populations ** num_agents
        for index in range(num_permutations):
            permutation = [0] * num_agents
            idx = index
            for id in range(num_agents):
                permutation[id] = idx % num_populations
                idx = idx // num_populations
            yield permutation


def cross_evaluate(populations, config, num_cpus, num_episodes):

    # NOTE: Used as a handle for single-threaded execution
    class dummy_async:

        def __init__(self, result):
            self._result = result
        
        def get(self):
            return self._result


    if num_cpus > 1:
        pool = Pool(num_cpus)

    max_steps = config.get("max_steps", 100)

    env_name = config.get("env")
    env_config = config.get("env_config", {})

    env_cls = get_env_class(env_name)
    env = env_cls(env_config, spec_only=True)

    agent_ids = list(env.observation_space.keys())
    population_ids = list(populations.keys())

    num_agents = len(agent_ids)
    num_populations = len(population_ids)

    threads = {}
    for permutation in permutations(num_agents, num_populations):
        policies = {}
        for a, p in enumerate(permutation):
            agent_id = agent_ids[a]
            policies[agent_id] = populations[p][agent_id]

        idx = tuple(permutation)
        if num_cpus > 1:
            threads[idx] = pool.apply_async(evaluate, (env_cls, env_config, 
                policies, num_episodes, max_steps), error_callback=print_error)
        else:
            threads[idx] = dummy_async(evaluate(env_cls, 
                env_config, policies, num_episodes, max_steps))

    returns = np.zeros(tuple([num_populations] * num_agents))
    for idx, thread in threads.items():
        stats = thread.get()
        returns[idx] = stats["mean_reward"]

    return returns


if __name__ == '__main__':
    args = parse_args()

    print(f"Loading policies from: {args.path}")
    populations, config = load_populations(args.path)

    print(f"Evaluating Policies")
    jpc = cross_evaluate(populations, config, args.num_cpus, args.num_episodes)

    print("\nJCP Tensor:")
    print(jpc)

    if args.output_path is not None:
        output_path = os.path.join(args.output_path, "jpc.npy")
    else:
        output_path = os.path.join(args.path, "jpc.npy")

    print(f"\nwriting JPC tensor to: {output_path}")
    np.save(output_path, jpc, allow_pickle=False)
