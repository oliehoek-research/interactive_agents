#!/usr/bin/env python3
'''Computes the Joint Policy Correlation matrix for a set of trained policies'''
import argparse
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
import os
import os.path
import pickle
import traceback
import yaml

from interactive_agents.envs import get_env_class
from interactive_agents.learning import get_trainer_class
from interactive_agents.sampling import Sampler

# TODO: Tensorflow models seem to hang whenwe try to run them in a distributed fashion, need to initialize models separately in each thread like pytorch does

def print_error(error):
    traceback.print_exception(type(error), error, error.__traceback__, limit=5)


def parse_args():
    parser = argparse.ArgumentParser("Computes the Joint Policy Correlation matrix for a set of trained policies")

    parser.add_argument("path", type=str, help="path to directory containing training results")
    parser.add_argument("-o", "--output-path", type=str, default=None,
                        help="directory in which we should save matrix (defaults to experiment directory)")
    parser.add_argument("-n", "--num-cpus", type=int, default=4,
                        help="the number of parallel worker processes to launch")
    parser.add_argument("-e", "--num-episodes", type=int, default=4,
                        help="the number of episodes to run for each policy combination")


    return parser.parse_args()


def load_populations(path):
    populations = {}
    config_path = os.path.join(path, "config.yaml")
    
    if not os.path.isfile(config_path):
        raise ValueError(f"Config File: '{config_path}' not defined")

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if "trainer" not in config:
        config = list(config.values())[0]

    trainer_cls = get_trainer_class(config.get("trainer", "independent"))
    trainer_config = config.get("config", {})

    for seed in range(config.get("num_seeds", 1)):
        sub_path = os.path.join(path, f"seed_{seed}")

        if os.path.isdir(sub_path):
            state_path = os.path.join(sub_path, "state.pickle")

            if os.path.isfile(state_path):
                with open(state_path, 'rb') as f:
                    state = pickle.load(f)

                trainer = trainer_cls(trainer_config)
                trainer.set_state(state)
                
                populations[seed] = trainer.get_policies()
    
    return populations, trainer_config


def evaluate(policies, env_name, env_config, num_episodes, max_steps):
    policy_fn = lambda id: id
    sampler = Sampler(env_name, env_config, policies, policy_fn, max_steps)
    _, stats = sampler.sample(num_episodes)
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
            threads[idx] = pool.apply_async(evaluate, 
                (policies, env_name, env_config, num_episodes, max_steps), error_callback=print_error)
        else:
            threads[idx] = dummy_async(evaluate(policies, env_name, env_config, num_episodes, max_steps))

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

    print("JCP Tensor:")
    print(jpc)

    if args.output_path is not None:
        output_path = os.path.join(args.output_path, "jpc.npy")
    else:
        output_path = os.path.join(args.path, "jpc.npy")

    np.save(output_path, jpc, allow_pickle=False)
