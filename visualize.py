# TODO: Modify this script to support environment visualization, and eventually rendering as well

#!/usr/bin/env python3
'''Computes the Joint Policy Correlation matrix for a set of trained policies'''
import argparse
from collections import defaultdict
import io
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import os
import os.path
import traceback
import yaml

import torch

from interactive_agents.envs import get_env_class
from interactive_agents.sampling import sample, FrozenPolicy


def print_error(error):
    traceback.print_exception(type(error), error, error.__traceback__, limit=5)


def parse_args():
    parser = argparse.ArgumentParser("Visualizes a set of trained policies")

    parser.add_argument("path", type=str, help="path to directory containing training results")
    parser.add_argument("-e", "--num-episodes", type=int, default=100,
                        help="the number of episodes to run for each policy combination")
    parser.add_argument("-m", "--map", nargs="+",
                        help="the mapping from agents to policies")
    parser.add_argument("-s", "--seeds", nargs="+",
                        help="which random seeds each policy should be drawn from")


    return parser.parse_args()


def load_policies(path, policy_map, seeds):
    # TODO: Give a path to a single batch, load the required policies, yielding a single policy dictionary (with different policies potentially drawn from different seeds)
    population = {}
    config_path = os.path.join(path, "config.yaml")
    
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config File: '{config_path}' not found")

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if "trainer" not in config:  # NOTE: Needed because some configs are actually dictionaries of named configs, while others are unnamed
        config = list(config.values())[0]

    trainer_config = config.get("config", {})

    if policy_map is None:
        env_name = trainer_config.get("env")
        env_config = trainer_config.get("env_config", {})
        env_config = trainer_config.get("env_eval_config", env_config)

        env_cls = get_env_class(env_name)
        env = env_cls(env_config, spec_only=True)  # NOTE: Ideally need a better way to do this, a better environment interface

        map = {}
        for policy_id in env.observation_space.keys():
            map[policy_id] = policy_id
    else:
        map = {}

        for idx in range(0, len(policy_map), 2):
            agent_id = policy_map[idx]
            policy_id = policy_map[idx + 1]

            if agent_id.isnumeric():
                agent_id = int(agent_id)

            map[agent_id] = policy_id

    sub_path = os.path.join(path, f"seed_{seed}/policies")

    if os.path.isdir(sub_path):
        print(f"\nloading path: {sub_path}")

        for agent_id, policy_id in map.items():
            policy_path = os.path.join(sub_path, f"{policy_id}.pt")
            print(f"loading: {policy_path}")

            if os.path.isfile(policy_path):
                model = torch.jit.load(policy_path)
                population[agent_id] = model
            else:
                raise FileNotFoundError(f"seed '{seed}' does not define policy '{policy_id}'")
    
    return population, trainer_config


def evaluate(env_cls, env_config, models, num_episodes, max_steps):

    # Build environment instance
    env = env_cls(env_config)  # NOTE: We were doing this for multi-threaded execution, do we need this anymore?

    # Instantiate policies
    policies = {}
    for id, model in models.items():
        if isinstance(model, io.BytesIO):
            model.seek(0)
            model = torch.jit.load(model)
        
        policies[id] = FrozenPolicy(model)

    _, stats = sample(env, policies, num_episodes, max_steps)
    return stats


def cross_evaluate(populations, config, num_cpus, num_episodes):

    # NOTE: Used as a handle for single-threaded execution
    class dummy_async:  # NOTE, we don't really need this anymore, since visualization will be single threaded

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
        models = {}
        for a, p in enumerate(permutation):
            agent_id = agent_ids[a]
            models[agent_id] = populations[p][agent_id]

        idx = tuple(permutation)
        if num_cpus > 1:

            # Serialize torch policies
            for id, model in models.items():
                buffer = io.BytesIO()
                torch.jit.save(model, buffer)
                models[id] = buffer

            threads[idx] = pool.apply_async(evaluate, (env_cls, env_config, 
                models, num_episodes, max_steps), error_callback=print_error)
        else:
            threads[idx] = dummy_async(evaluate(env_cls, 
                env_config, models, num_episodes, max_steps))

    returns = np.zeros(tuple([num_populations] * num_agents))
    for idx, thread in threads.items():
        stats = thread.get()
        returns[idx] = stats["mean_reward"]

    return returns


if __name__ == '__main__':
    args = parse_args()

    # TODO: How do we specify the mapping we want to analyze?
    print(f"Loading policies from: {args.path}")  # TODO: Need to load two specific populations from two runs
    population, config = load_population(args.path, args.seed, args.map)

    print(f"Evaluating Policies")
    jpc = cross_evaluate(populations, config, args.num_cpus, args.num_episodes)

    print("\nJCP Tensor:")
    print(jpc)

    if args.output_path is not None:
        matrix_path = os.path.join(args.output_path, args.filename + ".npy")
        image_path = os.path.join(args.output_path, args.filename + ".png")
    else:
        matrix_path = os.path.join(args.path, args.filename + ".npy")
        image_path = os.path.join(args.path, args.filename + ".png")

    print(f"\nwriting JPC tensor to: {matrix_path}")
    np.save(matrix_path, jpc, allow_pickle=False)

    if len(jpc.shape) == 2:
        print(f"\nrendering JPC tensor to: {matrix_path}")
        plot_matrix(
            jpc, 
            image_path,
            title=args.title,
            min=args.min,
            max=args.max,
            disp=args.display)    

