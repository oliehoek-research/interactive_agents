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
    parser = argparse.ArgumentParser("Computes the Joint Policy Correlation matrix for a set of trained policies")

    parser.add_argument("path", type=str, help="path to directory containing training results")
    parser.add_argument("-o", "--output-path", type=str, default=None,
                        help="directory in which we should save matrix (defaults to experiment directory)")
    parser.add_argument("-f", "--filename", type=str, default="jpc",
                        help="filename for saved matrix")
    parser.add_argument("-n", "--num-cpus", type=int, default=1,
                        help="the number of parallel worker processes to launch")
    parser.add_argument("-e", "--num-episodes", type=int, default=100,
                        help="the number of episodes to run for each policy combination")
    parser.add_argument("-m", "--map", nargs="+")

    parser.add_argument("--title", type=str, default="Joint Policy Correlation",
                        help="title for figure")
    parser.add_argument("--min", type=float, help="min payoff value (for image rendering)")
    parser.add_argument("--max", type=float, help="max payoff value (for image rendering)")

    return parser.parse_args()


def plot_matrix(matrix, path, title, min, max, size=300):
    if min is None:
        min = matrix.min()

    if max is None:
        max = matrix.max()

    # Scale range to cut off dark reds
    max += 0.15 * (max - min)
    cm = plt.get_cmap("jet")

    # Ticks for each seed on the x and y axis
    tick_space = size / matrix.shape[0]
    tick_pos = 0.5 * tick_space
    ticks = []
    labels = []

    for idx in range(matrix.shape[0]):
        ticks.append(tick_pos)
        labels.append(idx)
        tick_pos += tick_space
    
    # Generate figure
    plt.clf()
    im = plt.imshow(matrix, 
        cmap=cm,
        vmin=min,
        vmax=max,
        extent=(0,size,0,size))
    plt.colorbar(im)

    plt.xticks(ticks, labels=labels)
    plt.yticks(ticks, labels=labels)

    ax = plt.gca()
    ax.grid(which='minor', color='k', linestyle='-', linewidth=2)

    plt.title(title, fontsize=14)
    plt.xlabel("seeds", fontsize=16)
    plt.ylabel("seeds", fontsize=16)
    plt.savefig(path, bbox_inches="tight")


def load_config(path, map):
    config_path = os.path.join(path, "config.yaml")
    
    if not os.path.isfile(config_path):
        raise ValueError(f"Config File: '{config_path}' not defined")

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if "trainer" not in config:  # NOTE: When would this be needed?
        config = list(config.values())[0]

    trainer_config = config.get("config", {})

    if map is None:
        env_name = trainer_config.get("env")
        env_config = trainer_config.get("env_config", {})

        env_cls = get_env_class(env_name)
        env = env_cls(env_config, spec_only=True)

        map = {}
        for policy_id in env.observation_space.keys():
            map[policy_id] = policy_id

    return trainer_config, map, config.get("num_seeds", 1)


def load_populations(path, policy_map):
    populations = defaultdict(dict)
    config_path = os.path.join(path, "config.yaml")
    
    if not os.path.isfile(config_path):
        raise ValueError(f"Config File: '{config_path}' not defined")

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if "trainer" not in config:  # NOTE: When would this be needed?
        config = list(config.values())[0]

    trainer_config = config.get("config", {})

    if policy_map is None:
        env_name = trainer_config.get("env")
        env_config = trainer_config.get("env_config", {})
        env_config = trainer_config.get("env_eval_config", env_config)

        env_cls = get_env_class(env_name)
        env = env_cls(env_config, spec_only=True)

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

    for seed in range(config.get("num_seeds", 1)):
        sub_path = os.path.join(path, f"seed_{seed}/policies")

        if os.path.isdir(sub_path):
            print(f"\nloading path: {sub_path}")

            for agent_id, policy_id in map.items():
                policy_path = os.path.join(sub_path, f"{policy_id}.pt")
                print(f"loading: {policy_path}")

                if os.path.isfile(policy_path):
                    model = torch.jit.load(policy_path)
                    populations[seed][agent_id] = model
                else:
                    raise FileNotFoundError(f"seed '{seed}' does not define policy '{policy_id}'")
    
    return populations, trainer_config


def evaluate(env_cls, env_config, models, num_episodes, max_steps):

    # Build environment instance
    env = env_cls(env_config)

    # Instantiate policies
    policies = {}
    for id, model in models.items():
        if isinstance(model, io.BytesIO):
            model.seek(0)
            model = torch.jit.load(model)
        
        policies[id] = FrozenPolicy(model)

    _, stats = sample(env, policies, num_episodes, max_steps)
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

    # Limit CPU paralellism
    torch.set_num_threads(args.num_cpus)

    print(f"Loading policies from: {args.path}")
    populations, config = load_populations(args.path, args.map)

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
            max=args.max)    

