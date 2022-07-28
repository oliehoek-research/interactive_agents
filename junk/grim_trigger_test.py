#!/usr/bin/env python3
'''Computes Joint Policy Correlation (JPC) metrics for a collection of different configurations'''
import argparse
from collections import defaultdict
import numpy as np
import os
import os.path
import re
import yaml

import torch

from interactive_agents.envs import get_env_class
from interactive_agents.sampling import sample, FrozenPolicy

# TODO: We should probably automatically save the config if we are saving results to an alternative directory

def parse_args():
    parser = argparse.ArgumentParser("Computes the Joint Policy Correlation matrix (JPC) for a set of trained policies.")

    parser.add_argument("path", type=str, help="path to the policy to be analyzed")
    parser.add_argument("--stages", type=int, default=8,
                        help="number of stages of the coordination game to run")
    parser.add_argument("--actions", type=int, default=5,
                        help="number of actions in the coordination game (must be compatible with policy)")

    return parser.parse_args()



def load_populations(path, 
                     policy_map):  # NOTE: Need to make sure this can handle arbitrary seeds
    populations = defaultdict(dict)  # NOTE: Dictionary of dictionaries
    config_path = os.path.join(path, "config.yaml")
    
    if not os.path.isfile(config_path):
        raise ValueError(f"Config File '{config_path}' does not exist")

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if "trainer" not in config:  # NOTE: Needed when the top-level config dict has a single entry with the experiment name (this is the default)
        config = list(config.values())[0]

    trainer_config = config.get("config", {})

    max_steps = config.get("max_steps", 100)

    env_name = trainer_config.get("env")
    env_cls = get_env_class(env_name)
    
    env_config = trainer_config.get("env_config", {})
    env_config = trainer_config.get("env_eval_config", env_config)  # NOTE: If available, use the evaluation env configuration

    env = env_cls(env_config, spec_only=True)

    if policy_map is None:
        map = {}
        for policy_id in env.possible_agents:
            map[policy_id] = policy_id
    else:
        map = {}  # NOTE: Need two policy maps

        for idx in range(0, len(policy_map), 2):
            agent_id = policy_map[idx]
            policy_id = policy_map[idx + 1]
            map[agent_id] = policy_id
        
        for agent_id in env.possible_agents:
            assert agent_id in map, f"no policy map given for '{agent_id}'"

    exp = re.compile("seed_(\d+)")
    for obj in os.listdir(path):
        match = exp.match(obj)
        if match is not None:
            seed = match.group(1)  # NOTE: Okay to leave the seed as a string
            seed_path = os.path.join(path, obj, "policies")

            if os.path.isdir(seed_path):
                print(f"\nloading policies from: {seed_path}")
                
                for agent_id, policy_id in map.items():
                    policy_path = os.path.join(seed_path, f"{policy_id}.pt")
                    print(f"loading: {policy_path}")

                    if os.path.isfile(policy_path):
                        model = torch.jit.load(policy_path)
                        populations[seed][agent_id] = model  # NOTE: could end up loading the policy multiple times - not ideal
                    else:
                        raise FileNotFoundError(f"seed {seed} does not define policy '{policy_id}'")
    
    return populations, env.possible_agents, env_cls, env_config, max_steps


if __name__ == '__main__':
    args = parse_args()

    # Load pre-trained policy
    if os.path.isfile(args.path):
        model = torch.jit.load(args.path)
        policy = FrozenPolicy(model)
    else:
        raise FileNotFoundError(f"No pre-trained policy found!")

    # Initialize environment
    

    # Limit CPU paralellism for policy inference
    torch.set_num_threads(args.num_cpus)

    if args.output_path is not None:
        path = get_dir(args.output_path, args.tag)
    else:
        path = get_dir(args.path, args.tag)

    print(f"Loading policies from: {args.path}")
    populations, agent_ids, env_cls, env_config, max_steps = load_populations(args.path, args.mapping)

    print(f"Evaluating Policies with {args.num_cpus} processes")
    jpc, seeds = cross_evaluate(populations=populations,
                                agent_ids=agent_ids,
                                env_cls=env_cls,
                                env_config=env_config,
                                max_steps=max_steps,
                                num_cpus=args.num_cpus,
                                num_episodes=args.num_episodes,
                                adversaries=args.adversaries)

    print("\nJPC Tensor:")
    print(jpc)

    np.save(os.path.join(path, "jpc.npy"), jpc, allow_pickle=False)

    stats = jpc_stats(jpc)
    print("statistics:")

    for key, value in stats.items():
        print(f"    {key}: {value}")

    with open(os.path.join(path, "jpc_stats.yaml"), 'w') as f:
        yaml.dump(stats, f)

    print(f"\nrendering JPC tensor")
    plot_matrix(
        matrix=jpc,
        seeds=seeds,
        path=os.path.join(path, "jpc.png"),
        title=args.title,
        min=args.min,
        max=args.max,
        hide_seeds=args.hide_seeds,
        disp=args.display)    
