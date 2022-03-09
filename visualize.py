#!/usr/bin/env python3
'''Visualizes (and optionally records) rollouts of a collection of policy checkpoints'''
import argparse
import gym
import os
import os.path
import yaml

import torch

from interactive_agents.envs import get_env_class, VisualizeGym
from interactive_agents.sampling import FrozenPolicy


def parse_args():
    parser = argparse.ArgumentParser("Visualizes a set of trained policies")

    parser.add_argument("path", type=str, help="path to directory containing the policy checkpoints")
    parser.add_argument("-e", "--num-episodes", type=int, default=100,
                        help="the number of episodes to run (default: 100)")
    parser.add_argument("-s", "--max-steps", type=int, default=1000,
                        help="the maximum number of steps per episode (default: 1000)")
    parser.add_argument("-m", "--map", nargs="+",
                        help="the mapping from agents to policies")
    parser.add_argument("--seed", type=int, default=0,
                        help="the random seed of the training run to load (default: 0)")
    parser.add_argument("-r", "--record", type=str,
                        help="the path to save recorded videos (no recording if not provided)")
    parser.add_argument("--headless", action="store_true",
                        help="do not display visualization (record only in headless environments)")
    parser.add_argument("--speed", type=float, default=50,
                        help="the speed at which to play the visualization (in steps per second)")

    return parser.parse_args()


# TODO: Move policy loading code to the main library, so we don't need to reproduce it for every 
def load_experiment(path, seed, policy_map):
    policies = {}
    config_path = os.path.join(path, "config.yaml")
    
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config File: '{config_path}' not found")

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # NOTE: Needed because most configs are actually dictionaries with multiple configs
    if "trainer" not in config:
        config = list(config.values())[0]

    # Load environment
    trainer_config = config.get("config", {})
    env_name = trainer_config.get("env")
    env_config = trainer_config.get("env_config", {})
    env_config = trainer_config.get("env_eval_config", env_config)

    env_cls = get_env_class(env_name)
    env = env_cls(env_config)

    # If we don't specify a mapping from policies to agents, assume there is a 1-1 mapping between policies and agents
    if policy_map is None:
        
        policy_map = {}
        for policy_id in env.observation_space.keys():
            policy_map[policy_id] = policy_id

    # Load directory that for the desired random seed
    sub_path = os.path.join(path, f"seed_{seed}/policies")

    if not os.path.isdir(sub_path):
        raise FileNotFoundError(f"Directory: '{sub_path}' not found")

    for agent_id, policy_id in policy_map.items():
        policy_path = os.path.join(sub_path, f"{policy_id}.pt")

        if os.path.isfile(policy_path):
            model = torch.jit.load(policy_path)
            policies[agent_id] = FrozenPolicy(model)
        else:
            raise FileNotFoundError(f"seed '{seed}' does not define policy '{policy_id}'")
    
    return policies, env


if __name__ == '__main__':
    args = parse_args()

    # Parse policy mapping if provided as a command line argument
    if args.map is not None:
        policy_map = {}
        for idx in range(0, len(args.map), 2):
            agent_id = policy_map[idx]
            policy_id = policy_map[idx + 1]

            if agent_id.isnumeric():  # NOTE: This is a hack due to the fact that most environments us integer agent IDs
                agent_id = int(agent_id)

            policy_map[agent_id] = policy_id
        
        policy_fn = lambda id: policy_map[id]
    else:
        policy_map = None
        policy_fn = lambda id: id

    # Load policies from experiment directory
    print(f"Loading policies from: {args.path}")
    policies, env = load_experiment(args.path, args.seed, policy_map)

    # If environment doesn't support visualization, wrap with gym visualizer
    if not hasattr(env, "visualize"):
        if isinstance(env, gym.Env):
            env = VisualizeGym(env)
        else:
            raise NotImplementedError("Environment does not support visualization")

    # Launch visualization
    env.visualize(policies=policies,
                  policy_fn=policy_fn,
                  max_episodes=args.num_episodes,
                  max_steps=args.max_steps,
                  speed=args.speed,
                  record_path=args.record,
                  headless=args.headless) 
