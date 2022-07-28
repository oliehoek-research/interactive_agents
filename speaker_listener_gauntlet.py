#!/usr/bin/env python3
"""Evaluates trained policies against a set of fixed agents in the speaker-listener game"""
import argparse
from collections import defaultdict
import numpy as np
import os
import os.path
import re

import torch

from interactive_agents.envs import get_env_class
from interactive_agents.sampling import FrozenPolicy


def parse_args():
    parser = argparse.ArgumentParser("Evaluates trained speaker policies against a set of fixed agents in the speaker-listener game.")

    parser.add_argument("path", type=str, help="path to directory containing training results")

    parser.add_argument("-p", "--pid", type=str, default="speaker",
                        help="name of the saved policy to evaluate")

    parser.add_argument("-s", "--stages", type=int, default=16,
                        help="the number of stages per episode of the speaker-listener game")
    parser.add_argument("-a", "--actions", type=int, default=5,
                        help="the number of actions in the speaker-listener game")

    parser.add_argument("-e", "--num-episodes", type=int, default=100,
                        help="the number of episodes to run for each policy-partner combination")
    parser.add_argument("-l", "--num-languages", type=int, default=10,
                        help="the number of fixed-language agents to evaluate against")
    parser.add_argument("-n", "--noise", type=float, default=None,
                        help="if given, repeats gauntlet with noisy actions")

    return parser.parse_args()


class FixedListenerAgent:

    def __init__(self, language):
        self._language = language

    def reset(self):
        pass

    def observe(self, last_action):
        pass

    def act(self, last_statement):
        return self._language[last_statement]


class AdaptiveListenerAgent:

    def __init__(self, num_actions):
        self._num_actions = num_actions
        self._language = None
        self._last_statement = None

    def reset(self):
        self._language = np.random.permutation(self._num_actions)
        self._last_statement = None

    def observe(self, last_action):
        if last_action is not None and self._last_statement is not None:
            self._language[self._last_statement] = last_action

    def act(self, last_statement):
        self._last_statement = last_statement
        return self._language[last_statement]


def load_policies(path, pid):
    print(f"\nloading policies from: {path}")
    policies = []

    exp = re.compile("seed_(\d+)")
    for obj in os.listdir(path):
        match = exp.match(obj)
        if match is not None:
            seed = match.group(1)
            seed_path = os.path.join(path, obj, "policies")

            if os.path.isdir(seed_path):
                policy_path = os.path.join(seed_path, f"{pid}.pt")
                
                if os.path.isfile(policy_path):
                    model = torch.jit.load(policy_path)
                    policies.append(FrozenPolicy(model))
                else:
                    raise FileNotFoundError(f"seed {seed} does not define policy '{pid}'")
    
    print(f"loaded {len(policies)} policies")
    return policies

# TODO: Update to handle speaker-listener game
def sample(env, policy, partner, episodes, noise):
    total_reward = 0

    for _ in range(episodes):
        agent = policy.make_agent()
        partner.reset()

        last_statement = None
        last_action = None

        obs = env.reset()
        dones = {"speaker": False}
        steps = 0

        while not dones["speaker"]:
            actions = {}

            if 0.0 == noise or np.random.random() > noise:
                actions["speaker"], _ = agent.act(obs["speaker"])
            else:
                actions["speaker"] = env.action_spaces["speaker"].sample()

            steps += 1
            if 0 == steps % 2:
                partner.observe(last_action)
                last_statement = actions["speaker"]
                actions["listener"] = 0
            else:
                last_action = actions["speaker"]
                actions["listener"] = partner.act(last_statement)

            obs, rewards, dones, _ = env.step(actions)
            total_reward += rewards["speaker"]
    
    return total_reward / episodes


def gauntlet(env, policies, episodes, languages, noise=0.0):
    num_actions = env.action_spaces["speaker"].n
    stats = defaultdict(lambda: 0)

    for policy in policies:

        # Fixed language agents
        returns = []
        for _ in range(languages):
            agent = FixedListenerAgent(np.random.permutation(num_actions))
            returns.append(sample(env, policy, agent, episodes, noise))
        
        stats["fixed_mean"] += np.mean(returns)
        stats["fixed_min"] += np.min(returns)
        stats["fixed_max"] += np.max(returns)

        # Best Response
        agent = AdaptiveListenerAgent(num_actions)
        adaptive = sample(env, policy, agent, episodes, noise)
        stats["adaptive"] += adaptive
        returns.append(adaptive)


        # Population Stats
        stats["mean"] += np.mean(returns)
        stats["min"] += np.min(returns)
        stats["max"] += np.max(returns)
    
    # Compute means across seeds
    for key in stats.keys():
        stats[key] /= len(policies)

    return stats


if __name__ == '__main__':
    args = parse_args()

    # Load pre-trained policies
    policies = load_policies(args.path, args.pid)

    # Initialize coordination environment
    env_cls = get_env_class("linguistic")
    env = env_cls({
        "stages": args.stages,
        "actions": args.actions
    })

    # Run noiseless gauntlet
    stats = gauntlet(env, policies, args.num_episodes, args.num_languages)

    print("\n\nSPEAKER-LISTENER GAUNTLET:")

    for key, value in stats.items():
        print(f"{key}: {value}")

    # Optionally run gauntlet against noisy partners
    if args.noise is not None:
        stats = gauntlet(env, policies, 
            args.num_episodes, args.num_languages, noise=args.noise)

        print("\n\nNOISY GAUNTLET:")

        for key, value in stats.items():
            print(f"{key}: {value}")

    print("\n")  
