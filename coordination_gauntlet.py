#!/usr/bin/env python3
"""Evaluates trained policies against a set of fixed agents in the coordination game"""
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
    parser = argparse.ArgumentParser("Evaluates trained policies against a set of fixed agents in the coordination game.")  # TODO: I think we are using this string wrong

    parser.add_argument("path", type=str, help="path to directory containing training results")

    parser.add_argument("-p", "--pid", type=str, default="agent_0",
                        help="name of the saved policy to evaluate")

    parser.add_argument("-s", "--stages", type=int, default=16,
                        help="the number of stages per episode of the coordination game")
    parser.add_argument("-a", "--actions", type=int, default=5,
                        help="the number of actions in the coordination game")

    parser.add_argument("-e", "--num-episodes", type=int, default=100,
                        help="the number of episodes to run for each policy-partner combination")
    parser.add_argument("-n", "--noise", type=float, default=None,
                        help="if given, repeats gauntlet with noisy actions")

    return parser.parse_args()


class FixedActionAgent:

    def __init__(self, action):
        self._action = action

    def reset(self):
        pass

    def act(self, last_partner_action):
        return self._action


class BestResponseAgent:

    def __init__(self, num_actions):
        self._num_actions = num_actions

    def reset(self):
        pass

    def act(self, last_partner_action):
        if last_partner_action is not None:
            return last_partner_action
        else:
            return np.random.randint(self._num_actions)


class FictitiousPlayAgent:

    def __init__(self, num_actions):
        self._num_actions = num_actions
        self._action_counts = None
    
    def reset(self):
        self._action_counts = np.zeros(self._num_actions)

    def act(self, last_partner_action):
        if last_partner_action is not None:
            self._action_counts[last_partner_action] += 1
            return self._action_counts.argmax()
        else:
            return np.random.randint(self._num_actions)


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


def sample(env, policy, partner, episodes, noise):
    total_reward = 0

    for _ in range(episodes):
        agent = policy.make_agent()
        partner.reset()

        last_action = None

        obs = env.reset()
        dones = {"agent_0": False}

        while not dones["agent_0"]:
            actions = {"agent_1": partner.act(last_action)}

            if 0.0 == noise or np.random.random() > noise:
                actions["agent_0"], _ = agent.act(obs["agent_0"])
            else:
                actions["agent_0"] = env.action_spaces["agent_0"].sample()

            obs, rewards, dones, _ = env.step(actions)

            total_reward += rewards["agent_0"]
            last_action = actions["agent_0"]
    
    return total_reward / episodes


def gauntlet(env, policies, episodes, noise=0.0):
    num_actions = env.action_spaces["agent_0"].n
    stats = defaultdict(lambda: 0)

    for policy in policies:

        # Fixed action agents
        returns = []
        for action in range(num_actions):
            agent = FixedActionAgent(action)
            returns.append(sample(env, policy, agent, episodes, noise))
            stats[f"fixed_{action}"] += returns[action]
        
        stats["fixed_mean"] += np.mean(returns)
        stats["fixed_min"] += np.min(returns)
        stats["fixed_max"] += np.max(returns)

        # Best Response
        agent = BestResponseAgent(num_actions)
        best_response = sample(env, policy, agent, episodes, noise)
        stats["best_response"] += best_response
        returns.append(best_response)

        # Fictitious Play
        agent = FictitiousPlayAgent(num_actions)
        fictitious_play = sample(env, policy, agent, episodes, noise)
        stats["fictitious_play"] += fictitious_play
        returns.append(fictitious_play)

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
    env_cls = get_env_class("coordination")
    env = env_cls({
        "stages": args.stages,
        "actions": args.actions
    })

    # Run noiseless gauntlet
    stats = gauntlet(env, policies, args.num_episodes)

    print("\n\nCOORDINATION GAUNTLET:")

    for key, value in stats.items():
        print(f"{key}: {value}")

    # Optionally run gauntlet against noisy partners
    if args.noise is not None:
        stats = gauntlet(env, policies, args.num_episodes, noise=args.noise)

        print("\n\nNOISY GAUNTLET:")

        for key, value in stats.items():
            print(f"{key}: {value}")

    print("\n")  
