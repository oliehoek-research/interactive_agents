#!/usr/bin/env python3
"""Evaluates one population's ability to coordinate with policies from another population"""
import argparse
from collections import defaultdict
import io
import numpy as np
import os
import os.path
import re
import traceback
import yaml

import torch
from torch.multiprocessing import Pool

from interactive_agents.envs import get_env_class
from interactive_agents.sampling import sample, FrozenPolicy


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("path", nargs="+", type=str, 
                        help="paths to the directories containing learned policies")
    
    parser.add_argument("-n", "--num-cpus", type=int, default=1,
                        help="the number of parallel worker processes to launch")
    parser.add_argument("-e", "--num-episodes", type=int, default=100,
                        help="the number of episodes to run for each policy combination")

    parser.add_argument("-m", "--mapping", nargs="+",
                        help="mapping from agent IDs to policy names (agent_id policy_name agent_id policy_name ...)")
    parser.add_argument("-a", "--adversaries", nargs="+",
                        help="list of agent IDs corresponding to the 'adversary' team of agents, for games with >2 players")

    return parser.parse_args()


def parse_map(map_spec):
    map = {}
    for idx in range(0, len(map_spec), 2):
        agent_id = map_spec[idx]
        policy_id = map_spec[idx + 1]
        map[agent_id] = policy_id
        
    return map


def load_config(path):
    if not os.path.isfile(path):
        raise ValueError(f"Config File '{path}' does not exist")

    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if "trainer" not in config:
        config = list(config.values())[0]

    trainer_config = config.get("config", {})

    max_steps = config.get("max_steps", 100)

    env_name = trainer_config.get("env")
    env_cls = get_env_class(env_name)
    
    env_config = trainer_config.get("env_config", {})
    env_config = trainer_config.get("env_eval_config", env_config)

    return env_cls, env_config, max_steps


def load_policies(path, policy_map):
    policies = defaultdict(dict)

    exp = re.compile("seed_(\d+)")
    for obj in os.listdir(path):
        match = exp.match(obj)
        if match is not None:
            seed = match.group(1)
            seed_path = os.path.join(path, obj, "policies")

            if os.path.isdir(seed_path):                
                for agent_id, policy_id in policy_map.items():
                    policy_path = os.path.join(seed_path, f"{policy_id}.pt")

                    if os.path.isfile(policy_path):
                        model = torch.jit.load(policy_path)
                        policies[seed][agent_id] = model
    
    return policies


def select_models(population, agent_ids):
    policies = []
    for seed in population.values():
        models = {}
        for agent_id in agent_ids:
            models[agent_id] = seed[agent_id]
        
        policies.append(models)
    
    return policies


def evaluate(env_cls, 
             env_config, 
             models, 
             num_episodes, 
             max_steps):

    # Build environment instance
    env = env_cls(env_config)

    # Instantiate policies
    policies = {}
    for id, model in models.items():
        if isinstance(model, io.BytesIO):
            model.seek(0)
            model = torch.jit.load(model)  # NOTE: Restore models from binary representation
        
        policies[id] = FrozenPolicy(model)

    batch = sample(env, policies, num_episodes, max_steps)
    return batch.statistics()


def print_error(error):
    """Error callback for python multiprocessing"""
    traceback.print_exception(type(error), error, error.__traceback__, limit=5)


def jpc(eval_policies,
        target_policies,
        env_cls, 
        env_config, 
        max_steps, 
        num_episodes,
        num_cpus):

    # NOTE: Used as a handle for single-threaded execution
    class dummy_async:

        def __init__(self, result):
            self._result = result
        
        def get(self):
            return self._result


    if num_cpus > 1:
        pool = Pool(num_cpus)

    threads = {}
    for eval_id, eval_team in enumerate(eval_policies):
        for target_id, target_team in enumerate(target_policies):
            models = {}

            for id, model in eval_team.items():
                models[id] = model
            
            for id, model in target_team.items():
                models[id] = model

            idx = (eval_id, target_id)

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

    jpc = np.zeros((len(eval_policies), len(target_policies)))
    for idx, thread in threads.items():
        stats = thread.get()
        jpc[idx] = stats["reward_mean"]

    return jpc


def cross_evaluate(eval_path, 
                   target_path, 
                   eval_map_spec, 
                   adversaries, 
                   num_episodes,
                   num_cpus):

    # Load environment config
    config_path = os.path.join(target_path, "config.yaml")
    env_cls, env_config, max_steps = load_config(config_path)

    # Build policy maps to eval and target population
    env = env_cls(env_config, spec_only=True)
    target_map = {id:id for id in env.possible_agents}

    if eval_map_spec is not None:
        eval_map = parse_map(eval_map_spec)
    else:
        eval_map = target_map
    
    # Load eval policies
    print(f"Loading policies from: {eval_path}")
    eval_policies = load_policies(eval_path, eval_map)

    # Load target policies
    print(f"Loading policies from: {target_path}")
    target_policies = load_policies(target_path, target_map)

    # Divide agent IDs into agents and adversaries
    agent_ids = env.possible_agents
    if adversaries is None:
        assert 2 == len(agent_ids), "must specify '--adversaries' if environment has >2 agents"
        agents = frozenset([agent_ids[0]])
        adversaries = frozenset([agent_ids[1]])
    else:
        adversaries = frozenset(adversaries)
        agents = frozenset([id for id in agent_ids if id not in adversaries])

    print(f"Evaluating Policies with {args.num_cpus} processes")

    # Generate cross-play JPC matrix
    cross_play = jpc(select_models(eval_policies, agents), 
                     select_models(target_policies, adversaries),
                     env_cls, 
                     env_config, 
                     max_steps, 
                     num_episodes,
                     num_cpus)

    # Generate self-play JPC matrix
    self_play = jpc(select_models(target_policies, agents), 
                    select_models(target_policies, adversaries),
                    env_cls,
                    env_config,
                    max_steps,
                    num_episodes,
                    num_cpus)
    
    return cross_play, self_play


if __name__ == '__main__':
    args = parse_args()

    # Limit CPU paralellism for policy inference
    torch.set_num_threads(args.num_cpus)

    eval_path = args.path[0]
    target_path = args.path[1] if len(args.path) > 1 else eval_path

    cross_play, self_play = cross_evaluate(eval_path, 
                                           target_path, 
                                           args.mapping, 
                                           args.adversaries, 
                                           args.num_episodes,
                                           args.num_cpus)

    print("\nCross-Play Matrix:")
    print(cross_play)

    print("\nSelf-Play Values:")
    print(np.diag(self_play))

    regrets = np.diag(self_play) - cross_play

    print("\nStatistics:")
    print(f"mean reward: {cross_play.mean()}")
    print(f"min reward: {cross_play.min(axis=1).mean()}")
    print(f"mean regret: {regrets.mean()}")
    print(f"max regret: {regrets.max(axis=1).mean()}") 
