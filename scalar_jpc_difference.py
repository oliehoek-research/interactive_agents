#!/usr/bin/env python3

# NOTE: This is Mert's script, so I don't know much about how it works.  Likely replicates some of the functionality of the "cross_evaluate.py" script

'''Computes Joint Policy Correlation (JPC) metrics for a collection of different configurations'''
import argparse
from collections import defaultdict
import io
import matplotlib.pyplot as plt
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

# NOTE: This todo was copied over with the "scalar_jpc.py" script
# TODO: We should probably automatically save the config if we are saving results to an alternative directory

def parse_args():
    parser = argparse.ArgumentParser("Computes the Joint Policy Correlation matrix (JPC) for a regret game trained Alice paired with a self-play trained Bob.")

    # NOTE: New parameters; we now have two experiment directories, one with runs from the algorithm being evaluated, and one with runs from self-play (or presumed to be from self-play)
    parser.add_argument("path_rg", type=str, help="(regret game!) path to directory containing Alice training results")
    parser.add_argument("path_sp", type=str, help="(self play!) path to directory containing Bob training results")

    # NOTE: Defaults to the "regret game" directory, because this is presumably the algorithm we are actually evaluating
    parser.add_argument("-o", "--output-path", type=str, default=None,
                        help="directory in which we should save results (defaults to regret game experiment directory)")
    parser.add_argument("-t", "--tag", type=str, default="jpc",
                        help="sub-directory for JPC data (default: 'jpc')")

    parser.add_argument("-n", "--num-cpus", type=int, default=1,
                        help="the number of parallel worker processes to launch")
    parser.add_argument("-e", "--num-episodes", type=int, default=100,
                        help="the number of episodes to run for each policy combination")

    parser.add_argument("-m", "--mapping", nargs="+",
                        help="mapping from agent IDs to policy names (agent_id policy_name agent_id policy_name ...)")
    
    # NOTE: This is new, how is this used?
    parser.add_argument("-s", "--sp_agents", nargs="+",
                    help="list of agent IDs to take from self-play results")

    parser.add_argument("-a", "--adversaries", nargs="+",
                        help="list of agent IDs corresponding to the 'adversary' team of agents")

    parser.add_argument("--title", type=str, default="Joint Policy Correlation",
                        help="title for figure")
    parser.add_argument("--min", type=float, help="min payoff value (for image rendering)")
    parser.add_argument("--max", type=float, help="max payoff value (for image rendering)")
    parser.add_argument("--hide-seeds", action="store_true", 
                        help="print seed indexes, rather than values, when generating JPC plot")

    parser.add_argument("-d", "--display", action="store_true", help="display JPC matrix when finished")

    return parser.parse_args()


def get_dir(parent, name):
    '''Creates a sub-directory with the given name if it does not alrady exist.'''
    path = os.path.join(parent, name)
    
    if not os.path.exists(path):
        print(f"Creating results directory '{path}'")
        os.makedirs(path)
    else:
        print(f"Warning: overwriting any results in '{path}'")
    
    return path

# NOTE: This matix plotting function is unchanged
def plot_matrix(matrix,
                seeds,
                path, 
                title, 
                min,
                max,
                size=300, 
                hide_seeds=False, 
                disp=False): # NOTE: Only works for 2D JPC tensors (matrices)
    '''Generates a color-coded image representing the JPC matrix'''
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

        if hide_seeds:
            labels.append(idx)
        else:
            labels.append(seeds[idx])
        
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
    plt.yticks(ticks, labels=reversed(labels))  # NOTE: Need to reverse labels since y-ticks start from the bottom

    ax = plt.gca()
    ax.grid(which='minor', color='k', linestyle='-', linewidth=2)

    plt.title(title, fontsize=14)
    plt.xlabel("seeds", fontsize=16)
    plt.ylabel("seeds", fontsize=16)
    plt.savefig(path, bbox_inches="tight")

    if disp:
        plt.show(block=True)

# NOTE: This method is also unchanged
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

# NOTE: This is new, may just be a slightly modified version of "load_populations"
def load_populations_selfvsregret(path_rg, path_sp, # NOTE: Strange, uses both paths (maybe "load_populations()" is ignored?)
                     policy_map, sp_agents):  # NOTE: Need to make sure this can handle arbitrary seeds
    populations = defaultdict(dict)  # NOTE: Dictionary of dictionaries

    # NOTE: Seems to have copied several lines to have two versions, one for the regret game, and one for self-play
    rg_config_path = os.path.join(path_rg, "config.yaml")
    sp_config_path = os.path.join(path_sp, "config.yaml")

    # NOTE: Makes sure that both files exist
    if not os.path.isfile(rg_config_path) or not os.path.isfile(sp_config_path):
        raise ValueError(f" Config File '{rg_config_path}' and/or '{sp_config_path}' do not exist")

    with open(rg_config_path, "r") as f:
        rg_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(sp_config_path, "r") as f:
        sp_config = yaml.load(f, Loader=yaml.FullLoader)

    if "trainer" not in rg_config:  # NOTE: Needed when the top-level config dict has a single entry with the experiment name (this is the default)
        rg_config = list(rg_config.values())[0]
    if "trainer" not in sp_config:  # NOTE: Needed when the top-level config dict has a single entry with the experiment name (this is the default)
        sp_config = list(sp_config.values())[0]


    rg_trainer_config = rg_config.get("config", {})
    sp_trainer_config = sp_config.get("config", {})

    # NOTE: Issues a warning if max steps doesn't match between configs, but allows us to continue
    rg_max_steps = rg_config.get("max_steps", 100)
    sp_max_steps = sp_config.get("max_steps", 100)
    if rg_max_steps != sp_max_steps:
        print(f"The regret game and self-play maximum evaluation steps are not the same ('{rg_max_steps}' != '{sp_max_steps}'). I will use the minimum of two.")
    max_steps = min(rg_max_steps, sp_max_steps)

    # NOTE: Checks that the environment classes are the same, otherwise throws an exception
    assert rg_trainer_config.get("env") == sp_trainer_config.get("env"), "The regret game and self-play environments are not the same"
    env_name = rg_trainer_config.get("env")
    env_cls = get_env_class(env_name)
    
    #TODO: This is using the regret game trainer's environment configuration. 
    env_config = rg_trainer_config.get("env_config", {})
    env_config = rg_trainer_config.get("env_eval_config", env_config)  # NOTE: If available, use the evaluation env configuration

    env = env_cls(env_config, spec_only=True)

    # NOTE: Seems to require that we ALWAYS provide a policy map
    # NOTE: Space of agents divided into those that will come from the regret game, and those that will come from self-play
    rg_map = {}  # NOTE: Need two policy maps
    sp_map = {}
    map = {}
    for idx in range(0, len(policy_map), 2):
        agent_id = policy_map[idx]
        policy_id = policy_map[idx + 1]
        map[agent_id] = policy_id  # NOTE: Maintains a global map as well - is this used?
        if agent_id in sp_agents:
            sp_map[agent_id] = policy_id
        else:
            rg_map[agent_id] = policy_id
    
    for agent_id in env.possible_agents:
        assert agent_id in map, f"no policy map given for '{agent_id}'"

    # NOTE: Repeats the loading process for the regret game and self-play policies - kind of a hack
    # NOTE: Adds both self-play and regret-game policies to the same "population" dictionary, a little odd if they have differing numbers of seeds (which wouldn't be unusual)
    #Load regret game policies
    exp = re.compile("seed_(\d+)")
    print("Loading regret game policies")
    for obj in os.listdir(path_rg):
        match = exp.match(obj)
        if match is not None:
            seed = match.group(1)  # NOTE: Okay to leave the seed as a string
            seed_path = os.path.join(path_rg, obj, "policies")

            if os.path.isdir(seed_path):
                print(f"\nloading policies from: {seed_path}")
                
                for agent_id, policy_id in rg_map.items():
                    policy_path = os.path.join(seed_path, f"{policy_id}.pt")
                    print(f"loading: {policy_path}")

                    if os.path.isfile(policy_path):
                        model = torch.jit.load(policy_path)
                        populations[seed][agent_id] = model  # NOTE: could end up loading the policy multiple times - not ideal
                    else:
                        raise FileNotFoundError(f"seed {seed} does not define policy '{policy_id}'")
    #Load self-play policies
    exp = re.compile("seed_(\d+)")
    print("Loading self-play policies")
    for obj in os.listdir(path_sp):
        match = exp.match(obj)
        if match is not None:
            seed = match.group(1)  # NOTE: Okay to leave the seed as a string
            seed_path = os.path.join(path_sp, obj, "policies")

            if os.path.isdir(seed_path):
                print(f"\nloading policies from: {seed_path}")
                
                for agent_id, policy_id in sp_map.items():
                    policy_path = os.path.join(seed_path, f"{policy_id}.pt")
                    print(f"loading: {policy_path}")

                    if os.path.isfile(policy_path):
                        model = torch.jit.load(policy_path)
                        populations[seed][agent_id] = model  # NOTE: could end up loading the policy multiple times - not ideal
                    else:
                        raise FileNotFoundError(f"seed {seed} does not define policy '{policy_id}'")

    return populations, env.possible_agents, env_cls, env_config, max_steps

# NOTE: This seems to be unchanged
def permutations(num_teams, 
                 num_populations):
        num_permutations = num_populations ** num_teams  # NOTE: Right now we only support two teams
        for index in range(num_permutations):
            permutation = [0] * num_teams
            idx = index
            for id in range(num_teams):
                permutation[id] = idx % num_populations
                idx = idx // num_populations
            yield permutation

# NOTE: Again, completely unchanged
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
    '''Error callback for python multiprocessing'''
    traceback.print_exception(type(error), error, error.__traceback__, limit=5)

# NOTE: This method also seems to be unchanged - how do they compute regret then?
def cross_evaluate(populations, 
                   agent_ids,
                   env_cls, 
                   env_config, 
                   max_steps, 
                   num_cpus, 
                   num_episodes,
                   adversaries):

    # NOTE: Used as a handle for single-threaded execution
    class dummy_async:

        def __init__(self, result):
            self._result = result
        
        def get(self):
            return self._result


    if num_cpus > 1:
        pool = Pool(num_cpus)

    population_ids = list(populations.keys())
    num_populations = len(population_ids)

    num_agents = len(agent_ids)
    assert 2 <= num_agents, "environment must contain at least 2 agents for cross evaluation"

    if adversaries is None:
        assert 2 == num_agents, "must specify adversary team with '--adversaries' if environment has more than 2 agents"
        adversaries = frozenset([agent_ids[1]])
    else:
        adversaries = frozenset(adversaries)

    threads = {}
    for permutation in permutations(2, num_populations):
        models = {}

        for agent_id in agent_ids:
            if agent_id in adversaries:
                seed = population_ids[permutation[1]]
            else:
                seed = population_ids[permutation[0]]

            models[agent_id] = populations[seed][agent_id]

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

    jpc = np.zeros((num_populations, num_populations))
    for idx, thread in threads.items():
        stats = thread.get()
        jpc[idx] = stats["reward_mean"]

    return jpc, population_ids


# NOTE: Again, unchanged - this may not be the version of the script that Mert is using
def jpc_stats(jpc):
    stats = {}
    stats["mean"] = jpc.mean().item()
    stats["cross_mean"] = (jpc - np.diag(np.diag(jpc))).mean().item()
    
    stats["agent_min"] = jpc.min(1).mean().item()
    stats["adeversary_min"] = jpc.min(0).mean().item()

    stats["symmetric_min"] = ((jpc + jpc.T) / 2).min(1).mean().item()

    return stats


if __name__ == '__main__':
    args = parse_args()

    # Limit CPU paralellism for policy inference
    torch.set_num_threads(args.num_cpus)

    if args.output_path is not None:
        path = get_dir(args.output_path, args.tag)
    else:
        path = get_dir(args.path_rg, args.tag)

    print(f"Loading regret game policies from: {args.path_rg} and self-play policies from: {args.path_sp}")
    populations, agent_ids, env_cls, env_config, max_steps = load_populations_selfvsregret(args.path_rg, args.path_sp, args.mapping, args.sp_agents)

    # NOTE: Mert first loads and performs cross-evaluation between regret-game policies and self-play policies, computes the JPC and relevant statistics
    print(f"Evaluating Policies with {args.num_cpus} processes")
    jpc_selfvsregret, seeds = cross_evaluate(populations=populations,
                                agent_ids=agent_ids,
                                env_cls=env_cls,
                                env_config=env_config,
                                max_steps=max_steps,
                                num_cpus=args.num_cpus,
                                num_episodes=args.num_episodes,
                                adversaries=args.adversaries)
    print("\nJPC Tensor for Self vs Regret:")
    print(jpc_selfvsregret)
    stats_selfvsregret = jpc_stats(jpc_selfvsregret)
    print("statistics:")

    for key, value in stats_selfvsregret.items():
        print(f"    {key}: {value}")

    # NOTE: Then we compute the JPC and stats just for the self-play populations alone
    print(f"Loading self-play policies from: {args.path_sp}")
    populations, agent_ids, env_cls, env_config, max_steps = load_populations(args.path_sp, None)
    
    jpc_self, seeds = cross_evaluate(populations=populations,
                                agent_ids=agent_ids,
                                env_cls=env_cls,
                                env_config=env_config,
                                max_steps=max_steps,
                                num_cpus=args.num_cpus,
                                num_episodes=args.num_episodes,
                                adversaries=args.adversaries)
    print("\nJPC Tensor for Self-Play:")
    print(jpc_self)
    stats_self = jpc_stats(jpc_self)
    print("statistics:")

    for key, value in stats_self.items():
        print(f"    {key}: {value}")

    # NOTE: This is where the comparison between the two JPCs is actually done
    print("\n Difference between Self-Play and Self vs Regret:")

    # NOTE: Directly computes the difference between scalar statistics, which isn't always meaningful 
    diff_dict = {key: stats_selfvsregret[key] - stats_self[key] for key in stats_selfvsregret.keys()}
    for key, value in diff_dict.items():
        print(f"    {key}: {value}")


    #np.save(os.path.join(path, "jpc_selfvsregret.npy"), jpc, allow_pickle=False)

    with open(os.path.join(path, "jpc_difference_stats.yaml"), 'w') as f:
        yaml.dump(diff_dict, f)

    # NOTE: Renders the JPC differential matrix, but doesn't analyze it - this can only work if the regret-game and self-play have the same number of seeds
    print(f"\nrendering JPC tensor")
    plot_matrix(
        matrix=jpc_selfvsregret - jpc_self,
        seeds=seeds,
        path=os.path.join(path, "jpc_diff.png"),
        title=args.title,
        min=args.min,
        max=args.max,
        hide_seeds=args.hide_seeds,
        disp=args.display) 




