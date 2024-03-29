#!/usr/bin/env python3
'''Computes Joint Policy Correlation (JPC) metrics for a collection of different configurations'''
import argparse
from collections import defaultdict
import io
import matplotlib
matplotlib.use('Agg')  # NOTE: This backend does not require access to an X-server, but is it compatible with the "--display"?
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

# TODO: We should probably automatically save the config if we are saving results to an alternative directory

# NOTE: We no longer define the "print_error" callback here, why is that?

def parse_args():
    parser = argparse.ArgumentParser("Computes the Joint Policy Correlation matrix (JPC) for a set of trained policies.")

    parser.add_argument("path", type=str, help="path to directory containing training results")

    # NOTE: How does this handle
    parser.add_argument("num_seeds", type=int, help="number of seeds used in the experiment")
    
    parser.add_argument("-o", "--output-path", type=str, default=None,
                        help="directory in which we should save results (defaults to experiment directory)")
    
    # NOTE: There is no "tag" option in the base JPC script, since this script stores multipe results files
    parser.add_argument("-t", "--tag", type=str, default="jpc",
                        help="sub-directory for JPC data (default: 'jpc')")

    parser.add_argument("-n", "--num-cpus", type=int, default=1,
                        help="the number of parallel worker processes to launch")
    parser.add_argument("-e", "--num-episodes", type=int, default=100,
                        help="the number of episodes to run for each policy combination")

    parser.add_argument("-m", "--mapping", nargs="+",
                        help="mapping from agent IDs to policy names (agent_id policy_name agent_id policy_name ...)")
    
    # NOTE: For enviornments with more that two agents, this allows us to organize agents into opposing teams
    parser.add_argument("-a", "--adversaries", nargs="+",
                        help="list of agent IDs corresponding to the 'adversary' team of agents")

    parser.add_argument("--title", type=str, default="Joint Policy Correlation",
                        help="title for figure")
    parser.add_argument("--min", type=float, help="min payoff value (for image rendering)")
    parser.add_argument("--max", type=float, help="max payoff value (for image rendering)")

    # NOTE: Additional option to hide the actual seed values in the figure
    parser.add_argument("--hide-seeds", action="store_true", 
                        help="print seed indexes, rather than values, when generating JPC plot")

    parser.add_argument("-d", "--display", action="store_true", help="display JPC matrix when finished")

    return parser.parse_args()

# NOTE: This method isn't present in the original JPC method
def get_dir(parent, name):
    '''Creates a sub-directory with the given name if it does not alrady exist.'''
    path = os.path.join(parent, name)
    
    if not os.path.exists(path):
        print(f"Creating results directory '{path}'")
        os.makedirs(path)
    else:
        print(f"Warning: overwriting any results in '{path}'")
    
    return path

# TODO: We wanted to add a different color scheme for these figures
def plot_matrix(matrix,
                seeds,  # NOTE: Now allows us to explicitly specify a list of random seed values
                path, 
                title, 
                min,
                max,
                size=300, 
                hide_seeds=False,  # NOTE: Again, has the option of hiding the actual seed values
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

        # NOTE: Has the option of just using seed ids rather than values
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
    plt.yticks(ticks, labels=reversed(labels))  # NOTE: Need to reverse labels since y-ticks start from the bottom (Broken in the original JPC script)

    ax = plt.gca()
    ax.grid(which='minor', color='k', linestyle='-', linewidth=2)

    plt.title(title, fontsize=14)
    plt.xlabel("seeds", fontsize=16)
    plt.ylabel("seeds", fontsize=16)
    plt.savefig(path, bbox_inches="tight")

    if disp:
        plt.show(block=True)


def load_populations(path, 
                     policy_map):  # NOTE: Now allows for arbitrary seeds
    populations = defaultdict(dict)  # NOTE: Dictionary of dictionaries
    config_path = os.path.join(path, "config.yaml")
    
    if not os.path.isfile(config_path):
        raise ValueError(f"Config File '{config_path}' does not exist")

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # NOTE: We probably shouldn't need this logic here
    if "trainer" not in config:  # NOTE: Needed when the top-level config dict has a single entry with the experiment name (this is the default)
        config = list(config.values())[0]

    trainer_config = config.get("config", {})

    max_steps = config.get("max_steps", 100)  # NOTE: Original JPC script doesn't check this, could that be a problem?

    env_name = trainer_config.get("env")
    env_cls = get_env_class(env_name)
    
    env_config = trainer_config.get("env_config", {})
    env_config = trainer_config.get("env_eval_config", env_config)  # NOTE: If available, use the evaluation env configuration

    env = env_cls(env_config, spec_only=True)

    if policy_map is None:
        map = {}
        for policy_id in env.possible_agents:  # NOTE: Previously the "possible_agents" field did not exist
            map[policy_id] = policy_id
    else:
        map = {}  # NOTE: Need two policy maps (why?)

        for idx in range(0, len(policy_map), 2):
            agent_id = policy_map[idx]
            policy_id = policy_map[idx + 1]
            map[agent_id] = policy_id  # NOTE: Since all agent IDs are now strings, doesn't need the "isnumeric()" check
        
        for agent_id in env.possible_agents:  # NOTE: Now requires that the policy mapping be "complete"
            assert agent_id in map, f"no policy map given for '{agent_id}'"

    # NOTE: Unlike the original JPC script, this script doesn't need to know how many seeds there are
    exp = re.compile("seed_(\d+)")  # NOTE: Now uses regular expressions to extract seed values
    for obj in os.listdir(path):  # NOTE: Checks each sub-directory to see if it contains a run
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
    
    return populations, env.possible_agents, env_cls, env_config, max_steps  # NOTE: No longer returns the entire traner config

# NOTE: For some reason we swapped the order of the "permutation()" and "evaluate()" functions

# NOTE: This method is essentially unchanged from the 
def permutations(num_teams, # NOTE: We now refer to permutations of "teams", to handle cases with more than two agents gracefully (eg. the particle environments)
                 num_populations):
        num_permutations = num_populations ** num_teams  # NOTE: The script itself only supports two teams right now
        for index in range(num_permutations):
            permutation = [0] * num_teams  # NOTE: A permutation is a list of population IDs, one for each "team", or coherent group of agents
            idx = index
            for id in range(num_teams):
                permutation[id] = idx % num_populations  # NOTE: ID for the current team can be thought of as least-significant digit of the index
                idx = idx // num_populations
            yield permutation  # NOTE: Returns a list, not a tuple


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

    batch = sample(env, policies, num_episodes, max_steps)  # NOTE: Not sure why we capture the batch itself here, but both this and the original JPC script are compatible with the new statistics API
    return batch.statistics()  # NOTE: Why did we switch to the new API?

# NOTE: Didn't get rid of this function, just moved it from the top of the script to be closer to where it is used
def print_error(error):
    '''Error callback for python multiprocessing'''
    traceback.print_exception(type(error), error, error.__traceback__, limit=5)

# NOTE: Some of the machinery for collecting policies could be moved outside this method it seems
def cross_evaluate(populations, # NOTE: Each "population" refers to the set of policies generate by a single random seed
                   agent_ids,  # NOTE: Why did we add this parameter?
                   env_cls, 
                   env_config, 
                   max_steps, # NOTE: Now we pass the environment class and config separately
                   num_cpus, 
                   num_episodes,
                   adversaries):

    # NOTE: Used as a handle for single-threaded execution
    class dummy_async:  # NOTE: We could use something like this as part of "train.py"

        def __init__(self, result):
            self._result = result
        
        def get(self):
            return self._result


    if num_cpus > 1:  #NOTE: We are using the Pool class wrong here as well
        pool = Pool(num_cpus)

    population_ids = list(populations.keys())  # NOTE: Basically just getting the random seeds, as strings
    num_populations = len(population_ids)  # NOTE: Basically just getting the number of seeds

    num_agents = len(agent_ids)  # NOTE: Gets the number of individual agents in the environemt (not the number of teams)
    assert 2 <= num_agents, "environment must contain at least 2 agents for cross evaluation"

    # NOTE: The new script supports dividing agents into logical "teams"
    if adversaries is None:
        assert 2 == num_agents, "must specify adversary team with '--adversaries' if environment has more than 2 agents"
        adversaries = frozenset([agent_ids[1]])
    else:
        adversaries = frozenset(adversaries)

    threads = {}  # NOTE: We use the general-purpose "permutations" method, but do we need this (since always rank-2)?
    for permutation in permutations(2, num_populations):  # NOTE: We now use the number of "teams" rather than indivudal agents
        models = {}

        # NOTE: Now we divide up teams "within" the run process
        for agent_id in agent_ids:  # NOTE: This process has changed from the original script to accomodate the idea of teams
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
        jpc[idx] = stats["reward_mean"]  # NOTE: The statistic being evaluated is hard-coded

    return jpc, population_ids  # NOTE: Presumably the "population_ids" are now the seeds (as strings)

# NOTE: This method is altogether new
def jpc_stats(jpc):
    stats = {}

    # NOTE: Mean over all combinations
    stats["mean"] = jpc.mean().item()

    # NOTE: Mean over off-diagonal entries
    stats["cross_mean"] = (jpc - np.diag(np.diag(jpc))).mean().item()
    
    # NOTE: Average worst case performance (not regret) over all "protagonists"
    stats["agent_min"] = jpc.min(1).mean().item()

    # NOTE: Average worst case performance over all "adeversaries"
    stats["adeversary_min"] = jpc.min(0).mean().item()

    # NOTE: Computes average worst-case performance for the "symmetrized" JPC, which may not make sense if the game is asymmetric
    stats["symmetric_min"] = ((jpc + jpc.T) / 2).min(1).mean().item()

    return stats


if __name__ == '__main__':
    args = parse_args()

    # Limit CPU paralellism for policy inference
    torch.set_num_threads(args.num_cpus)  # NOTE: Need to fix this, same issue as the training script

    if args.output_path is not None:  # NOTE: Minor point, but the "tag" parameter is a little confusing to the user
        path = get_dir(args.output_path, args.tag)
    else:
        path = get_dir(args.path, args.tag)

    print(f"Loading policies from: {args.path}")
    populations, agent_ids, env_cls, env_config, max_steps = load_populations(args.path, args.mapping)  # NOTE: Again, seeds are now extracted from directory names

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

    # NOTE: We now save scalar statistics to a yaml file as well (more human readable)
    for key, value in stats.items():
        print(f"    {key}: {value}")

    with open(os.path.join(path, "jpc_stats.yaml"), 'w') as f:
        yaml.dump(stats, f)

    print(f"\nrendering JPC tensor")  # NOTE: Need to check that "--display" still works with the "Agg" backend
    plot_matrix(
        matrix=jpc,
        seeds=seeds,
        path=os.path.join(path, "jpc.png"),
        title=args.title,
        min=args.min,
        max=args.max,
        hide_seeds=args.hide_seeds,
        disp=args.display)    
