"""
Trainer class for the simplified action 
decoder (SAD) as described in the paper "Simplified 
action decoder for deep multi-agent reinforcement 
learning." (Hu et al. 2019).

Almost identical to the "independent" trainer, but allows
each player to see the other's exploration and exploitation action
"""
import numpy as np
import torch

from interactive_agents.envs import get_env_class
from interactive_agents.training.learners.r2d2_sad import R2D2_SAD
from interactive_agents.sampling import sample
from interactive_agents.stopwatch import Stopwatch

"""Methods and utilities for sampling experiences"""
from collections import defaultdict
import numpy as np

import torch

class Batch(dict):  # NOTE: Seems to organize batches by the "policy" they are associated with, rather than the "agent"
    """A dictionary object representing a multi-agent experience batch"""

    OBS = "obs"
    NEXT_OBS = "next_obs"
    ACTION = "actions"
    REWARD = "rewards"
    DONE = "dones"

    def __init__(self, batches={}, episodes=0, timesteps=0):
        super(Batch, self).__init__(batches)
        self._episodes = episodes
        self._timesteps = timesteps  # NOTE: What is this used for?

    @property
    def episodes(self):
        return self._episodes

    @property
    def timesteps(self):
        return self._timesteps

    def policy_batches(self, policy_ids):  # NOTE: Seems to return a batch associated with a single policy
        batch = {}
        for pid in policy_ids:
            if pid in self:
                batch[pid] = self[pid]
        
        return Batch(batch, self._episodes, self._timesteps)

    def policy_batch(self, policy_id):  # NOTE: Just a utility that returns a single policy
        return self.policy_batches([policy_id])

    def extend(self, batch):
        for policy_id, episodes in batch.items():
            if policy_id not in self:
                self[policy_id] = []
            
            self[policy_id].extend(episodes)
        
        self._episodes += batch.episodes
        self._timesteps += batch.timesteps

    def statistics(self, alt_names=None):  # NOTE: Statistics computed internally by the batch itself
        if alt_names is None:
            alt_names = {pid:pid for pid in self.keys()}
        
        stats = {
            "reward_mean": 0,
            "reward_max": -np.inf,
            "reward_min": np.inf
        }

        for policy_id, agent_batch in self.items():
            if policy_id in alt_names:
                logging_id = alt_names[policy_id]
                r_mean = 0
                r_max = -np.inf
                r_min = np.inf

                for episode in agent_batch:
                    episode_reward = np.sum(episode[Batch.REWARD])
                    r_mean += episode_reward
                    r_max = max(r_max, episode_reward)
                    r_min = min(r_min, episode_reward)

                r_mean /= len(agent_batch)
                stats[str(logging_id) + "/reward_mean"] = r_mean
                stats[str(logging_id) + "/reward_max"] = r_max
                stats[str(logging_id) + "/reward_min"] = r_min

                stats["reward_mean"] += r_mean
                stats["reward_max"] = max(r_max, stats["reward_max"])
                stats["reward_min"] = min(r_min, stats["reward_min"])

        stats["episodes"] = self._episodes
        stats["timesteps"] = self._timesteps

        return stats


class BatchBuilder:  # NOTE: Basically a batch with an internal state corresponding to the current episode
    """Used to record a multi-agent batch during sampling"""

    def __init__(self):

        # NOTE: A dictionary of episodes organized by the policy they need to be sent to for training
        self._policy_batches = defaultdict(list)

        self._episodes = 0
        self._timesteps = 0
        
        # NOTE: What is this?
        self._agent_episodes = None
        self._policy_map = None  # NOTE: Needed since multiple agents may be controlled by a single policy
        self._episode_steps = 0

    def _store_episode(self):
        for agent_id, episode in self._agent_episodes.items():
            d = {}
            d[Batch.ACTION] = np.asarray(episode.pop(Batch.ACTION), np.int64)
            d[Batch.REWARD] = np.asarray(episode.pop(Batch.REWARD), np.float32)
            d[Batch.DONE] = np.asarray(episode.pop(Batch.DONE), np.float32)

            obs_t = np.asarray(episode.pop(Batch.OBS), np.float32)
            d[Batch.OBS] = obs_t[:-1]
            d[Batch.NEXT_OBS] = obs_t[1:]

            for key, value in episode.items():  # NOTE: This seems to handle policy-specific outputs, such as internal state
                d[key] = np.asarray(value, np.float32)

            if self._policy_map is not None:
                self._policy_batches[self._policy_map[agent_id]].append(d)
            else:
                self._policy_batches[agent_id].append(d)

        self._episodes += 1
        self._timesteps += self._episode_steps

    # NOTE: Call this before adding any data, need to provide the initial observation for each player
    def start_episode(self, initial_obs, policy_map=None):

        # NOTE: If we are still recording an existing episode, we must process its data first
        assert self._agent_episodes is None, "Must call 'end_episode()' first to end current episode"

        # NOTE: A dictionary of dictionaries (what gets stored here?)
        self._agent_episodes = defaultdict(lambda: defaultdict(list))

        # NOTE: The policy map can be changed from one episode to the next (for example, if we want to swap agent IDs)
        self._policy_map = policy_map
        self._episode_steps = 0

        # NOTE: Add initial observation for each agent
        for agent_id, obs in initial_obs.items():
            self._agent_episodes[agent_id][Batch.OBS].append(obs)

    def end_episode(self):
        if self._episode_steps > 0:
            self._store_episode()
        
        self._agent_episodes = None

    def step(self, obs, actions, rewards, dones, fetches):
        assert self._agent_episodes is not None, "Must call 'start_episode()' first to start new episode"
        for agent_id in obs.keys():
            episode = self._agent_episodes[agent_id]

            episode[Batch.OBS].append(obs[agent_id])
            episode[Batch.ACTION].append(actions[agent_id])
            episode[Batch.REWARD].append(rewards[agent_id])
            episode[Batch.DONE].append(dones[agent_id])
            
            for key, value in fetches[agent_id].items():                
                episode[key].append(value)
        
        self._episode_steps += 1

    def build(self):
        return Batch(self._policy_batches, self._episodes, self._timesteps)


# NOTE: Move this into the SADTrainer class as well
# NOTE: Need to update to the latest PettingZoo interface

# NOTE: PettingZoo does not define a vectored environment wrapper.  Moreover, 
# vectorizing multi-agent environments is tricky, since the set of agents in
# an environment can change over time.


# TODO: Enable support for multiple policies maintained by a single actor (needed for SAD, CC methods)
def sample(env, policies, num_episodes=128, max_steps=1e6, policy_fn=None):  # NOTE: Accepts a single environment, rather than a batch
    """Generates a batch of episodes using the given policies"""
    batch = BatchBuilder()
    
    for _ in range(num_episodes):

        # Initialize episode and episode batch
        obs = env.reset()
        current_step = 0

        agents = {}
        policy_map = {}
        dones = {}
        for agent_id in obs.keys():
            if policy_fn is not None:
                policy_id = policy_fn(agent_id)
            else:
                policy_id = agent_id
            
            agents[agent_id] = policies[policy_id].make_agent()
            policy_map[agent_id] = policy_id
            dones[agent_id] = False

        batch.start_episode(obs, policy_map)

        # Rollout episode
        while current_step < max_steps and not all(dones.values()):
            actions = {}
            fetches = {}
            for agent_id, ob in obs.items():
                actions[agent_id], fetches[agent_id] = agents[agent_id].act(ob)

            obs, rewards, dones, _ = env.step(actions)

            batch.step(obs, actions, rewards, dones, fetches)  # NOTE: All data added on a 'per-agent' basis
            current_step += 1
    
        # TODO: Allow actors to do additional postprocessing
        batch.end_episode()
    
    return batch.build()

# NOTE: LSTMs and Decision Transformers may benefit from different memory
# layouts.  Since everything is being done in memory here, there is no
# need to "pack" data before passing it to the learner.  Pass the raw
# numpy arrays to the learner.

# NOTE: Challenging to manage environments with variable 
# numbers of agents and variable-length episodes.
class VectorWrapper:

    def __init__(self, envs, max_steps=np.infty):
        self._envs = envs
        self._mx_steps = max_steps

        self.observation_spaces = envs[0].observation_spaces
        self.action_spaces = envs[0].actions_spaces
        self.possible_agents = envs[0].possible_agents

        self._step = None
        self._dones = None


class SADTrainer:

    def __init__(self, config, seed=0, device="cpu", verbose=False):
        self._iteration_episodes = config.get("iteration_episodes", 128)
        self._num_envs = config.get("num_envs", 32)
        self._max_steps = config.get("max_steps", np.inf)  # NOTE: Maximum episode length (what environments do we need this for?)
        
        self._eval_iterations = config.get("eval_iterations", 10)
        self._eval_episodes = config.get("eval_episodes", 64)
        
        self._seed = seed
        self._verbose = verbose

        # Seed random number generators
        np.random.seed(seed)  # NOTE: Should switch to creating a new RNG object for each 
        torch.manual_seed(seed)

        # Get environment class and config
        if "env" not in config:  # NOTE: Not defult environment
            raise ValueError("must specify environment through 'env' parameter")

        env_name = config.get("env")
        env_config = config.get("env_config", {})
        env_eval_config = config.get("env_eval_config", env_config)
        env_cls = get_env_class(env_name)

        # Build environments
        self._envs = [env_cls(env_config) for _ in range(self._num_envs)]
        self._eval_envs = [env_cls(env_eval_config) for _ in range(self._num_envs)]

        # Initialize Learner
        first_agent = self._envs[0].possible_agents[0]
        obs_space = self._envs[0].observation_space(first_agent)
        action_space = self._envs[0].action_space(first_agent)

        # TODO: Check that all observation spaces are identical
        learner_config = config.get("learner_config", {})
        self._learner = R2D2_SAD(obs_space, action_space, learner_config, device)

        # Statistics and timers
        self._global_timer = Stopwatch()
        self._sampling_timer = Stopwatch()
        self._learning_timer = Stopwatch()

        self._timesteps_total = 0  # NOTE: Keep track of the total number of timesteps and episodes generated
        self._episodes_total = 0

        self._current_iteration = 0  # NOTE: Keeps track of the total number of iterations

    # NOTE: Makes the simplifying assumption that 
    def _get_actions(self, obs, actions, states):
        pass

    # NOTE: In newer versions of Gym (and PettingZoo) there has been a move 
    # towards treating experiences as one long sequence of data, rather than
    # as discrete episodes.  Learner's can use the "truncated" or "terminated"
    # signals to determine when episodes begin or end.

    # NOTE: Figure out how to do vectored SAD sampling before doing batching
    def _sample_batch(self, num_episodes):
        for _ in range(num_episodes):

            # Reset each environment and initial states
            obs = []
            dones = []

            


            # Initialize episode and episode batch
            obs = env.reset()  # NOTE: Reset the single environment
            current_step = 0

            # NOTE: Initialized agents (all agents use the same policy, so we don't need this)
            agents = {}
            policy_map = {}
            dones = {}
            for agent_id in obs.keys():
                if policy_fn is not None:
                    policy_id = policy_fn(agent_id)
                else:
                    policy_id = agent_id
                
                agents[agent_id] = policies[policy_id].make_agent()
                policy_map[agent_id] = policy_id
                dones[agent_id] = False

            # NOTE: For the moment, assume the set of agents is fixed

            # NOTE: For inference, we should combine environment and agent IDs into a single batch dimension

            # NOTE: Need to tell the batch builder to start a new set of trajectories
            batch.start_episode(obs, policy_map)

            # NOTE: Rollout a single episode
            while current_step < max_steps and not all(dones.values()):
                actions = {}
                fetches = {}  # NOTE: Agents are allowed to return arbitrary additional information
                for agent_id, ob in obs.items():  # NOTE: Evaluate each policy independently
                    actions[agent_id], fetches[agent_id] = agents[agent_id].act(ob)

                obs, rewards, dones, _ = env.step(actions)

                # NOTE: Adds dictionaries with entries for each agent, need a way to build batches for multiple episodes at once
                batch.step(obs, actions, rewards, dones, fetches)  # NOTE: All data added on a 'per-agent' basis

                current_step += 1
        
            # NOTE: Have to tell the batch-builder to accumulate the latest trajectories
            # TODO: Allow actors to do additional postprocessing
            batch.end_episode()
        
        return batch.build()

    def train(self):
        self._global_timer.start()
        stats = {} 

        # NOTE: Move parameters from learner objects to training 
        # policies.  Since we haven't been running distributed experiments,
        # there is no reason to keep this.

        # Update sampling policies
        for id, learner in self._learners.items():
            self._training_policies[id].update(learner.get_actor_update())
        
        # Collect training batch and batch statistics
        sampling_time = self._sampling_timer.elapsed()
        self._sampling_timer.start()
        training_batch = sample(self._env, self._training_policies,
             self._iteration_episodes, self._max_steps)  # NOTE: Generate a new batch of experiences
        self._sampling_timer.stop()
        sampling_time = self._sampling_timer.elapsed() - sampling_time + 1e-6

        # NOTE: Use the Batch object itself to accumulate statistics
        for key, value in training_batch.statistics().items():
            stats["sampling/" + key] = value

        # NOTE: Calculate sampling throughputs
        stats["sampling/episodes_per_s"] = training_batch.episodes / sampling_time
        stats["sampling/timesteps_per_s"] = training_batch.timesteps / sampling_time

        # NOTE: The batch is essentially a dictionary of episode arrays organized by policy IDs

        # Train learners on new training batch
        for id, episodes in training_batch.items():
            self._learning_timer.start()
            learning_stats = self._learners[id].learn(episodes)  # NOTE: Do a learning update for each policy
            self._learning_timer.stop()

            for key, value in learning_stats.items():
                stats[f"learning/{id}/{key}"] = value

        # Increment iteration
        self._current_iteration += 1

        # NOTE: Do evaluation if need be at this iteration
        # Do evaluation if needed (update eval policies first)
        if self._current_iteration % self._eval_iterations == 0:
            for id, learner in self._learners.items():
                self._eval_policies[id].update(learner.get_actor_update(eval=True))

            eval_batch = sample(self._eval_env, 
                self._eval_policies, self._eval_episodes, self._max_steps)

            for key, value in eval_batch.statistics().items():
                stats["eval/" + key] = value
            
            if self._verbose:
                print(f"\n\nEVALUATION, seed {self._seed}, iteration {self._current_iteration}")
                print(f"mean eval reward: {stats['eval/reward_mean']}")

        # Accumulate global statistics
        self._global_timer.stop()

        self._episodes_total += training_batch.episodes
        self._timesteps_total += training_batch.timesteps

        stats["global/episodes_total"] = self._episodes_total
        stats["global/timesteps_total"] = self._timesteps_total

        stats["global/total_time_s"] = self._global_timer.elapsed()
        stats["global/sampling_time_s"] = self._sampling_timer.elapsed()
        stats["global/learning_time_s"] = self._learning_timer.elapsed()

        # Print iteration stats
        if self._verbose:
            print(f"\n\nSEED {self._seed}, ITERATION {self._current_iteration}")
            print(f"total episodes: {self._episodes_total}")
            print(f"total timesteps: {self._timesteps_total}")
            print(f"mean sampling reward: {stats['sampling/reward_mean']}")

        return stats
    
    # NOTE: For the moment, don't bother exporting policies
    def export_policies(self):
        return {}
