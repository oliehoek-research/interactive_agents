# NOTE: When we figure out how to do generic league architectures, we will likely put this file in a "self-play" subdirectory

"""Simple Trainer that runs independent learners for each agent"""
from collections import defaultdict
import numpy as np
import torch

from interactive_agents.envs import get_env_class
from interactive_agents.training.learners import get_learner_class
from interactive_agents.sampling import sample, FrozenPolicy, Batch
from interactive_agents.stopwatch import Stopwatch


class SelfPlayTrainer:

    def __init__(self, config, seed=0, device="cpu", verbose=False):  # NOTE: Need to make sure we use these new parameters
        self._round_iterations = config.get("round_iterations", 100)
        self._burn_in_iterations = config.get("burn_in_iterations", 100)
        self._weight_decay = config.get("weight_decay", 0.0)
        
        self._iteration_episodes = config.get("iteration_episodes", 128)
        self._max_steps = config.get("max_steps", 100)

        self._eval_iterations = config.get("eval_iterations", 10)
        self._eval_episodes = config.get("eval_episodes", 64)

        self._seed = seed
        self._device = device
        self._verbose = verbose

        # Seed random number generators
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Get environment class and config
        if "env" not in config:
            raise ValueError("must specify environment through 'env' parameter")

        env_name = config.get("env")
        env_config = config.get("env_config", {})
        env_eval_config = config.get("env_eval_config", env_config)
        env_cls = get_env_class(env_name)

        # Build environment - get observation and action spaces
        self._env = env_cls(env_config, spec_only=False) 
        obs_spaces = self._env.observation_spaces
        action_spaces = self._env.action_spaces

        self._eval_env = env_cls(env_eval_config, spec_only=False)

        # Initialize learning agents
        if "learner" in config:
            base_cls = get_learner_class(config["learner"])
            base_config = config.get("learner_config", {})
        else:
            base_cls = None
            base_config = None
        
        self._learners = {}
        for id in obs_spaces.keys():
            if "learners" in config and id in config["learners"]:
                learner_cls = get_learner_class(config["learners"][id])
                learner_config = config["learners"].get(f"{id}_config", {})
            elif base_cls is not None:
                learner_cls = base_cls
                learner_config = base_config
            else:
                raise ValueError(f"must specify either a base learner or a learner for '{id}'")
            
            self._learners[id] = learner_cls(obs_spaces[id], 
                action_spaces[id], learner_config, device)

        # Initialize training and eval policies
        self._training_policies = {}
        self._eval_policies = {}
        for id, learner in self._learners.items():
            self._training_policies[id] = learner.make_policy()
            self._eval_policies[id] = learner.make_policy(eval=True)

        # Statistics and timers
        self._global_timer = Stopwatch()
        self._sampling_timer = Stopwatch()
        self._learning_timer = Stopwatch()

        self._timesteps_total = 0
        self._episodes_total = 0

        self._current_iteration = 0

        # Self-play info
        self._round = 0
        self._checkpoint_dist = None
        self._checkpoint_pids = defaultdict(list)

    def _checkpoint(self, round):

        # Update checkpoint distribution
        dist = np.empty((round,))
        p = 1

        for idx in reversed(range(round)):
            dist[idx] = p
            p *= self._weight_decay

        self._checkpoint_dist = dist / np.sum(dist)

        # Save policy checkpoints
        for id, learner in self._learners.items():  
            policy = FrozenPolicy(learner.export_policy(), self._device)
            pid = f"{id}_{round - 1}"

            self._training_policies[pid] = policy
            self._checkpoint_pids[id].append(pid)

    def _sample_checkpoints(self):
        batches = Batch()

        for pid in self._learners.keys():  

            # Randomized policy mapping
            pids = self._checkpoint_pids
            dist = self._checkpoint_dist
            
            policy_fn = lambda id: pid if id == pid else np.random.choice(pids[id], p=dist)

            learner_batch = sample(self._env, self._training_policies,
                self._iteration_episodes, self._max_steps, policy_fn)

            # Add learning policy batch
            batches.extend(learner_batch.policy_batch(pid))

        return batches

    def train(self):
        self._global_timer.start()
        stats = {}

        # Update sampling policies
        for id, learner in self._learners.items():
            self._training_policies[id].update(learner.get_actor_update())

        # Collect training batch and batch statistics
        sampling_time = self._sampling_timer.elapsed()
        self._sampling_timer.start()

        if self._current_iteration <= self._burn_in_iterations:
            training_batch = sample(self._env, self._training_policies,
                self._iteration_episodes, self._max_steps)
        else:
            training_batch = self._sample_checkpoints()
        
        self._sampling_timer.stop()
        sampling_time = self._sampling_timer.elapsed() - sampling_time + 1e-6

        for key, value in training_batch.statistics().items():
            stats["sampling/" + key] = value

        stats["sampling/episodes_per_s"] = training_batch.episodes / sampling_time
        stats["sampling/timesteps_per_s"] = training_batch.timesteps / sampling_time

        # Train learners on new training batch
        for id, episodes in training_batch.items():
            self._learning_timer.start()
            learning_stats = self._learners[id].learn(episodes)
            self._learning_timer.stop()

            for key, value in learning_stats.items():
                stats[f"learning/{id}/{key}"] = value

        # Increment iteration
        self._current_iteration += 1

        # Do checkpointing if necessary
        if self._current_iteration >= self._burn_in_iterations:
            round_iteration = self._current_iteration - self._burn_in_iterations

            if round_iteration % self._round_iterations == 0:
                self._round += 1
                self._checkpoint(self._round)

        # Do evaluation if needed (update eval policies first)
        if self._current_iteration % self._eval_iterations == 0:
            for id, learner in self._learners.items():
                self._eval_policies[id].update(learner.get_actor_update(eval=True))

            eval_batch = sample(self._eval_env, 
                self._eval_policies, self._eval_episodes, self._max_steps)

            for key, value in eval_batch.statistics().items():
                stats["eval/" + key] = value
        
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
    
    def export_policies(self):
        policies = {}
        for id, learner in self._learners.items():
            policies[id] = learner.export_policy()

            for r in range(self._round):
                pid = f"{id}_{r}"
                policies[pid] = self._training_policies[pid].model
        
        return policies
