"""Trainer uses cooperative RL to maximize the regret of a given, pre-trained policy"""
from collections import defaultdict
import numpy as np
import os
import torch

from interactive_agents.envs import get_env_class
from interactive_agents.training.learners import get_learner_class
from interactive_agents.sampling import sample, Batch, FrozenPolicy
from interactive_agents.util.stopwatch import Stopwatch


class MaxRegretTrainer:

    def __init__(self, config, seed=0, device="cpu", verbose=False):
        self._alice_episodes = config.get("alice_episodes", 128)
        self._bob_episodes = config.get("bob_episodes", 128)
        self._max_steps = config.get("max_steps", 100)

        self._eval_iterations = config.get("eval_iterations", 10)

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

        assert len(obs_spaces.keys()) == 2, "regret games are only defined for two-player games"

        # Initialize learners
        ids = list(obs_spaces.keys())
        self._alice_id = config.get("alice_id", ids[0])
        self._bob_id = config.get("bob_id", ids[1])

        spaces = {
            "bob": [
                obs_spaces[self._bob_id],
                action_spaces[self._bob_id]
            ],
            "eve": [
                obs_spaces[self._alice_id],
                action_spaces[self._alice_id]
            ]
        }

        if "learner" in config:
            base_cls = get_learner_class(config["learner"])
            base_config = config.get("learner_config", {})
        else:
            base_cls = None
            base_config = None

        self._learners = {}
        for id, (obs_space, action_space) in spaces.items():
            if "learners" in config and id in config["learners"]:
                learner_cls = get_learner_class(config["learners"][id])
                learner_config = config["learners"].get(f"{id}_config", {})
            elif base_cls is not None:
                learner_cls = base_cls
                learner_config = base_config
            else:
                raise ValueError(f"must specify either a base learner or a learner for '{id}'")
            
            self._learners[id] = learner_cls(obs_space, 
                action_space, learner_config, device)

        # Load Alices frozen policy
        policy_path = config.get("resources", {}).get("agent", None)
        if policy_path is not None and os.path.isfile(policy_path):
            alice_model = torch.jit.load(policy_path)
            alice_policy = FrozenPolicy(alice_model)
        else:
            raise FileNotFoundError(f"no agent policy found")

        # Initialize training policies
        self._training_policies = {"alice": alice_policy}
        self._eval_policies = {"alice": alice_policy}

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

    def _sample(self, map, policies, episodes):
        policy_fn = lambda id: map[id]

        return sample(self._env, policies, 
            episodes, self._max_steps, policy_fn)

    def _sampling(self, policies):

        # Sample Alice playing with Bob
        alice_batch = self._sample({
            self._alice_id: "alice",
            self._bob_id: "bob"
        }, policies, self._alice_episodes)

        # Sample Eve playing with Bob
        bob_batch = self._sample({
            self._alice_id: "eve",
            self._bob_id: "bob"
        }, policies, self._bob_episodes)

        # Modify Bobs rewards when playing with Alice
        for episode in alice_batch["bob"]:
            episode[Batch.REWARD] = -episode[Batch.REWARD]

        # Combine and return Batches
        batches = Batch()
        batches.extend(alice_batch)
        batches.extend(bob_batch)

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

        training_batch = self._sampling(self._training_policies)
        
        self._sampling_timer.stop()
        sampling_time = self._sampling_timer.elapsed() - sampling_time + 1e-6

        for key, value in training_batch.statistics().items():
            stats["sampling/" + key] = value

        stats["sampling/episodes_per_s"] = training_batch.episodes / sampling_time
        stats["sampling/timesteps_per_s"] = training_batch.timesteps / sampling_time

        # Train learners on new training batch
        for id, learner in self._learners.items():
            self._learning_timer.start()
            learning_stats = learner.learn(training_batch[id])
            self._learning_timer.stop()

            for key, value in learning_stats.items():
                stats[f"learning/{id}/{key}"] = value

        # Increment iteration
        self._current_iteration += 1
        
        # Do evaluation if needed (only evaluate Alice and Bob)  # NOTE This doesn't work correctly - doesn't compute regret
        if self._current_iteration % self._eval_iterations == 0:
            for id, learner in self._learners.items():
                self._eval_policies[id].update(learner.get_actor_update(eval=True))

            eval_batch = self._sampling(self._eval_policies)

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
        
        return policies
