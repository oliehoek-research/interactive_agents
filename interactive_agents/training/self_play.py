"""Simple Trainer that runs independent learners for each agent"""
from collections import defaultdict
import numpy as np

from interactive_agents.envs import get_env_class
from interactive_agents.training.learners import get_learner_class
from interactive_agents.sampling import sample, FrozenPolicy
from interactive_agents.stopwatch import Stopwatch


class SelfPlayTrainer:

    def __init__(self, config):
        self._round_iterations = config.get("round_iterations", 10)
        self._burn_in_iterations = config.get("burn_in_iterations", 10)
        self._weight_decay = config.get("weight_decay", 0.0)
        
        self._iteration_episodes = config.get("iteration_episodes", 100)
        self._eval_episodes = config.get("eval_episodes", 10)
        self._max_steps = config.get("max_steps", 100)

        # Get environment class and config
        if "env" not in config:
            raise ValueError("must specify environment through 'env' parameter")

        env_name = config.get("env")
        env_config = config.get("env_config", {})
        env_eval_config = config.get("env_eval_config", env_config)
        env_cls = get_env_class(env_name)

        # Build environment - get observation and action spaces
        self._env = env_cls(env_config, spec_only=False)
        obs_space = self._env.observation_space
        action_space = self._env.action_space

        self._eval_env = env_cls(env_eval_config, spec_only=False)

        # Get learner class and config
        if "learner" not in config:
            raise ValueError("must specify learning algorithm through 'learner' parameter")

        learner_config = config.get("learner_config", {})
        learner_cls = get_learner_class(config.get("learner"))

        # Initialize learners
        self._learners = {}
        for id in obs_space.keys():
            self._learners[id] = learner_cls(obs_space[id], action_space[id], learner_config)

        # Initialize training and eval policies
        self._training_policies = {}
        self._eval_policies = {}
        for id, learner in self._learners.items():
            self._training_policies[id] = learner.make_policy()
            self._eval_policies[id] = learner.make_policy(eval=True)

        # Global timer
        self._timer = Stopwatch()

        # Accumulated statistics
        self._total_iterations = 0
        self._total_episodes = 0
        self._total_samples = 0

        self._total_sampling_time = 0
        self._total_learning_time = 0

        # Self-play round
        self._round = 0
    
    def _checkpoint(self, round):
        for id, learner in self._learners.items():
            policy = FrozenPolicy(learner.export_policy())
            self._training_policies[f"{id}_{round - 1}"] = policy

    def _sample_checkpoints(self, round):

        # Compute weighted checkpoint distribution
        dist = np.empty((round,))
        p = 1

        for idx in range(round):
            dist[idx] = p
            p *= self._weight_decay

        dist = dist / np.sum(dist)
        
        # Build array of checkpoint ids
        pids = defaultdict(list)
        for pid in self._learners.keys():
            for r in range(round):
                pids[pid].append(f"{pid}_{r}")

        # Generate samples for each learning player
        batches = {}
        stats = defaultdict(lambda: 0)

        for pid in self._learners.keys():

            # Randomized policy mapping
            policy_fn = lambda id: pid if id == pid else np.random.choice(pids[id], p=dist)

            batch, batch_stats = sample(self._env, self._training_policies,
                self._iteration_episodes, self._max_steps, policy_fn)

            # Get learning policy batch
            batches[pid] = batch.policy_batch(pid)

            # Accumulate statistic
            for key, value in batch_stats.items():
                stats[key] += value

        return batches, stats

    def train(self):
        self._timer.start()
        watch = Stopwatch()

        # Update sampling policies
        for id, learner in self._learners.items():
            self._training_policies[id].update(learner.get_update())

        # Collect training batch and batch statistics
        watch.restart()

        if self._total_iterations < self._burn_in_iterations:
            batch, batch_stats = sample(self._env, self._training_policies,
                self._iteration_episodes, self._max_steps)
        else:
            batch, batch_stats = self._sample_checkpoints(self._round)
        
        watch.stop()

        # Accumulate sampling statistics
        stats = {}

        # NOTE: This doesn't work because stats collected during burn-in and self-play are different
        # for key, value in batch_stats.items():
        #     stats["sampling/" + key] = value

        stats["sampling/episodes_per_s"] = batch_stats["episodes"] / watch.elapsed()
        stats["sampling/samples_per_s"] = batch_stats["samples"] / watch.elapsed()

        self._total_sampling_time += watch.elapsed()
        stats["sampling/total_time_s"] = self._total_sampling_time

        self._total_iterations += 1
        self._total_episodes += batch_stats["episodes"]
        self._total_samples += batch_stats["samples"]
        
        stats["total_iterations"] = self._total_iterations
        stats["total_episodes"] = self._total_episodes
        stats["total_samples"] = self._total_samples

        # Train learners on new training batch
        watch.reset()
        for id, episodes in batch.items():  # TODO: Have MultiBatch implement an immutable dictionary interface
            watch.start()
            batch_stats = self._learners[id].learn(episodes)
            watch.stop()

            for key, value in batch_stats.items():
                stats[f"learning/{id}/{key}"] = value
        
        self._total_learning_time += watch.elapsed()
        stats["learning/total_time_s"] = self._total_learning_time

        # Do checkpointing if necessary
        if self._total_iterations >= self._burn_in_iterations:
            round_iteration = self._total_iterations - self._burn_in_iterations

            if round_iteration % self._round_iterations == 0:
                self._round += 1
                self._checkpoint(self._round)

        # Update eval policies
        for id, learner in self._learners.items():
            self._eval_policies[id].update(learner.get_update(eval=True))

        # Run evaluation episodes
        _, eval_stats = sample(self._eval_env, self._eval_policies,
             self._eval_episodes, self._max_steps)

        for key, value in eval_stats.items():
            stats["eval/" + key] = value
        
        # Add total training time
        self._timer.stop()
        stats["total_time_s"] = self._timer.elapsed()

        return stats
    
    def export_policies(self):
        policies = {}
        for id, learner in self._learners.items():
            policies[id] = learner.export_policy()

            for r in range(self._round):
                pid = f"{id}_{r}"
                policies[pid] = self._training_policies[pid].model
        
        return policies
