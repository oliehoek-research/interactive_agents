"""Simple Trainer that runs independent learners for each agent"""
import numpy as np
import torch

from interactive_agents.envs import get_env_class
from interactive_agents.training.learners import get_learner_class
from interactive_agents.sampling import sample
from interactive_agents.util.stopwatch import Stopwatch

class IndependentTrainer:

    def __init__(self, config, seed=0, device="cpu", verbose=False):
        self._iteration_episodes = config.get("iteration_episodes", 128)
        self._max_steps = config.get("max_steps", 100)
        
        self._eval_iterations = config.get("eval_iterations", 10)
        self._eval_episodes = config.get("eval_episodes", 64)
        
        self._seed = seed
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

        # Get learner class and config
        if "learner" not in config:
            raise ValueError("must specify learning algorithm through 'learner' parameter")

        learner_config = config.get("learner_config", {})
        learner_cls = get_learner_class(config.get("learner"))

        # Initialize learners
        self._learners = {}
        for id in obs_spaces.keys():
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

    def train(self):
        self._global_timer.start()
        stats = {} 

        # Update sampling policies
        for id, learner in self._learners.items():
            self._training_policies[id].update(learner.get_actor_update())
        
        # Collect training batch and batch statistics
        sampling_time = self._sampling_timer.elapsed()
        self._sampling_timer.start()
        training_batch = sample(self._env, self._training_policies,
             self._iteration_episodes, self._max_steps)
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
        
        return policies
