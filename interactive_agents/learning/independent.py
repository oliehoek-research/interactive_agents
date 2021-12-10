from collections import defaultdict

import numpy as np

from interactive_agents.envs import get_env_class
from interactive_agents.learning import get_learner_class
from interactive_agents.sampling import Sampler
from interactive_agents.stopwatch import Stopwatch


class IndependentTrainer:

    def __init__(self, config):
        self._iteration_episodes = config.get("iteration_episodes", 100)
        self._eval_episodes = config.get("eval_episodes", 10)

        # Get environment class and config
        if "env" not in config:
            raise ValueError("must specify environment through 'env' parameter")

        env_name = config.get("env")
        env_config = config.get("env_config", {})
        env_cls = get_env_class(env_name)

        # Build dummy environment - get observation and action spaces
        env = env_cls(env_config, spec_only=True)
        obs_space = env.observation_space
        action_space = env.action_space

        # Get learner class and config
        if "learner" not in config:
            raise ValueError("must specify learning algorithm through 'learner' parameter")

        learner_config = config.get("learner_config", {})
        learner_cls = get_learner_class(config.get("learner"))

        # Initialize learners
        self._learners = {}
        for id in obs_space.keys():
            self._learners[id] = learner_cls(obs_space[id], action_space[id], learner_config)

        # Initialize training and eval samplers
        training_policies = {}
        eval_policies = {}
        for id, learner in self._learners.items():
            training_policies[id] = learner.make_policy()
            eval_policies[id] = learner.make_policy(eval=True)
        
        max_steps = config.get("max_steps", 100)
        policy_fn = lambda id: id

        self._training_sampler = Sampler(env_name, env_config, training_policies, policy_fn, max_steps)
        self._eval_sampler = Sampler(env_name, env_config, eval_policies, policy_fn, max_steps)

        # Global timer
        self._timer = Stopwatch()

        # Accumulated statistics
        self._total_iterations = 0
        self._total_episodes = 0
        self._total_samples = 0

        self._total_sampling_time = 0
        self._total_learning_time = 0

    def train(self):
        self._timer.start()
        watch = Stopwatch()

        # Update sampling policies
        updates = {}
        for id, learner in self._learners.items():
            updates[id] = learner.get_policy_update()
        self._training_sampler.update_policies(updates)

        # Collect training batch and batch statistics
        watch.restart()
        batch, batch_stats = self._training_sampler.sample(self._iteration_episodes)
        watch.stop()
        
        for id, learner in self._learners.items():
            learner.add_batch(batch[id])

        stats = {}
        for key, value in batch_stats.items():
            stats["sampling/" + key] = value

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
        for id, learner in self._learners.items():
            learner.add_batch(batch[id])
            
            watch.start()
            batch_stats = learner.learn()
            watch.stop()

            for key, value in batch_stats.items():
                stats[f"learning/{id}/{key}"] = value
        
        self._total_learning_time += watch.elapsed()
        stats["learning/total_time_s"] = self._total_learning_time

        # Update eval policies
        updates = {}
        for id, learner in self._learners.items():
            updates[id] = learner.get_policy_update(eval=True)
        self._eval_sampler.update_policies(updates)

        # Run evaluation episodes
        _, eval_stats = self._eval_sampler.sample(self._eval_episodes)

        for key, value in batch_stats.items():
            stats["eval/" + key] = value
        
        # Add total training time
        self._timer.stop()
        stats["total_time_s"] = self._timer.elapsed()

        return stats
    
    def get_policies(self):
        policies = {}
        for id, learner in self._learners.items():
            policies[id] = learner.make_policy(eval=True)
            policies[id].update(learner.get_policy_update(eval=True))
        
        return policies
    
    def get_state(self):
        state = {}
        for id, learner in self._learners.items():
            state[id] = learner.get_state()
        
        return state
    
    def set_state(self, state):
        for id, learner in self._learners.items():
            if id in state:
                learner.set_state(state[id])
