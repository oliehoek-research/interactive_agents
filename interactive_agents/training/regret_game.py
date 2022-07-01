"""Trainer that applies independent RL to the regret-game of a two-player cooperative game"""
from collections import defaultdict
import numpy as np
import torch

from interactive_agents.envs import get_env_class
from interactive_agents.training.learners import get_learner_class
from interactive_agents.sampling import sample, Batch, FrozenPolicy
from interactive_agents.stopwatch import Stopwatch


class RegretGameTrainer:

    def __init__(self, config, seed=0, device="cpu", verbose=False):
        self._round_iterations = config.get("round_iterations", 100)
        self._burn_in_iterations = config.get("burn_in_iterations", 100)
        self._weight_decay = config.get("weight_decay", 0.0)

        self._alice_episodes = config.get("alice_episodes", 128)
        self._bob_episodes = config.get("bob_episodes", 128)
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

        assert len(obs_spaces.keys()) == 2, "regret games are only defined for two-player games"

        # Initialize learners
        ids = list(obs_spaces.keys())
        self._alice_id = config.get("alice_id", ids[0])
        self._bob_id = config.get("bob_id", ids[1])

        spaces = {
            "alice": [
                obs_spaces[self._alice_id],
                action_spaces[self._alice_id]
            ],
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

        # Initialize training policies
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

        # Save policy checkpoints - no need to checkpoint Eve
        for id in ["alice", "bob"]:
            policy = FrozenPolicy(self._learners[id].export_policy(), self._device)
            pid = f"{id}_{round - 1}"

            self._training_policies[pid] = policy
            self._checkpoint_pids[id].append(pid)

    def _sample(self, map, episodes):
        policy_fn = lambda id: map[id]

        return sample(self._env, self._training_policies, 
            episodes, self._max_steps, policy_fn)

    def _sample_checkpoints(self, id, map, episodes):
        pid = map[id]
        pids = self._checkpoint_pids
        dist = self._checkpoint_dist

        policy_fn = lambda a: pid if a == id else np.random.choice(pids[map[a]], p=dist)            

        return sample(self._env, self._training_policies, 
            episodes, self._max_steps, policy_fn)

    def _burn_in_sampling(self):

        # Sample Alice playing with Bob
        alice_batch = self._sample({
            self._alice_id: "alice",
            self._bob_id: "bob"
        }, self._alice_episodes)

        # Sample Eve playing with Bob
        bob_batch = self._sample({
            self._alice_id: "eve",
            self._bob_id: "bob"
        }, self._bob_episodes)

        # Modify Bobs rewards when playing with Alice
        for episode in alice_batch["bob"]:
            episode[Batch.REWARD] = -episode[Batch.REWARD]

        # Combine and return Batches
        batches = Batch()
        batches.extend(alice_batch)
        batches.extend(bob_batch)

        return batches

    def _sampling(self):
        
        # Sample Alice playing with Bob's past checkpoints
        alice_batch = self._sample_checkpoints(self._alice_id, {
            self._alice_id: "alice",
            self._bob_id: "bob"
        }, self._alice_episodes)

        # Sample Bob playing with Alice's past checkpoints
        bob_batch = self._sample_checkpoints(self._bob_id, {
            self._alice_id: "alice",
            self._bob_id: "bob"
        }, self._alice_episodes)

        # NOTE: It may eventually make sense to implement a CDTE approach for Eve and Bob
        # Sample Bob and Eve playing with each other's current policies
        eve_bob_batch = self._sample({
            self._alice_id: "eve",
            self._bob_id: "bob"
        }, self._bob_episodes)

        # Modify Bobs rewards when playing with Alice
        for episode in bob_batch["bob"]:
            episode[Batch.REWARD] = -episode[Batch.REWARD]

        # Combine and return batches
        batches = Batch()
        batches.extend(alice_batch.policy_batch("alice"))
        batches.extend(bob_batch.policy_batch("bob"))
        batches.extend(eve_bob_batch)

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
            training_batch = self._burn_in_sampling()
        else:
            training_batch = self._sampling()
        
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
        
        # Do evaluation if needed (only evaluate Alice and Bob)
        if self._current_iteration % self._eval_iterations == 0:
            for id, learner in self._learners.items():
                self._eval_policies[id].update(learner.get_actor_update(eval=True))

            policy_fn = lambda id: "alice" if id == self._alice_id else "bob"

            eval_batch = sample(self._eval_env, self._eval_policies, 
                self._eval_episodes, self._max_steps, policy_fn)

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

        for id in ["alice", "bob"]:
            for r in range(self._round):
                pid = f"{id}_{r}"
                policies[pid] = self._training_policies[pid].model
        
        return policies
