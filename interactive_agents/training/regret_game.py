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
        self._round_iterations = config.get("round_iterations", 10)
        self._burn_in_iterations = config.get("burn_in_iterations", 10)
        self._weight_decay = config.get("weight_decay", 0.0)
        self._alice_episodes = config.get("alice_episodes", 100)
        self._bob_episodes = config.get("bob_episodes", 100)
        self._max_steps = config.get("max_steps", 100)
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
        env_cls = get_env_class(env_name)

        # Build environment - get observation and action spaces
        self._env = env_cls(env_config, spec_only=False)
        obs_spaces = self._env.observation_spaces
        action_spaces = self._env.action_spaces

        assert len(obs_spaces.keys()) == 2, "regret games are only defined for two-player games"

        ids = list(obs_spaces.keys())
        self._alice_id = config.get("alice_id", ids[0])
        self._bob_id = config.get("bob_id", ids[1])

        # Initialize learners
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

        self._learners = {}
        for pid, (obs_space, action_space) in spaces.items():
            if pid in config:
                cls_name = config.get(pid, "R2D2")
                conf = config.get(f"{pid}_config", {})
            else:
                cls_name = config.get("alice", "R2D2")
                conf = config.get("alice_config", {})

            cls = get_learner_class(cls_name)
            self._learners[pid] = cls(obs_space, action_space, conf, device)

        # Initialize training policies
        self._training_policies = {}
        for id, learner in self._learners.items():
            self._training_policies[id] = learner.make_policy()

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

    def _sample(self, map, episodes):
        policy_fn = lambda id: map[id]

        return sample(self._env, self._training_policies, 
            episodes, self._max_steps, policy_fn)

    # NOTE: Check that this usage is correct
    def _sample_checkpoints(self, id, pid, checkpoint_pid, episodes):
        pids = self._checkpoint_pids[checkpoint_pid]
        dist = self._checkpoint_dist

        policy_fn = lambda a: pid if a == id else np.random.choice(pids, p=dist)            

        return sample(self._env, self._training_policies, 
            episodes, self._max_steps, policy_fn)

    def _learn(self, 
            alice_batch, 
            bob_alice_batch, 
            bob_eve_batch, 
            eve_batch):
        stats = {}

        # Train alice policy
        stats["alice"] = self._learners["alice"].learn(alice_batch)

        # Train response strategy
        stats["eve"] = self._learners["eve"].learn(eve_batch)

        # Modify partner rewards and combine batches
        for episode in bob_alice_batch:
            episode[Batch.REWARD] = -episode[Batch.REWARD]

        bob_batch = bob_alice_batch + bob_eve_batch

        # Train partner strategy
        stats["bob"] = self._learners["bob"].learn(bob_batch)

        return stats

    def _burn_in_update(self):
        stats = {}

        # Collect training batches
        sampling_time = self._sampling_timer.elapsed()
        self._sampling_timer.start()

        alice_batch, alice_stats = self._sample({
            self._alice_id: "alice",
            self._bob_id: "bob"
        }, self._alice_episodes)

        bob_batch, bob_stats = self._sample({
            self._alice_id: "eve",
            self._bob_id: "bob"
        }, self._bob_episodes)

        self._sampling_timer.stop()
        sampling_time = self._sampling_timer.elapsed() - sampling_time + 1e-6

        # Collect sampling statistics
        for key, value in alice_stats.items():
            stats["sampling/alice/" + key] = value
        
        for key, value in bob_stats.items():
            stats["sampling/bob/" + key] = value

        episodes = alice_stats["episodes"] + bob_stats["episodes"]
        stats["sampling/episodes_per_s"] = episodes / sampling_time

        timesteps = alice_stats["timesteps"] + alice_stats["timesteps"]
        stats["sampling/timesteps_per_s"] = timesteps / sampling_time

        self._episodes_total += episodes
        self._timesteps_total += timesteps

        stats["global/episodes_total"] = self._episodes_total
        stats["global/timesteps_total"] = self._timesteps_total
        stats["global/sampling_time_s"] = self._sampling_timer.elapsed()

        # Train on batches
        self._learning_timer.start()
        learning_stats = self._learn(
            alice_batch.policy_batch("alice"),
            alice_batch.policy_batch("bob"),
            bob_batch.policy_batch("bob"),
            bob_batch.policy_batch("eve"))
        self._learning_timer.stop()

        # Collect learning statistics
        for pid, policy_stats in learning_stats.items():
            for key, value in policy_stats.items():
                stats[f"learning/{pid}/{key}"] = value

        stats["global/learning_time_s"] = self._learning_timer.elapsed()

        return stats

    def _update(self):
        stats = {}

        # Collect training batches
        sampling_time = self._sampling_timer.elapsed()
        self._sampling_timer.start()

        alice_batch, alice_stats = self._sample_checkpoints(
            self._alice_id, "alice", "bob", self._alice_episodes)

        bob_alice_batch, bob_alice_stats = self._sample_checkpoints(
            self._bob_id, "bob", "alice", self._alice_episodes)
        
        bob_eve_batch, bob_eve_stats = self._sample_checkpoints(
            self._bob_id, "bob", "eve", self._bob_episodes)

        eve_batch, eve_stats = self._sample_checkpoints(
            self._alice_id, "eve", "bob", self._bob_episodes)

        self._sampling_timer.stop()
        sampling_time = self._sampling_timer.elapsed() - sampling_time + 1e-6

        # Collect sampling statistics
        for key, value in alice_stats.items():
            stats["sampling/alice/" + key] = value

        for key, value in bob_alice_stats.items():
            stats["sampling/bob_alice/" + key] = value

        for key, value in bob_eve_stats.items():
            stats["sampling/bob_eve/" + key] = value

        for key, value in eve_stats.items():
            stats["sampling/eve/" + key] = value

        episodes = alice_stats["episodes"] + bob_alice_stats["episodes"] \
            + bob_eve_stats["episodes"] + eve_stats["episodes"]
        stats["sampling/episodes_per_s"] = episodes / sampling_time

        timesteps = alice_stats["timesteps"] + bob_alice_stats["timesteps"] \
            + bob_eve_stats["timesteps"] + eve_stats["timesteps"]
        stats["sampling/timesteps_per_s"] = timesteps / sampling_time

        self._episodes_total += episodes
        self._timesteps_total += timesteps

        stats["global/episodes_total"] = self._episodes_total
        stats["global/timesteps_total"] = self._timesteps_total
        stats["global/sampling_time_s"] = self._sampling_timer.elapsed()

        # Train on batches
        self._learning_timer.start()
        learning_stats = self._learn(
            alice_batch.policy_batch("alice"),
            bob_alice_batch.policy_batch("bob"),
            bob_eve_batch.policy_batch("bob"),
            eve_batch.policy_batch("eve"))
        self._learning_timer.stop()

        # Collect learning statistics
        for pid, policy_stats in learning_stats.items():
            for key, value in policy_stats.items():
                stats[f"learning/{pid}/{key}"] = value

        stats["global/learning_time_s"] = self._learning_timer.elapsed()

        return stats

    def train(self):
        self._global_timer.start()

        # Update sampling policies
        for id, learner in self._learners.items():
            self._training_policies[id].update(learner.get_actor_update())

        # Do training updates
        if self._current_iteration < self._burn_in_iterations:
            stats = self._burn_in_update()
        else:
            stats = self._update()

        # Print iteration stats
        if self._verbose:
            print(f"\n\nSEED {self._seed}, ITERATION {self._current_iteration}")
            print(f"total episodes: {self._episodes_total}")
            print(f"total timesteps: {self._timesteps_total}")

        # Increment iteration
        self._current_iteration += 1

        # Do checkpointing if necessary
        if self._current_iteration >= self._burn_in_iterations:
            round_iteration = self._current_iteration - self._burn_in_iterations

            if round_iteration % self._round_iterations == 0:
                self._round += 1
                self._checkpoint(self._round)
        
        # Add total training time
        self._global_timer.stop()
        stats["global/total_time_s"] = self._global_timer.elapsed()

        return stats
    
    def export_policies(self):
        policies = {}
        for id, learner in self._learners.items():
            policies[id] = learner.export_policy()

            for r in range(self._round):
                pid = f"{id}_{r}"
                policies[pid] = self._training_policies[pid].model
        
        return policies
