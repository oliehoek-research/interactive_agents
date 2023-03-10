"""
Trainer class for the simplified action 
decoder (SAD) as described in the paper "Simplified 
action decoder for deep multi-agent reinforcement 
learning." (Hu et al. 2019).

Almost identical to the "independent" trainer, but allows
each player to see the other's exploration and exploitation action
"""
from gymnasium.spaces import Box
import numpy as np
import torch

from interactive_agents.envs import get_env_class, BatchedEnv
from interactive_agents.training.learners.r2d2 import R2D2
from interactive_agents.sampling import BatchBuilder
from interactive_agents.stopwatch import Stopwatch


class SADTrainer:

    def __init__(self, config, seed=0, device="cpu", verbose=False):
        self._iteration_episodes = config.get("iteration_episodes", 128)
        self._num_envs = config.get("num_envs", 32)
        self._max_steps = config.get("max_steps", np.inf)  # NOTE: Maximum episode length (what environments do we need this for?)
        
        self._eval_iterations = config.get("eval_iterations", 10)
        self._eval_episodes = config.get("eval_episodes", 64)
        
        self._verbose = verbose
        self._seed = seed

        # Seed random number generators
        np.random.seed(seed)  # NOTE: Cannot be sure all environments use the seed we give them
        torch.manual_seed(seed)

        seq = np.random.SeedSequence(seed)
        self._rng = np.random.default_rng(seq)

        # Get environment class and config
        if "env" not in config:  # NOTE: Not defult environment
            raise ValueError("must specify environment through 'env' parameter")

        env_name = config.get("env")
        env_config = config.get("env_config", {})
        env_eval_config = config.get("env_eval_config", env_config)
        env_cls = get_env_class(env_name)

        self._env = env_cls(env_config)
        self._eval_env = env_cls(env_eval_config)
        self._agent_ids = list(self._env.possible_agents)

        first_agent = self._env.possible_agents[0]
        self._obs_space = self._env.observation_space(first_agent)
        self._action_space = self._env.action_space(first_agent)

        # TODO: Check that all observation and action spaces are identical, or handle the case where they are not

        # Setup SAD extended observation space
        # obs_size = np.prod(self._obs_space.shape) + 2 * (len(self._agent_ids) - 1) * self._action_space.n
        obs_size = np.prod(self._obs_space.shape) + (len(self._agent_ids) - 1) * self._action_space.n
        sad_obs_space = Box(0, 1, shape=(obs_size,))  # NOTE: We never enforce the low-high bounds, so their values don't matter

        # Initialize Learner
        learner_config = config.get("learner_config", {})
        self._learner = R2D2(sad_obs_space, self._action_space, learner_config, device)

        # Initialize training and eval policies
        self._policy = self._learner.make_policy()
        self._eval_policy = self._learner.make_policy(eval=True)

        # Statistics and timers
        self._global_timer = Stopwatch()
        self._sampling_timer = Stopwatch()
        self._learning_timer = Stopwatch()

        self._timesteps_total = 0  # NOTE: Keep track of the total number of timesteps and episodes generated
        self._episodes_total = 0

        self._current_iteration = 0  # NOTE: Keeps track of the total number of iterations

    def _get_obs(self, obervations, actions=None, fetches=None):
        """
        Converts multi-agent observation and action disctionaries into a
        dictionary of observations with the actions of other agents appended
        """
        new_observations = {}

        for agent_id, obs in obervations.items():
            new_obs = [obs.reshape(-1)]

            for other_id in self._agent_ids:
                if other_id != agent_id:
                    action = np.zeros(self._action_space.n)

                    if fetches is not None and other_id in fetches:
                        action[fetches[other_id]["q_values"].argmax()] = 1
                    elif actions is not None and other_id in actions:
                        action[actions[other_id]] = 1
                    
                    new_obs.append(action)

            new_observations[agent_id] = np.concatenate(new_obs)

        return new_observations

    # NOTE: In newer versions of Gym (and PettingZoo) there has been a move 
    # towards treating experiences as one long sequence of data, rather than
    # as discrete episodes.  Learner's can use the "truncated" or "terminated"
    # signals to determine when episodes begin or end.

    def _sample(self, policy, num_episodes, eval=False):
        """Generates a batch of episodes using the given policies"""
        batch = BatchBuilder()

        if eval:
            env = self._eval_env
        else:
            env = self._env
        
        for _ in range(num_episodes):
            obs = self._get_obs(env.reset(seed=self._rng.bit_generator.random_raw()))
            current_step = 0
            done = False

            # NOTE: Eventually we want to support batched inference for vectored policies
            agents = {}
            policy_map = {}
            terminated = {}
            truncated = {}
            for agent_id in self._agent_ids:
                agents[agent_id] = policy.make_agent()
                policy_map[agent_id] = "policy_0"
                terminated[agent_id] = False
                truncated[agent_id] = False

            batch.start_episode(obs, policy_map)

            while current_step < self._max_steps and not done:
                actions = {}
                fetches = {}
                for agent_id, ob in obs.items():
                    actions[agent_id], fetches[agent_id] = agents[agent_id].act(ob)

                obs, rewards, terminated, truncated, _ = env.step(actions)

                if eval:
                    obs = self._get_obs(obs, actions)
                else:
                    obs = self._get_obs(obs, actions, fetches)

                done = all(terminated.values()) or all(truncated.values())

                batch.step(obs, actions, rewards, terminated, truncated, fetches)  # NOTE: Need to modify to handle new PettingZoo interface
                current_step += 1
        
            batch.end_episode()
        
        return batch.build()

    def train(self):
        self._global_timer.start()
        stats = {} 

        # Update sampling policy
        self._policy.update(self._learner.get_actor_update())
        
        # Collect training batch and batch statistics
        sampling_time = self._sampling_timer.elapsed()
        self._sampling_timer.start()
        batch = self._sample(self._policy, self._iteration_episodes)
        self._sampling_timer.stop()
        sampling_time = self._sampling_timer.elapsed() - sampling_time + 1e-6

        for key, value in batch.statistics().items():
            stats["sampling/" + key] = value

        # NOTE: Calculate sampling throughputs
        stats["sampling/episodes_per_s"] = batch.episodes / sampling_time
        stats["sampling/timesteps_per_s"] = batch.timesteps / sampling_time

        # Train learners on new training batch
        self._learning_timer.start()
        learning_stats = self._learner.learn(batch["policy_0"])
        self._learning_timer.stop()

        for key, value in learning_stats.items():
            stats[f"learning/{key}"] = value

        # Increment iteration
        self._current_iteration += 1

        # Do evaluation if needed (update eval policies first)
        if self._current_iteration % self._eval_iterations == 0:
            self._eval_policy.update(self._learner.get_actor_update(eval=True))
            eval_batch = self._sample(self._eval_policy, self._eval_episodes, eval=True)

            for key, value in eval_batch.statistics().items():
                stats["eval/" + key] = value
            
            if self._verbose:
                print(f"\n\nEVALUATION, seed {self._seed}, iteration {self._current_iteration}")
                print(f"mean eval reward: {stats['eval/reward_mean']}")

        # Accumulate global statistics
        self._global_timer.stop()

        self._episodes_total += batch.episodes
        self._timesteps_total += batch.timesteps

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
