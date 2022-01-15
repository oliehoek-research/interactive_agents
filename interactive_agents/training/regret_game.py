"""Trainer that applies independent RL to the regret-game of a two-player cooperative game"""
from interactive_agents.envs import get_env_class
from interactive_agents.training.learners import get_learner_class
from interactive_agents.sampling import sample, MultiBatch
from interactive_agents.stopwatch import Stopwatch


class RegretGameTrainer:

    def __init__(self, config):
        self._learner_episodes = config.get("learner_episodes", 100)
        self._learner_eval_episodes = config.get("earner_eval_episodes", 10)

        self._partner_episodes = config.get("partner_episodes", 100)
        self._partner_eval_episodes = config.get("partner_eval_episodes", 10)

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
        obs_spaces = self._env.observation_space
        action_spaces = self._env.action_space

        self._eval_env = env_cls(env_eval_config, spec_only=False)

        assert len(obs_spaces.keys()) == 2, "regret games are only defined for two-player games"

        ids = list(obs_spaces.keys())
        self._learner_id = config.get("learner_id", ids[0])
        self._partner_id = config.get("partner_id", ids[1])

        # Initialize learners
        spaces = {
            "learner": [
                obs_spaces[self._learner_id],
                action_spaces[self._learner_id]
            ],
            "response": [
                obs_spaces[self._learner_id],
                action_spaces[self._learner_id]
            ],
            "partner": [
                obs_spaces[self._partner_id],
                action_spaces[self._partner_id]
            ]
        }

        self._learners = {}
        for pid, (obs_space, action_space) in spaces.items():
            if pid in config:
                cls_name = config.get(pid)
                conf = config.get(f"{pid}_config", {})
            else:
                cls_name = config.get("learner", "R2D2")
                conf = config.get("learner_config", {})

            cls = get_learner_class(cls_name)
            self._learners[pid] = cls(obs_space, action_space, conf)

        # Initialize training and eval policies
        self._training_policies = {}
        self._eval_policies = {}
        for id, learner in self._learners.items():
            self._training_policies[id] = learner.make_policy()
            self._eval_policies[id] = learner.make_policy(eval=True)

        # Define policy maps
        self._learner_map = {
            self._learner_id: "learner",
            self._partner_id: "partner"
        }

        self._partner_map = {
            self._learner_id: "response",
            self._partner_id: "partner"
        }

        # Global timer
        self._timer = Stopwatch()

        # Accumulated statistics
        self._total_iterations = 0
        self._total_episodes = 0
        self._total_samples = 0

        self._total_sampling_time = 0
        self._total_learning_time = 0

    def _learn(self, learner_batch, partner_batch):
        batch_learner = learner_batch.policy_batch("learner")
        batch_response = partner_batch.policy_batch("response")

        batch_partner_learner = learner_batch.policy_batch("partner")
        batch_partner_response = partner_batch.policy_batch("partner")

        stats = {}

        # Train learner strategy
        stats["learner"] = self._learners["learner"].learn(batch_learner)

        # Train response strategy
        stats["response"] = self._learners["response"].learn(batch_response)

        # Modify partner rewards and combine batches
        for episode in batch_partner_learner:
            episode[MultiBatch.REWARD] = -episode[MultiBatch.REWARD]

        batch_partner = batch_partner_learner + batch_partner_response

        # Train partner strategy
        stats["partner"] = self._learners["partner"].learn(batch_partner)

        return stats

    def train(self):
        self._timer.start()
        watch = Stopwatch()

        # Update sampling policies
        for id, learner in self._learners.items():
            self._training_policies[id].update(learner.get_update())

        # Collect training batches and statistics
        watch.restart()
        learner_batch, learner_stats = sample(self._env, self._training_policies,
             self._learner_episodes, self._max_steps, self._learner_map)
        partner_batch, partner_stats = sample(self._env, self._training_policies,
             self._partner_episodes, self._max_steps, self._partner_map)
        watch.stop()

        stats = {}

        for key, value in learner_stats.items():
            stats["sampling/learner/" + key] = value
        
        for key, value in partner_stats.items():
            stats["sampling/partner/" + key] = value

        episodes = learner_stats["episodes"] + partner_stats["episodes"]
        stats["sampling/episodes_per_s"] = episodes / watch.elapsed()

        samples = learner_stats["samples"] + partner_stats["samples"]
        stats["sampling/samples_per_s"] = samples / watch.elapsed()

        self._total_sampling_time += watch.elapsed()
        stats["sampling/total_time_s"] = self._total_sampling_time

        self._total_iterations += 1
        self._total_episodes += episodes
        self._total_samples += samples
        
        stats["total_iterations"] = self._total_iterations
        stats["total_episodes"] = self._total_episodes
        stats["total_samples"] = self._total_samples

        # Train policies on sampled experience
        watch.restart()
        learning_stats = self._learn(learner_batch, partner_batch)
        watch.stop()
        
        for id, policy_stats in learning_stats.items():
            for key, value in policy_stats:
                stats[f"learning/{id}/{key}"] = value

        self._total_learning_time += watch.elapsed()
        stats["learning/total_time_s"] = self._total_learning_time

        # Update eval policies
        for id, learner in self._learners.items():
            self._eval_policies[id].update(learner.get_update(eval=True))

        # Run evaluation episodes
        _, learner_eval_stats = sample(self._eval_env, self._eval_policies,
             self._learner_eval_episodes, self._max_steps, self._learner_map)

        _, partner_eval_stats = sample(self._eval_env, self._eval_policies,
             self._partner_eval_episodes, self._max_steps, self._partner_map)

        for key, value in learner_eval_stats.items():
            stats["eval/learner/" + key] = value
        
        for key, value in partner_eval_stats.items():
            stats["eval/partner/" + key] = value
        
        # Add total training time
        self._timer.stop()
        stats["total_time_s"] = self._timer.elapsed()

        return stats
    
    def export_policies(self):
        policies = {}
        for id, learner in self._learners.items():
            policies[id] = learner.export_policy()

        return policies
