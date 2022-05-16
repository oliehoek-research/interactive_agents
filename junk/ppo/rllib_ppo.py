import logging

from ray.rllib.agents import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.optimizers import SyncSamplesOptimizer, LocalMultiGPUOptimizer
from ray.rllib.utils import try_import_tf

tf = try_import_tf()

from algorithms.agents.ppo.ppo_policy import PPOTFPolicy

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = with_common_config({
    # The GAE(lambda) parameter.
    "lambda": 0.95,  # NOTE: Used to balance between the leanred value function and the monte-carlo value estimate
    # The intrinsic GAE(lambda) parameter.
    "intrinsic_lambda": 0.95,  # NOTE: Only used for intrinsic reward (curiosity) allows us to apply these separately
    # The intrinsic reward discount factor parameter.
    "intrinsic_gamma": 0.95,  # NOTE: Also allows a separate discount factor for intrinsice rewards
    # Initial coefficient for KL divergence.
    "kl_coeff": 0.2,  # NOTE: PPO optionally uses adaptive KL divergence targets, this is the starting point for those targets
    # Size of batches collected from each worker.
    "rollout_fragment_length": 200,  # NOTE: The length of individual trajectories to sample from the environment (we won't need this)
    # Number of timesteps collected for each SGD round. This defines the size
    # of each SGD epoch.
    "train_batch_size": 4000,  # NOTE: Total number of timesteps (not episodes) sampled between each update
    # Total SGD batch size across all devices for SGD. This defines the
    # minibatch size within each epoch.
    "sgd_minibatch_size": 128,  # NOTE: Batch size, is this in timesteps or trajectories?
    # Whether to shuffle sequences in the batch when training (recommended).
    "shuffle_sequences": True,  # NOTE What does this actually do?
    # Number of SGD iterations in each outer loop (i.e., number of epochs to
    # execute per train batch).
    "num_sgd_iter": 30,  # NOTE: The number of gradient updates done per training update
    # Stepsize of SGD.
    "lr": 5e-5,
    # Learning rate schedule.
    "lr_schedule": None,  # NOTE: We should support parameter schedules generically
    # Share layers for value function. If you set this to True, it's important
    # to tune vf_loss_coeff.
    "vf_share_layers": False,  # NOTE: Particularly for CNNs, important to support this, tricky to do in a generic way though
    # Coefficient of the value function loss. IMPORTANT: you must tune this if
    # you set vf_share_layers: True.
    "vf_loss_coeff": 1.0,  # NOTE: Because we may use a shared feature layer, we need to balance between bellman loss and policy loss
    # Coefficient of the entropy regularizer.
    "entropy_coeff": 0.0,  # NOTE PPO also supports entropy regularization (basically a bonus for policies with high-entropy action distributions)
    # Decay schedule for the entropy regularizer.
    "entropy_coeff_schedule": None,  # NOTE: Again, a generic scheduling mechanism would be useful
    # PPO clip parameter.
    "clip_param": 0.3,  # NOTE: This is a big one, basically determines how large a step size to allow in the policy space at each iteration
    # Clip param for the value function. Note that this is sensitive to the
    # scale of the rewards. If your expected V is large, increase this.
    "vf_clip_param": 10.0,  # NOTE: Optional to implement, clipping used to stabilize learning 
    # If specified, clip the global norm of gradients by this amount.
    "grad_clip": None,  # NOTE: Also optional, another stabilization method, seems like the optimizer should do this
    # Target value for KL divergence.
    "kl_target": 0.01,  # NOTE: Effectively an alterniative to the clipped loss, but often used together
    # Whether to rollout "complete_episodes" or "truncate_episodes".
    "batch_mode": "truncate_episodes",  # NOTE: Because we want to learn long-term adaptive policies, we should use complete episodes
    # Which observation filter to apply to the observation.
    "observation_filter": "NoFilter",  # NOTE: This is something specific to RLLib, we won't need to implement this
    # Uses the sync samples optimizer instead of the multi-gpu one. This is
    # usually slower, but you might want to try it if you run into issues with
    # the default optimizer.
    "simple_optimizer": False,  # NOTE: This is for advanced usage (multi-gpu training)
    # Use a separate head for intrinsic value function?
    "intrinsic_head": True,  # NOTE: This was for our custom implementation with curiosity, don't need this yet
    # The number of agents in the environment - for joint curiosity
    "num_agents": 1,  # NOTE: This was used for the joint curiosity mechanism
})


def choose_policy_optimizer(workers, config):  # NOTE: This seems to have been copied verbatim
    if config["simple_optimizer"]:
        return SyncSamplesOptimizer(
            workers,
            num_sgd_iter=config["num_sgd_iter"],
            train_batch_size=config["train_batch_size"],
            sgd_minibatch_size=config["sgd_minibatch_size"],
            standardize_fields=["advantages"])

    return LocalMultiGPUOptimizer(
        workers,
        sgd_batch_size=config["sgd_minibatch_size"],
        num_sgd_iter=config["num_sgd_iter"],
        num_gpus=config["num_gpus"],
        rollout_fragment_length=config["rollout_fragment_length"],
        num_envs_per_worker=config["num_envs_per_worker"],
        train_batch_size=config["train_batch_size"],
        standardize_fields=["advantages"],
        shuffle_sequences=config["shuffle_sequences"])


def update_kl(trainer, fetches):  # NOTE: A PPO-specific method that is used to update the adaptive KL-divergence coefficient
    # Single-agent.
    if "kl" in fetches:
        trainer.workers.local_worker().for_policy(
            lambda pi: pi.update_kl(fetches["kl"]))

    # Multi-agent.
    else:

        def update(pi, pi_id):
            if pi_id in fetches:
                pi.update_kl(fetches[pi_id]["kl"])
            else:
                logger.debug("No data for {}, not updating kl".format(pi_id))

        trainer.workers.local_worker().foreach_trainable_policy(update)


def validate_config(config):  # NOTE: Copied verbatim
    if config["entropy_coeff"] < 0:
        raise DeprecationWarning("entropy_coeff must be >= 0")
    if isinstance(config["entropy_coeff"], int):
        config["entropy_coeff"] = float(config["entropy_coeff"])
    if config["sgd_minibatch_size"] > config["train_batch_size"]:
        raise ValueError(
            "Minibatch size {} must be <= train batch size {}.".format(
                config["sgd_minibatch_size"], config["train_batch_size"]))
    if config["multiagent"]["policies"] and not config["simple_optimizer"]:
        logger.info(
            "In multi-agent mode, policies will be optimized sequentially "
            "by the multi-GPU optimizer. Consider setting "
            "simple_optimizer=True if this doesn't work for you.")
    if config["simple_optimizer"]:
        logger.warning(
            "Using the simple minibatch optimizer. This will significantly "
            "reduce performance, consider simple_optimizer=False.")
    elif (tf and tf.executing_eagerly()):
        config["simple_optimizer"] = True  # multi-gpu not supported


def get_policy_class(config):  # NOTE: Seems to have been copied verbatim
    return PPOTFPolicy


PPOTrainer = build_trainer(
    name="PPO",
    default_config=DEFAULT_CONFIG,
    default_policy=PPOTFPolicy,
    get_policy_class=get_policy_class,
    make_policy_optimizer=choose_policy_optimizer,
    validate_config=validate_config,
    after_optimizer_step=update_kl)  # NOTE: Seems to be identical to the default RLLib PPO, but imports a different definition of 'PPOTFPolicy' that supports intrinsic reward