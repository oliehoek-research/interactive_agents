"""
Defines a reinforcement learning agent which uses a
simple version of proximal policy optimization, without an
actor-critic style baseline value function.
"""

# NOTE: It seems the documentation was wrong, this class implements a critic, while the other class doesn't

import tensorflow as tf
import numpy as np


class Sample:  # NOTE: Seems to store a state-action pair along with a monte-carlo value
    """
    Represents an arbitrary state-action sample.  Mainly to overcome the
    issue with Numpy treating arrays of tuples as two-dimensional arrays.
    """

    def __init__(self, state, action, value):
        """
        Initializes the sample.
        :param state: the state
        :param action: the action taken
        :param value: the value of the sample
        """

        self.state = state
        self.action = action
        self.value = value


class Trajectory:
    """
    Represents a state action trajectory
    """

    def __init__(self):
        self._states = []
        self._actions = []
        self._rewards = []

    def __len__(self):
        return len(self._states)

    def step(self, state, action):
        """
        Adds a new step to the trajectory.
        :param state: the state
        :param action: the action
        """

        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(0.0)

    def reward(self, reward):
        """
        Adds to the reward value of the most recent step.
        :param reward: the immediate reward
        """

        if len(self._rewards) != 0:
            self._rewards[-1] += reward

    @property
    def states(self):
        return self._states

    @property
    def actions(self):
        return self._actions

    @property
    def rewards(self):
        return self._rewards


class Agent:
    """
    An online RL agent which updates its policy
    using a version of the PPO algorithm.
    """

    def __init__(self, graph, session, **kwargs):
        """
        Initializes a new PPO agent.
        :param kwargs: the configuration options for the agent
        """

        # Get the state and action spaces
        state_space = kwargs['state_space']
        action_space = kwargs['action_space']

        # Capture the configuration parameters needed
        self._discrete_action = action_space.discrete
        self._discount = kwargs['discount']
        self._mixing = kwargs['mixing']  # NOTE: What is this used for?
        self._batch_size = kwargs['batch_size']
        self._num_batches = kwargs['num_batches']
        self._num_episodes = kwargs['num_episodes']

        # Capture the session
        self._session = session

        # Build the policy network and learning update graph
        with graph.as_default():  # NOTE: Using tensorflow, so maybe not the best reference point

            # Define common state input
            self._state_input = tf.placeholder(dtype=tf.float32, shape=[None] + list(state_space.shape))

            # Define critic
            with tf.variable_scope("critic"):
                self._critic = kwargs['critic_fn'](self._state_input)[:, 0]  # NOTE: This class does implement a critic, in spite of the initial comment

            # Define target actor
            with tf.variable_scope("target_actor"):
                target_actor = kwargs['actor_fn'](self._state_input)

            # Define hypothesis actor
            with tf.variable_scope("hypothesis_actor"):  # NOTE: Why do we maintain two actors? (does RLLib do this?)
                hypothesis_actor = kwargs['actor_fn'](self._state_input)

            # Define critic parameter transfer op
            target_actor_vars = tf.trainable_variables(scope='target_actor')  # NOTE: It seems like there should be two critics, not two actors - doesn't quite make sense
            target_actor_vars = dict(map(lambda x: (x.name[len('target_actor'):], x), target_actor_vars))

            hypothesis_actor_vars = tf.trainable_variables(scope='hypothesis_actor')
            hypothesis_actor_vars = dict(map(lambda x: (x.name[len('hypothesis_actor'):], x), hypothesis_actor_vars))

            self._transfer_actor = []

            for key, var in hypothesis_actor_vars.items():
                self._transfer_actor.append(tf.assign(target_actor_vars[key], var))  # NOTE: Seems we always move from hypothesis to target

            # Define action input and policy ratios
            if action_space.discrete:

                # Action input
                self._action_input = tf.placeholder(dtype=tf.int32, shape=[None])
                one_hot = tf.one_hot(self._action_input, action_space.size)

                # Probability ratios  # NOTE: Computes action distributions for both networks
                target = tf.exp(target_actor)
                target = tf.reduce_sum(one_hot * target, axis=1) / tf.reduce_sum(target, axis=1)

                hypothesis = tf.exp(hypothesis_actor)
                hypothesis = tf.reduce_sum(one_hot * hypothesis, axis=1) / tf.reduce_sum(hypothesis, axis=1)

                ratio = hypothesis / tf.stop_gradient(target)  # NOTE: Why do we compute this ratio?  How is it used

                # Action output
                self._action = tf.multinomial(policy_output, 1)  # NOTE: This variable is not defined, seems we never finished PPO
            else:

                # Action input
                self._action_input = tf.placeholder(dtype=tf.float32, shape=[None] + list(action_space.shape))

                # Probability ratios
                target_mean = target_actor[:, 0]
                target_deviation = target_actor[:, 1]

                hypothesis_mean = hypothesis_actor[:, 0]
                hypothesis_deviation = hypothesis_actor[:, 1]

                target = tf.square((self._action_input - target_mean) / tf.exp(target_deviation))
                target = tf.reduce_sum((0.5 * target) + target_deviation, axis=1)

                hypothesis = tf.square((self._action_input - hypothesis_mean) / tf.exp(hypothesis_deviation))
                hypothesis = tf.reduce_sum((0.5 * hypothesis) + hypothesis_deviation, axis=1)

                ratio = tf.exp(tf.stop_gradient(target) - hypothesis)

                # Action output
                noise = tf.random_normal(tf.shape(target_mean))
                self._action = target_mean + (noise * tf.exp(target_deviation))  # NOTE: Action deviation computed from the target network

            # Critic update
            self._value_input = tf.placeholder(dtype=tf.float32, shape=[None])

            loss = tf.reduce_mean(tf.square(self._value_input - self._critic))
            self._critic_update = tf.train.AdamOptimizer(learning_rate=kwargs['learning_rate']).minimize(loss)  # NOTE: Maintains separate optimizers for policy and critic, which means they can't share layers

            # Actor update
            self._advantage_input = tf.placeholder(dtype=tf.float32, shape=[None])

            clipped_ratio = tf.clip_by_value(ratio, 1.0 - kwargs['clip_epsilon'], 1.0 + kwargs['clip_epsilon'])  # NOTE: Seems that the target and hypothesis ratio is used to compute importance weights
            loss = -tf.reduce_mean(tf.minimum(ratio * self._advantage_input, clipped_ratio * self._advantage_input))

            self._actor_update = tf.train.AdamOptimizer(learning_rate=kwargs['learning_rate']).minimize(loss)

            # Initialize variables
            session.run(tf.global_variables_initializer())

        # Initialize internal state
        self._trajectories = []
        self._trajectory = None
        self._episode_count = 0

        self.reset()

    def _update(self):
        """
        Updates the agent's policy based on recent experience.
        """

        # Update critic
        samples = []

        for trajectory in self._trajectories:
            values = self._session.run(self._critic, feed_dict={self._state_input: trajectory.states})
            acc = 0.0

            for t in reversed(range(len(trajectory))):
                value = trajectory.rewards[t] + acc
                acc = self._discount * (((1.0 - self._mixing) * values[t]) + (self._mixing * value))
                samples.append(Sample(trajectory.states[t], trajectory.actions[t], value))

        for _ in range(self._num_batches):
            batch = np.random.choice(samples, self._batch_size, replace=True)
            states = []
            values = []

            for sample in batch:
                states.append(sample.state)
                values.append(sample.value)

            self._session.run(self._critic_update, feed_dict={
                self._state_input: states,
                self._value_input: values
            })

        # Update actor
        samples = []

        for trajectory in self._trajectories:
            values = self._session.run(self._critic, feed_dict={self._state_input: trajectory.states})
            acc = 0.0

            for t in reversed(range(len(trajectory))):
                value = trajectory.rewards[t] + acc
                acc = self._discount * (((1.0 - self._mixing) * values[t]) + (self._mixing * value))
                samples.append(Sample(trajectory.states[t], trajectory.actions[t], value - values[t]))

        for _ in range(self._num_batches):

            # Construct batch
            batch = np.random.choice(samples, self._batch_size, replace=True)
            states = []
            actions = []
            advantages = []

            for sample in batch:
                states.append(sample.state)
                actions.append(sample.action)
                advantages.append(sample.value)

            # Run update
            self._session.run(self._actor_update, feed_dict={
                self._state_input: states,
                self._action_input: actions,
                self._advantage_input: advantages
            })

        # Transfer parameters
        self._session.run(self._transfer_actor)

    def reset(self):
        """
        Tells the agent that a new episode has been started, the agent may
        choose to run an update at this time.
        """

        if self._trajectory is not None and len(self._trajectory) != 0:
            self._trajectories.append(self._trajectory)
            self._episode_count += 1

        if self._episode_count == self._num_episodes:
            self._update()  # NOTE: Only does training updates on episode starts (makes sense since usually we do multiple updates per episode)
            self._trajectories = []
            self._episode_count = 0

        self._trajectory = Trajectory()

    def act(self, state, evaluation=False):  # NOTE: Responsible for recording its own actions and observations
        """
        Records the current state, and selects the agent's action
        :param state: a representation of the current state
        :param evaluation: if true, this indicates that this action should not be recorded
        :return: a representation of the next action
        """

        action = self._session.run(self._action, feed_dict={self._state_input: [state]})
        action = action[0, 0] if self._discrete_action else action[0]

        if not evaluation:
            self._trajectory.step(state, action)

        return action

    def reward(self, reward):
        """
        Gives an immediate reward to the agent for the most recent step.
        :param reward: the reward value
        """

        self._trajectory.reward(reward)


def manager(actor_fn, critic_fn, state_space, action_space,
            discount=0.99,
            mixing=0.9,
            learning_rate=0.0005,
            clip_epsilon=0.05,
            batch_size=50,
            num_batches=20,
            num_episodes=10):  # NOTE: This is only needed because we need to manage Tensorflow sessions
    """
    Returns a context manager which is used to instantiate and clean up
    an actor-critic PPO agent.
    :param actor_fn: the function used to build the actor graphs
    :param critic_fn: the function used to build the critic graphs
    :param state_space: the state space
    :param action_space: the action space
    :param discount: the discount factor of the MDP
    :param mixing: the mixing factor for the advantage estimators
    :param learning_rate: the learning rate used for training the policies
    :param clip_epsilon: the clipping radius for the policy ratio
    :param batch_size: the batch size used for training the policies
    :param num_batches: the number of gradient steps to do per update
    :param num_episodes: the number of episodes performed between updates
    :return: a context manager which creates a new PPO agent
    """

    class Manager:

        def __enter__(self):
            self._graph = tf.Graph()
            self._session = tf.Session(graph=self._graph)

            return Agent(self._graph, self._session,
                         actor_fn=actor_fn,
                         critic_fn=critic_fn,
                         state_space=state_space,
                         action_space=action_space,
                         discount=discount,
                         mixing=mixing,
                         learning_rate=learning_rate,
                         clip_epsilon=clip_epsilon,
                         batch_size=batch_size,
                         num_batches=num_batches,
                         num_episodes=num_episodes)

        def __exit__(self, exc_type, exc_val, exc_tb):
            """
            Closes the session associated with the current agent.
            :param exc_type: ignored
            :param exc_val: ignored
            :param exc_tb: ignored
            :return: always False, never suppress exceptions
            """

            self._session.close()
            self._session = None
            self._graph = None

            return False

    return Manager()