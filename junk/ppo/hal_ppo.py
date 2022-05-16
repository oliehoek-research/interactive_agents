"""
Defines a reinforcement learning agent which uses a
simple version of proximal policy optimization, without an
actor-critic style baseline value function.
"""

import tensorflow as tf
import numpy as np
import gym
import roboschool


class Sample:
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
        :param graph: the Tensorflow graph to be used
        :param session: the Tensorflow session to be used
        :param kwargs: the configuration options for the agent
        """

        # Get the state and action spaces
        state_space = kwargs['state_space']
        action_space = kwargs['action_space']

        # Capture the configuration parameters needed
        self._discrete_action = action_space.discrete
        self._discount = kwargs['discount']
        self._batch_size = kwargs['batch_size']
        self._num_batches = kwargs['num_batches']
        self._num_episodes = kwargs['num_episodes']

        # Capture session
        self._session = session

        # Build the policy network and learning update graph
        with graph.as_default():
            self._state_input = tf.placeholder(dtype=tf.float32, shape=[None] + list(state_space.shape))

            with tf.variable_scope("policy"):
                policy_output = kwargs['actor_fn'](self._state_input)

            with tf.variable_scope("hypothesis"):
                hypothesis_output = kwargs['actor_fn'](self._state_input)

            if action_space.discrete:
                self._action_input = tf.placeholder(dtype=tf.int32, shape=[None])

                one_hot = tf.one_hot(self._action_input, action_space.size)

                policy = tf.exp(policy_output)
                policy = tf.reduce_sum(one_hot * policy, axis=1) / tf.reduce_sum(policy, axis=1)

                hypothesis = tf.exp(hypothesis_output)
                hypothesis = tf.reduce_sum(one_hot * hypothesis, axis=1) / tf.reduce_sum(hypothesis, axis=1)

                ratio = hypothesis / tf.stop_gradient(policy)

                self._action_output = tf.multinomial(policy_output, 1)
            else:
                self._action_input = tf.placeholder(dtype=tf.float32, shape=[None] + list(action_space.shape))

                policy_mean = policy_output[:, 0]
                policy_deviation = policy_output[:, 1]

                hypothesis_mean = hypothesis_output[:, 0]
                hypothesis_deviation = hypothesis_output[:, 1]

                policy = tf.square((self._action_input - policy_mean) / tf.exp(policy_deviation))
                policy = tf.reduce_sum((0.5 * policy) + policy_deviation, axis=1)

                hypothesis = tf.square((self._action_input - hypothesis_mean) / tf.exp(hypothesis_deviation))
                hypothesis = tf.reduce_sum((0.5 * hypothesis) + hypothesis_deviation, axis=1)

                ratio = tf.exp(tf.stop_gradient(policy) - hypothesis)

                noise = tf.random_normal(tf.shape(policy_mean))
                self._action = policy_mean + (noise * tf.exp(policy_deviation))

            self._advantage_input = tf.placeholder(dtype=tf.float32, shape=[None])

            clipped_ratio = tf.clip_by_value(ratio, 1.0 - kwargs['clip_epsilon'], 1.0 + kwargs['clip_epsilon'])
            loss = -tf.reduce_mean(tf.minimum(ratio * self._advantage_input, clipped_ratio * self._advantage_input))

            self._update_hypothesis = tf.train.AdamOptimizer(learning_rate=kwargs['learning_rate']).minimize(loss)

            policy_variables = tf.trainable_variables(scope='policy')
            policy_variables = dict(map(lambda x: (x.name[len('policy'):], x), policy_variables))

            hypothesis_variables = tf.trainable_variables(scope='hypothesis')
            hypothesis_variables = dict(map(lambda x: (x.name[len('hypothesis'):], x), hypothesis_variables))

            self._transfer_hypothesis = []

            for key, var in hypothesis_variables.items():
                self._transfer_hypothesis.append(tf.assign(policy_variables[key], var))

            session.run(tf.variables_initializer(tf.global_variables()))

        self._trajectories = []
        self._trajectory = None
        self._episode_count = 0

        self.reset()

    def _update(self):
        """
        Updates the agent's policy based on recent experience.
        """

        # Compute advantages
        samples = []

        for trajectory in self._trajectories:
            advantage = 0.0

            for t in reversed(range(len(trajectory))):
                advantage += trajectory.rewards[t] + (self._discount * advantage)
                samples.append(Sample(trajectory.states[t], trajectory.actions[t], advantage))

        # Perform updates
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
            self._session.run(self._update_hypothesis, feed_dict={
                self._state_input: states,
                self._action_input: actions,
                self._advantage_input: advantages
            })

        # Transfer parameters
        self._session.run(self._transfer_hypothesis)

    def reset(self):
        """
        Tells the agent that a new episode has been started, the agent may
        choose to run an update at this time.
        """

        if self._trajectory is not None and len(self._trajectory) != 0:
            self._trajectories.append(self._trajectory)
            self._episode_count += 1

        if self._episode_count == self._num_episodes:
            self._update()
            self._trajectories = []
            self._episode_count = 0

        self._trajectory = Trajectory()

    def act(self, state, evaluation=False):
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


def manager(actor_fn, state_space, action_space,
            discount=0.99,
            learning_rate=0.0005,
            clip_epsilon=0.05,
            batch_size=50,
            num_batches=20,
            num_episodes=10):
    """
    Returns a context manager which is used to instantiate and clean up
    a PPO reinforcement learning agent with the given configuration.
    :param actor_fn: the function used to build the policy graphs
    :param state_space: the number of state features
    :param action_space: the number of actions or action features
    :param discount: the discount factor of the MDP
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
                         state_space=state_space,
                         action_space=action_space,
                         discount=discount,
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