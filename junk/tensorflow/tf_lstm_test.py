'''Tensorflow implementation of behavioral cloning with an LSTM network'''
import gym
from gym.spaces import Discrete, Box
import numpy as np
import tensorflow as tf

class MemoryGame(gym.Env):
    '''An instance of the memory game with noisy observations'''

    def __init__(self, min_length=4, max_length=8, num_cues=2, noise=0.1):
        self.observation_space = Box(0, 2, shape=(num_cues + 2,))
        self.action_space = Discrete(num_cues)
        self._min_length = min_length
        self._max_length = max_length
        self._num_cues = num_cues 
        self._noise = noise

        self._current_step = None
        self._current_cue = None
        self._current_length = None

    def _obs(self):
        obs = np.random.uniform(0, self._noise, self.observation_space.shape)
        if 0 == self._current_step:
            obs[-2] += 1
            obs[self._current_cue] += 1
        elif self._current_length == self._current_step:
            obs[-1] += 1
        return obs

    def reset(self):
        self._current_step = 0
        self._current_cue = np.random.randint(self._num_cues)
        self._current_length = np.random.randint(self._min_length, 
            self._max_length)
        return self._obs()

    def step(self, action):
        if self._current_step < self._current_length:
            self._current_step += 1
            return self._obs(), 0, False, {}
        else:
            reward = (1 if action == self._current_cue else 0)
            return self._obs(), reward, True, {}

    def expert(self):
        if self._current_step < self._current_length:
            return self.action_space.sample()
        else:
            return self._current_cue


def generate_data(env, num_episodes=100):
    obs_seqs = []
    action_seqs = []

    for _ in range(num_episodes):
        observations = []
        actions = []

        obs = env.reset()
        done = False

        while not done:
            action = env.expert()

            observations.append(obs)
            actions.append(action)

            obs, _, done, _ = env.step(action)
    
        observations = np.array(observations, dtype=np.float32)
        actions = np.array(actions, np.int32)

        obs_seqs.append(observations)
        action_seqs.append(actions)
    
    return obs_seqs, action_seqs


class LSTMNet(tf.keras.Model):

    def __init__(self, output_size, hidden_sizes):
        super(LSTMNet, self).__init__()
        cells = []

        for idx, size in enumerate(hidden_sizes):
            cells.append(tf.keras.layers.LSTMCell(size, name=f"lstm_{idx}"))

        cells = tf.keras.layers.StackedRNNCells(cells)
        self.lstm = tf.keras.layers.RNN(cells, 
            return_sequences=True, return_state=True)
        
        self.linear = tf.keras.layers.Dense(output_size, 
            activation=None, name="output")
    
    def call(self, x, state=None):
        output, *state = self.lstm(x, initial_state=state)
        output = self.linear(output)

        return output, state


class DenseNet(tf.keras.Model):

    def __init__(self, output_size, hidden_sizes, activation="tanh"):
        super(DenseNet, self).__init__()
        layers = []

        for idx, size in enumerate(hidden_sizes):
            layers.append(tf.keras.layers.Dense(size, 
                activation=activation, name=f"hidden_{idx}"))
        
        layers.append(tf.keras.layers.Dense(output_size, 
            activation=None, name="output"))

        self.dense = tf.keras.Sequential(layers)
    
    def call(self, x):
        return self.dense(x), None


if __name__ == "__main__":

    # Learning config
    num_episodes = 1024
    num_epochs = 50
    num_batches = 100
    batch_size = 32
    lr = 0.001

    # Generate data
    env = MemoryGame(min_length=2, max_length=3, num_cues=4, noise=0.1)
    observations, actions = generate_data(env, num_episodes=num_episodes)

    # Convert trajectories to tensors
    observations = tf.ragged.constant(observations, dtype=tf.float32)
    actions = tf.ragged.constant(actions, dtype=tf.int32)
    actions = tf.one_hot(actions, env.action_space.n, dtype=tf.float32)

    # Test LSTM evaluation
    '''
    obs = tf.gather(observations, np.random.randint(0, num_episodes, 5))

    cells = tf.keras.layers.StackedRNNCells([
        tf.keras.layers.LSTMCell(64),
        tf.keras.layers.LSTMCell(32),
        tf.keras.layers.LSTMCell(16)
    ])
    lstm = tf.keras.layers.RNN(cells, return_sequences=True, return_state=True)
    linear = tf.keras.layers.Dense(4)

    output, *state = lstm(obs.to_tensor())

    print(f"lstm output: {output.shape}")
    print(f"linear output: {linear(output).shape}")

    for idx, (hidden, cell) in enumerate(state):
        print(f"lstm {idx}:")
        print(f"    hidden: {hidden.shape}")
        print(f"    cell: {cell.shape}")

    lstm = LSTMNet(4, [64, 32, 16])
    output, state = lstm(obs.to_tensor())

    print(f"output: {output.shape}")

    for idx, (hidden, cell) in enumerate(state):
        print(f"lstm {idx}:")
        print(f"    hidden: {hidden.shape}")
        print(f"    cell: {cell.shape}")

    exit()
    '''

    # Build dense network
    # model = DenseNet(env.action_space.n, [32])

    # Build LSTM network
    model = LSTMNet(env.action_space.n, [10])

    # Initialize optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Train model
    for epoch in range(num_epochs):
        print(f"\nepoch {epoch}")

        for batch in range(num_batches):

            # Construct and pad batch
            indices = np.random.randint(0, num_episodes, batch_size)
            obs_batch = tf.gather(observations, indices)
            action_batch = tf.gather(actions, indices)

            mask = tf.sequence_mask(obs_batch.row_lengths(), dtype=tf.float32)
            obs_batch = obs_batch.to_tensor()
            action_batch = action_batch.to_tensor()

            # Do gradient update
            with tf.GradientTape() as tape:
                logits, state = model(obs_batch, training=True)

                losses = tf.nn.softmax_cross_entropy_with_logits(action_batch, logits)
                loss = tf.reduce_mean(mask * losses)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print(f"batch loss: {loss}")

        # Compute error
        mask = tf.sequence_mask(observations.row_lengths(), dtype=tf.float32)
        obs_batch = observations.to_tensor()
        action_batch = actions.to_tensor()

        logits, state = model(obs_batch)
        preds = tf.math.argmax(logits, axis=-1, output_type=tf.int32)
        action_batch = tf.math.argmax(action_batch, axis=-1, output_type=tf.int32)
        errors = tf.cast(tf.not_equal(action_batch, preds), dtype=tf.float32)

        print(f"prediction error: {tf.reduce_mean(mask * errors) * 100}%")

    # Export Tensorflow model to file

    # Import Tensorflow model from file
