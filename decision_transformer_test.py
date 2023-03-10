"""
Test of decision transformers on simple tasks.
"""
from collections import defaultdict
from gymnasium.spaces import Box
import numpy as np
import time

import torch
import torch.nn

class FactoredTransformer(torch.nn.Module):

    # NOTE: There is no such thing as a decision transformer, just 
    # Transformers used to represent policies.  This class simply
    # represents a transformer that takes in dictionaries of
    # sequences of inputs with potentially different shapes.
    #
    # We assume that embedding of discrete values has already occured.
    #
    # We'll also assume that the input and output factors are identical.
    #
    def __init__(self, 
                 input_factors,
                 output_factors,
                 max_len=5000,
                 model_dim=256,
                 num_layers=2,
                 num_heads=8, 
                 num_hidden=256,
                 non_linear=True,
                 pos_encoding="embedded",
                 dropout=0.2):
        super().__init__()
        
        # Configure input transformations
        self._non_linear = non_linear
        self._input_factors = list(input_factors.keys())
        
        input_dim = 0
        for factor in input_factors.values():
            input_dim += factor.shape[0]  # NOTE: Only supporting flat observations for now.
        
        if self._non_linear:
            self._input_encoder = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, model_dim),
                    torch.nn.ReLU())
        else:
            self._input_encoder = torch.nn.Linear(input_dim, model_dim)
        
        # Configure Positional Encoding
        if pos_encoding is None:
            self._pos_encoding = None
        elif "embedded" == pos_encoding:
            embedding = torch.zeros(max_len, 1, model_dim)
            torch.nn.init.normal_(embedding)
            self._pos_encoding = torch.nn.Parameter(embedding)  # NOTE: The singleton is the batch dimension
            self.register_parameter("position_encoding", self._pos_encoding)
        else:
            raise NotImplementedError(f"Position embedding type '{pos_encoding}' not supported")

        # Configure Transformer
        encoder_layer = torch.nn.TransformerEncoderLayer(model_dim, num_heads, num_hidden, dropout)
        self._transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers)

        # Configure output decoders
        decoders = {}
        for key, factor in output_factors.items():
            decoders[key] = torch.nn.Linear(model_dim, factor.shape[0])
        self._decoders = torch.nn.ModuleDict(decoders)

    def forward(self, sequences, mask):
        inputs = [sequences[key] for key in self._input_factors]
        inputs = self._input_encoder(torch.concat(inputs, dim=-1))

        if self._pos_encoding is not None:
            inputs = inputs + self._pos_encoding[:inputs.size(0)]

        outputs = self._transformer_encoder(inputs, mask)

        sequences = {}
        for key, decoder in self._decoders.items():
            sequences[key] = decoder(outputs)
        
        return sequences
    

class MemoryGame:

    def __init__(self, length=5, num_cues=2):
        self._length = length
        self._num_cues = num_cues

        self._obs_shape = (self._num_cues + 2,)
        self.observation_space = Box(0, 2, shape=self._obs_shape)
        self.action_space = Box(0, 1, (self._num_cues,))
  
        self._current_step = 0
        self._current_cue = 0

    def _obs(self):
        obs = np.zeros(self._obs_shape)

        if 0 == self._current_step:
            obs[-2] = 1
            obs[self._current_cue] = 1
        elif self._length == self._current_step:
            obs[-1] = 1

        return obs

    def reset(self):
        self._current_step = 0
        self._current_cue = np.random.randint(self._num_cues)
        return self._obs()

    def step(self, action):
        if self._current_step < self._length:
            self._current_step += 1
            return self._obs(), 0, False
        else:
            reward = 1 if np.argmax(action) == self._current_cue else 0
            return self._obs(), reward, True
    
    def action(self, policy="random"):
        action = np.zeros(self._num_cues)

        if self._length == self._current_step:
            if policy == "random":
                action[np.random.randint(self._num_cues)] = 1
            elif policy == "expert":
                action[self._current_cue] = 1
            else:
                raise ValueError(f"Policy '{policy}' not defined")

        return action


class ListenerGame:

    def __init__(self, num_stages=32, num_cues=4, identity=False):  # NOTE: Do we ever actually use "spec_only" anywhere?
        self._num_stages = num_stages
        self._num_cues = num_cues
        self._identity = identity

        self.observation_space = Box(0, 1, shape=(self._num_cues * 2,))
        self.action_space = Box(0, 1, (self._num_cues,))

        self._stage = None
        self._cue = None
        self._mapping = None

    def reset(self):
        if self._identity:
            self._mapping = np.flip(np.arange(self._num_cues))
        else:
            self._mapping = np.random.permutation(self._num_cues)
        
        self._stage = 0
        self._cue = np.random.randint(self._num_cues)

        obs = np.zeros(self._num_cues * 2)
        obs[self._cue] = 1
        
        return obs

    def step(self, action):
        action = np.argmax(action)
        reward = 1 if self._mapping[self._cue] == action else 0

        self._stage += 1
        done = (self._stage >= self._num_stages)

        obs = np.zeros(self._num_cues * 2)
        if not done:
            obs[self._num_cues + self._mapping[self._cue]] = 1  # NOTE: Need to make sure the learner can observe the previous cue
            self._cue = np.random.randint(self._num_cues)
            obs[self._cue] = 1

        return obs, reward, done

    def action(self, policy="random"):
        action = np.zeros(self._num_cues)

        if policy == "random":
            action[np.random.randint(self._num_cues)] = 1
        elif policy == "expert":
            action[self._mapping[self._cue]] = 1
        else:
            raise ValueError(f"Policy '{policy}' not defined")

        return action 


def generate_episodes(env, policy, num_episodes=2000):
    episodes = defaultdict(list)
    max_steps = 0

    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        step = 0

        observations = []
        actions = []
        rewards = []

        while not done:
            action = env.action(policy)
            observations.append(obs)
            actions.append(action)

            obs, reward, done = env.step(action)
            rewards.append(reward)
            step += 1

        observations = np.stack(observations)
        actions = np.stack(actions)
        rewards = np.asarray(rewards)

        prev_actions = np.roll(actions, 1)
        prev_actions[0] = np.zeros(env.action_space.shape, dtype=np.float32)

        rtg = np.cumsum(rewards)
        rtg = rtg[-1] - rtg

        episodes["obs"].append(observations)
        episodes["action"].append(actions)
        episodes["prev_action"].append(prev_actions)
        episodes["rtg"].append(np.expand_dims(rtg, -1))

        if step > max_steps:
            max_steps = step
    
    return episodes, max_steps


def make_batches(episodes, device):
    batches = {}
    for key in episodes.keys():
        data = [torch.as_tensor(a, dtype=torch.float32) for a in episodes[key]]
        batches[key] = torch.nn.utils.rnn.pad_sequence(data).contiguous().to(device)
    
    return batches


def train(model,
          batches, 
          mask,
          epochs=4, 
          batch_size=32,
          lr=0.001,
          gamma=0.5,
          delta=0.5,
          log_interval=16):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
    loss_fn = torch.nn.HuberLoss(delta=delta)

    for epoch in range(epochs):
        interval_time = time.time()
        interval_loss = 0

        for batch_id, i in enumerate(range(0, batches["obs"].shape[1], batch_size)):
            batch = {}
            for key, value in batches.items():
                batch[key] = value[:,i:i + batch_size]

            preds = model(batch, mask)
            loss = 0
            for key in preds.keys():
                loss = loss + loss_fn(preds[key], batch[key])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            interval_loss += loss.cpu().item()
            if (batch_id + 1) % log_interval == 0:
                current_time = time.time() - interval_time
                
                print(f"\nEpoch {epoch}, Batch {batch_id}:")
                print(f"    interval loss: {interval_loss: .4f} | exp loss: {np.exp(interval_loss): .4f}")
                print(f"    interval took: {current_time: .2f}s")
                
                interval_loss = 0
                interval_time = time.time()
        
        scheduler.step()


def test_policy(env, policy="expert", num_episodes=100):
    total_reward = 0
    for _ in range(num_episodes):
        obs = env.reset()
        done = False

        while not done:
            action = env.action(policy)
            obs, reward, done = env.step(action)
            total_reward += reward

    return total_reward / num_episodes


def evaluate(env, model, max_len, mask, target_rtg, device, num_episodes=50):
    with torch.no_grad():
        input_sequences = {
            "obs": torch.zeros((max_len, 1, *env.observation_space.shape), device=device),
            "prev_action": torch.zeros((max_len, 1, *env.action_space.shape), device=device),
            "rtg": torch.zeros((max_len, 1, 1)).to(device),
        }
        
        total_reward = 0
        for _ in range(num_episodes):
            obs = env.reset()
            action = torch.zeros(env.action_space.shape, dtype=torch.float32, device=device)
            rtg = target_rtg
            done = False
            step = 0

            while not done and step < max_len:
                input_sequences["obs"][step][0] = torch.as_tensor(obs, dtype=torch.float32, device=device)
                input_sequences["prev_action"][step][0] = torch.as_tensor(action, dtype=torch.float32, device=device)
                input_sequences["rtg"][step][0][0] = rtg

                action = model(input_sequences, mask)["action"][step][0].cpu().numpy()
                obs, reward, done = env.step(action)

                rtg -= reward
                total_reward += reward
                step += 1

    return total_reward / num_episodes


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Training on torh device: {device}")

    # Build environment
    # env = MemoryGame(length=10, num_cues=10)
    env = ListenerGame(num_stages=32, num_cues=4, identity=False)
    
    # expert_reward = test_policy(env, policy="expert", num_episodes=100)
    # print(f"expert policy reward: {expert_reward}")
    # exit()

    # Generate data
    train_episodes, max_len = generate_episodes(env, policy="random", num_episodes=2**17)

    # Train Model
    input_factors = {
        "obs": env.observation_space,
        # "prev_action": env.action_space,
        "rtg": Box(-np.infty, np.infty, (1,))
    }
    output_factors = { "action": env.action_space }
    model = FactoredTransformer(input_factors, output_factors, max_len).to(device)
    mask = torch.nn.Transformer.generate_square_subsequent_mask(max_len).to(device)

    train_batches = make_batches(train_episodes, device)
    train(model, train_batches, mask)

    # Evaluate model directly in the environment
    model.eval()
    average_return = evaluate(env, model, max_len, mask, 28, device)
    print("\nEVALUATION:")
    print(f"\n   average return: {average_return}")
