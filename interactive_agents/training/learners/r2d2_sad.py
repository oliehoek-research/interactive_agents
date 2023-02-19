"""
Implementation of the simplified action 
decoder (SAD) as described in the paper "Simplified 
action decoder for deep multi-agent reinforcement 
learning." (Hu et al. 2019).

Really just a modified version of R2D2 that returns two actions, 
one exploration and one exploitation, and allowing both of these 
actions to be seen by the other players.
"""
from collections import defaultdict
import numpy as np
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam

from interactive_agents.training.learners.models import build_model
from interactive_agents.training.learners.priority_tree import PriorityTree
from interactive_agents.sampling import Batch  # NOTE: The only thing we use this for is the "key" strings

class SADNetwork(nn.Module):

    def __init__(self, obs_size, num_actions, model_config, dueling=False):
        super(SADNetwork, self).__init__()
        self._num_actions = num_actions
        self._dueling = dueling

        if dueling:
            num_features = self._num_actions + 1
        else:
            num_features = self._num_actions

        # NOTE: For now, only support vector observations
        input_shape = (obs_size + num_actions,)
        self._model = build_model(input_shape, num_features, model_config)

    # NOTE: The type of "state" only works for LSTM, so does this serialize with GRU?
    def forward(self, 
            obs: torch.Tensor,
            other_action: torch.Tensor,  #NOTE: We assume these are input 1-hot encoded
            state: Tuple[torch.Tensor, torch.Tensor]):
        
        # NOTE: Why was this included, why was it removed?
        # if state is None:
        #    state = self._model.initial_state(obs.shape[1], str(obs.device))
        
        inputs = torch.cat((obs, other_action), dim=-1)
        features, state = self._model(obs, state)

        if self._dueling:
            A = features[:,:,:self._num_actions]
            V = features[:,:,-1:]
            Q = V + A - A.mean(dim=-1, keepdim=True)  # NOTE: Prevents A from learning the value function itself
        else:
            Q = features

        return Q, state

    # NOTE: How do we trace-export an auxilliary method like this?  Maybe one of the reasonse we used scripting?
    @torch.jit.export
    def initial_state(self, batch_size: int=1, device: str="cpu"):
        return self._model.initial_state(batch_size, device)


# NOTE: This is only used for serialization, don't need it at the moment
class SADPolicy(nn.Module):
    """Torchscript-compatible wrapper for greedy policies from a Q-network"""

    def __init__(self, model):
        super(QPolicy, self).__init__()
        self._model = model

    def forward(self, 
            obs: torch.Tensor, 
            state: Tuple[torch.Tensor, torch.Tensor]):
        Q, state = self._model(obs, state)
        return Q.argmax(-1), state
    
    @torch.jit.export
    def initial_state(self, batch_size: int=1, device: str="cpu"):
        return self._model.initial_state(batch_size, device)

# NOTE: Generalize this so it can handle arbitrary imputs
class RecurrentReplayBuffer:
    
    def __init__(self, capacity, prioritize=True, device="cpu"):
        self._capacity = capacity
        self._device = device

        self._next_index = 0
        self._samples = []

        if prioritize:
            self._priorities = PriorityTree(capacity)
        else:
            self._priorities = None

    def add(self, samples, priorities):
        indices = []
        for sample in samples:
            for key in sample:  # NOTE: Make sure things are actuallly getting stored on the GPU
                sample[key] = torch.as_tensor(sample[key], device=self._device)  # NOTE: Assumes all inputs are either numpy arrays or tensors

            if len(self._samples) < self._capacity:
                self._samples.append(sample)
            else:
                self._samples[self._next_index] = sample
            
            indices.append(self._next_index)
            self._next_index = (self._next_index + 1) % self._capacity
        
        if self._priorities is not None:
            priorities = np.asarray(priorities, dtype=np.float32)
            self._priorities.set(indices, priorities)

    def update_priorities(self, indices, priorities):
        if self._priorities is not None:
            priorities = np.asarray(priorities, dtype=np.float32)
            self._priorities.set(indices, priorities)
    
    def _sample_priority(self, batch_size, beta):
        masses = np.random.random(batch_size) * self._priorities.sum()
        indices = [self._priorities.prefix_index(m) for m in masses]
        
        priorities = self._priorities.get(indices)
        weights = (len(self._samples) * priorities) ** (-beta)

        p_min = self._priorities.min() / self._priorities.sum()
        max_weight = (len(self._samples) * p_min) ** (-beta)

        return indices, weights / max_weight

    def sample(self, batch_size, beta):
        if self._priorities is None:
            weights = np.full(batch_size, 1.0)
            indices = np.random.randint(0, len(self._samples), batch_size)
        else:
            indices, weights = self._sample_priority(batch_size, beta)

        batch = defaultdict(list)
        for idx in indices:
            for key, value in self._samples[idx].items():
                batch[key].append(value)

        seq_lens = [len(rewards) for rewards in batch[Batch.REWARD]]

        for key in batch.keys():
            batch[key] = nn.utils.rnn.pad_sequence(batch[key]).detach()

        return batch, weights, seq_lens, indices

# NOTE: Bake these into the learner itself
class R2D2Agent:  # NOTE: Represents an agent with a private state

    def __init__(self, actor, state):
        self._actor = actor
        self._state = state

    def act(self, obs):
        action, q_values, self._state = self._actor.act(obs, self._state)
        
        return action, {
            "q_values": q_values,
            "action_q": q_values[action]
        }  # NOTE: Seems that we already support returning extra information, just doesn't get passed to other agents

# NOTE: Need to add support for batches execution for better GPU utilization

# NOTE: Move some of this code outside of R2D2 - also, we aren't really doing distributed training
class R2D2Policy:

    def __init__(self, 
            obs_space=None, 
            action_space=None, 
            network_config={},
            dueling=True, 
            epsilon=0,
            device="cpu"):  # NOTE: Need to update epsilon over time
        self._action_space = action_space
        self._epsilon = epsilon
        self._device = device

        self._q_network = QNetwork(obs_space, 
            action_space, network_config, dueling).to(device)
    
    def act(self, obs, state):
        with torch.no_grad():
            obs = torch.as_tensor(obs, 
                dtype=torch.float32, device=self._device)
            obs = obs.unsqueeze(0)  # Add batch dimension  # NOTE: Ideally we won't use this in the future
            obs = obs.unsqueeze(0)  # Add time dimension (for RNNs)

            # state = state.detach()  # NOTE: May not be necessary in no_grad() context

            q_values, state = self._q_network(obs, state)
            q_values = q_values.detach().cpu().numpy()  # Convert back to a numpy array
            q_values = q_values.squeeze(0)  # Remove time dimension
            q_values = q_values.squeeze(0)  # Remove batch dimension

            if np.random.random() <= self._epsilon:
                action = self._action_space.sample()
            else:
                action = q_values.argmax()

        return action, q_values, state

    def make_agent(self):
        return R2D2Agent(self, self._q_network.initial_state(1, self._device))

    def update(self, updates):
        self._epsilon = updates["epsilon"]  # NOTE: Now need to return both exploration and exploitation actions
        self._q_network.load_state_dict(updates["network"])


# NOTE: To implement simplified action decoders, we will need something more flexible
class R2D2_SAD:

    def __init__(self, obs_space, action_space, config={}, device='cpu'):
        self._obs_space = obs_space
        self._action_space = action_space
        self._iteration_episodes = config.get("iteration_episodes", 128)
        self._num_batches = config.get("num_batches", 16)
        self._batch_size = config.get("batch_size", 16)
        self._sync_iterations = config.get("sync_iterations", 1)
        self._learning_starts = config.get("learning_starts", 10)
        self._gamma = config.get("gamma", 0.99)
        self._beta = config.get("beta", 0.5)
        self._double_q = config.get("double_q", True)
        self._device = device

        # Epsilon-greedy exploration
        self._epsilon = config.get("epsilon_initial", 1)
        self._epsilon_iterations = config.get("epsilon_iterations", 1000)
        self._epsilon_decay = self._epsilon - config.get("epsilon_final", 0.01)
        self._epsilon_decay /= self._epsilon_iterations

        # Replay buffer
        self._replay_alpha = config.get("replay_alpha", 0.6)
        self._replay_epsilon = config.get("replay_epsilon", 0.01)
        self._replay_eta = config.get("replay_eta", 0.5)
        self._replay_beta = 0.0
        self._replay_beta_step = 1.0 / config.get("replay_beta_iterations", 1000)
        self._replay_prioritize = 0.0 != self._replay_alpha
        self._replay_buffer = RecurrentReplayBuffer(
            config.get("buffer_size", 2048), self._replay_prioritize, self._device)

        # Q-Networks
        self._network_config = config.get("model_config", {})
        self._dueling = config.get("dueling", True)

        self._online_network = QNetwork(obs_space, 
            action_space, self._network_config, self._dueling)
        self._target_network = QNetwork(obs_space, 
            action_space, self._network_config, self._dueling)

        self._online_q.to(device)
        self._target_q.to(device)

        # Optimizer
        self._optimizer = Adam(self._online_q.parameters(), lr=config.get("lr", 0.01))

        # Track number of training iterations
        self._current_iteration = 0
    
    def _priority(self, td_errors):
        abs_td = np.abs(td_errors)
        max_error = abs_td.max(axis=0)
        mean_error = abs_td.mean(axis=0)

        priority = self._replay_eta * max_error + (1 - self._replay_eta) * mean_error
        return (priority + self._replay_epsilon) ** self._replay_alpha

    def _loss(self, batch, weights, seq_lens):
        h0 = self._online_network.initial_state(len(weights), self._device)

        mask = [torch.ones(l, device=self._device) for l in seq_lens]
        mask = nn.utils.rnn.pad_sequence(mask)

        weights = torch.as_tensor(weights, dtype=torch.float32, device=self._device)

        # NOTE: The torch error is here
        online_q, _ = self._online_network(batch[Batch.OBS], h0)
        target_q, _ = self._target_network(batch[Batch.NEXT_OBS], h0)

        if self._double_q:
            max_actions = online_q.argmax(-1).unsqueeze(-1)
            target_q = torch.gather(target_q, -1, max_actions).squeeze(-1)
        else:
            target_q, _ = target_q.max(-1)

        online_q = torch.gather(online_q, -1, batch[Batch.ACTION].unsqueeze(-1)).squeeze(-1)

        q_targets = batch[Batch.REWARD] + self._gamma * (1 - batch[Batch.DONE]) * target_q
        q_targets = q_targets.detach()

        errors = nn.functional.smooth_l1_loss(online_q, q_targets, beta=self._beta, reduction='none')
        loss = torch.mean(weights * torch.mean(mask * errors, 0))

        td_errors = torch.abs(online_q - q_targets)
        return loss, td_errors

    def _learn(self):
        outputs = defaultdict(list)

        for _ in range(self._num_batches):   
            # with torch.autograd.set_detect_anomaly(True):
                # Sample batch 
                batch, weights, seq_lengths, indices = \
                    self._replay_buffer.sample(self._batch_size, self._beta)
                
                outputs["replay_weight"].append(weights)

                # Compute loss and do gradient update
                self._optimizer.zero_grad()
                loss, td_errors = self._loss(batch, weights, seq_lengths)
                loss.backward()
                self._optimizer.step()

                td_errors = td_errors.detach().cpu().numpy()
                outputs["td_error"].append(td_errors.mean())
                outputs["loss"].append(loss.detach().cpu().numpy())

                # Update replay priorities
                priorities = self._priority(td_errors)
                self._replay_buffer.update_priorities(indices, priorities)

                outputs["priorities"].append(priorities)
        
        stats = {}
        for key, value in outputs.items():
            value = np.stack(value)
            stats[key + "_mean"] = value.mean()
            stats[key + "_max"] = value.max()
            stats[key + "_min"] = value.min()
        
        return stats

    # NOTE: This will change as we add new data
    def learn(self, episodes):  # NOTE: For offline RL, we would want managing data to be separate from the training loop
        stats = {}

        # Compute initial priorities and add new episodes to replay buffer
        if 0.0 != self._replay_alpha:
            priorities = []
            for episode in episodes:
                max_q = episode["q_values"][1:].max(-1)
                q_targets = episode[Batch.REWARD].copy()
                q_targets[:-1] += self._gamma * episode[Batch.DONE][:-1] * max_q
                
                priorities.append(self._priority(episode["action_q"] - q_targets))

                del episode["q_values"]
                del episode["action_q"]
        else:
            priorities = None

        self._replay_buffer.add(episodes, priorities)

        # Sync online and target networks at fixed intervals
        if self._current_iteration % self._sync_iterations == 0:
            parameters = self._online_network.state_dict()
            self._target_network.load_state_dict(parameters)

        # Do training updates
        if self._current_iteration >= self._learning_starts:
            stats.update(self._learn())

        # Update exploration rate
        if self._current_iteration < self._epsilon_iterations:
            self._epsilon -= self._epsilon_decay
        stats["epsilon"] = self._epsilon

        # Update replay beta
        self._replay_beta = min(1.0, self._replay_beta + self._replay_beta_step)
        stats["replay_beta"] = self._replay_beta

        # Increment iteration
        self._current_iteration += 1

        return stats

    def act(self, obs, other_actions, states):
        with torch.no_grad():
            obs = torch.as_tensor(obs, 
                dtype=torch.float32, device=self._device)
            obs = obs.unsqueeze(0)  # Add batch dimension  # NOTE: Ideally we won't use this in the future
            obs = obs.unsqueeze(0)  # Add time dimension (for RNNs)

            # state = state.detach()  # NOTE: May not be necessary in no_grad() context

            q_values, state = self._q_network(obs, state)
            q_values = q_values.detach().cpu().numpy()  # Convert back to a numpy array
            q_values = q_values.squeeze(0)  # Remove time dimension
            q_values = q_values.squeeze(0)  # Remove batch dimension

            if np.random.random() <= self._epsilon:
                action = self._action_space.sample()
            else:
                action = q_values.argmax()

        return action, q_values, state

    # NOTE: This will create an initial state on the same device as the policy itself
    def initial_state(self, batch_size):
        pass
