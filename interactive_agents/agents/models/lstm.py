import numpy as np
import torch
import torch.nn as nn

class LSTMNet(nn.Module):
    
    def __init__(self, observation_space, action_space, hidden_size=32, hidden_layers=1, deuling=False):
        super(LSTMNet, self).__init__()
        self._hidden_size = hidden_size
        self._hidden_layers = hidden_layers
        self._deuling = deuling

        self._lstm = nn.LSTM(np.prod(observation_space.shape), hidden_size, hidden_layers)
        self._q_function = nn.Linear(hidden_size, action_space.n)

        if deuling:
            self._value_function = nn.Linear(hidden_size, 1)

    def forward(self, obs, hidden):
        # obs = obs.flatten(2)  # NOTE: We may need to pre-flatten inputs so they can be packed
        outputs, hidden = self._lstm(obs, hidden)
        Q = self._q_function(outputs)

        if self._deuling:
            V = self._value_function(outputs)
            Q += V - Q.mean(2, keepdims=True)

        return Q, hidden

    def get_h0(self, batch_size=1):
        hidden = torch.zeros((self._hidden_layers, batch_size, self._hidden_size), dtype=torch.float32)
        cell = torch.zeros((self._hidden_layers, batch_size, self._hidden_size), dtype=torch.float32)
        return hidden, cell