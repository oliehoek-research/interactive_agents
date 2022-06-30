"""
Network classes and builder methods.  Currently supports dense networks, LSTM/GRUs, and CNNs.
"""
import numpy as np
import os.path
import torch
import torch.nn as nn
from typing import Optional, Tuple

DEFAULTS = {
    "model": "lstm",
    "hidden_layers": 1,
    "hidden_size": 32,
    "activation": "relu",
    "conv_filters": [
            (8, (5, 5), 2), 
            (16, (5, 5), 2), 
            (16, (5, 5), 2)],
    "conv_activation": "selu",
}

def build_model(input_shape, num_outputs, model_config):
    config = DEFAULTS.copy()
    config.update(model_config)

    model_cls = get_model_cls(config["model"])
    return model_cls(input_shape, num_outputs, config)


def get_activation_fn(name):
    if "relu" == name:
        return nn.ReLU
    elif "tanh" == name:
        return nn.Tanh
    elif "sigmoid" == name:
        return nn.Sigmoid
    elif "selu" == name:
        return nn.SELU
    elif "elu" == name:
        return nn.ELU
    elif "leaky_relu" == name:
        return nn.LeakyReLU
    else:
        raise ValueError(f"Activation function '{name}' is not defined")


def get_model_cls(name):
    if "dense" == name:
        return DenseNet
    elif "lstm" == name:
        return LSTMNet
    elif "gru" == name:
        return GRUNet
    else:
        raise ValueError(f"Model class '{name}' is not defined")


def build_conv_layers(input_shape, config):
    """
    Builds convolutional layers if needed for the given input shape
    """
    if isinstance(input_shape, int):
        return None, [input_shape]
    elif len(input_shape) == 1:
        return None, input_shape[0]
    elif len(input_shape) == 3:
        activation = get_activation_fn(config["conv_activation"])
        filters = config["conv_filters"]
    else:
        raise ValueError(f"Unsupported input shape {input_shape}")

    input_channels = input_shape[0]
    image_shape = list(input_shape)[1:]

    layers = []
    for l, (channels, kernel, stride) in enumerate(filters):
        kernel = list(kernel)
        padding = []
        for d in range(len(kernel)):
            kernel[d] += (kernel[d] + 1) % 2  # Make sue the kernel dims are odd
            padding.append(kernel[d] // 2)  # Pad to maintain the same size - has to be done manually for stride > 1

        layers.append(nn.Conv2d(
            input_channels,
            channels,
            kernel,
            stride=stride,
            padding=padding
        ))
        input_channels = channels
        
        if isinstance(stride, int):  # Expand stride to a list if it is not already
            stride = [stride] * len(image_shape)

        for d, s in enumerate(stride):
            image_shape[d] = -(-image_shape[d] // s) 

        if l < len(filters) - 1:
            layers.append(activation())
    
    network = nn.Sequential(*layers)
    output_shape = [input_channels] + image_shape
    
    return network, output_shape


class DenseNet(nn.Module):
    
    def __init__(self, input_shape, num_outputs, config):
        super(DenseNet, self).__init__()

        # Build convolutional layers if needed
        self._conv, input_shape = build_conv_layers(input_shape, config)
        input_size = int(np.prod(input_shape))

        # Build hidden layers
        activation = get_activation_fn(config["activation"])
        hidden_layers = config["hidden_layers"]
        hidden_size = config["hidden_size"]

        layers = []
        for l in range(hidden_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(activation())
            input_size = hidden_size

        # Add final linear layer
        layers.append(nn.Linear(input_size, num_outputs))

        self._features = nn.Sequential(*layers)

    def forward(self, 
            input: torch.Tensor, 
            state: Tuple[torch.Tensor, torch.Tensor]) \
                -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        if self._conv is not None:
            base_shape = list(input.shape)
            input = input.reshape([base_shape[0] * base_shape[1]] + base_shape[2:])
            input = self._conv(input)
            input = input.reshape((base_shape[0], base_shape[1], -1))

        return self._features(input), state

    @torch.jit.export
    def initial_state(self, batch_size: int=1, device: str="cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.zeros((1,), dtype=torch.float32, device=device)
        cell = torch.zeros((1,), dtype=torch.float32, device=device)
        
        return hidden, cell


class LSTMNet(nn.Module):
    
    def __init__(self, input_shape, num_outputs, config={}):
        super(LSTMNet, self).__init__()

        # Build convolutional layers if needed
        self._conv, input_shape = build_conv_layers(input_shape, config)
        
        if isinstance(input_shape, int):
            input_size = int(input_shape)  # NOTE: Need to explicitly cast for TorchScript to work
        else:
            input_size = int(np.prod(input_shape))
        
        # Build LSTM
        self._hidden_size = config["hidden_size"]
        self._hidden_layers = config["hidden_layers"]

        self._lstm = nn.LSTM(input_size, self._hidden_size, self._hidden_layers)

        # Add final linear layer
        self._output = nn.Linear(self._hidden_size, num_outputs)

    def forward(self, 
            input: torch.Tensor, 
            state: Tuple[torch.Tensor, torch.Tensor]) \
                -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        if self._conv is not None:
            base_shape = list(input.shape)
            input = input.reshape([base_shape[0] * base_shape[1]] + base_shape[2:])
            input = self._conv(input)
            input = input.reshape((base_shape[0], base_shape[1], -1))

        features, state = self._lstm(input, state)
        return self._output(features), state

    @torch.jit.export
    def initial_state(self, batch_size: int=1, device: str="cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        shape = [self._hidden_layers, batch_size, self._hidden_size]  # NOTE: Shape must be a list for TorchScript serialization to work
        hidden = torch.zeros(shape, dtype=torch.float32, device=device)
        cell = torch.zeros(shape, dtype=torch.float32, device=device)

        return hidden, cell


class GRUNet(nn.Module):
    
    def __init__(self, input_shape, num_outputs, config={}):
        super(GRUNet, self).__init__()
        
        # Build convolutional layers if needed
        self._conv, input_shape = build_conv_layers(input_shape, config)
        
        if isinstance(input_shape, int):
            input_size = int(input_shape)  # NOTE: Need to explicitly cast for TorchScript to work
        else:
            input_size = int(np.prod(input_shape))

        # Build GRU
        self._hidden_size = config["hidden_size"]
        self._hidden_layers = config["hidden_layers"]

        self._gru = nn.GRU(input_size, self._hidden_size, self._hidden_layers)

        # Add final linear layer
        self._output = nn.Linear(self._hidden_size, num_outputs)

    def forward(self, 
            input: torch.Tensor, 
            state: Tuple[torch.Tensor, torch.Tensor]) \
                -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        if self._conv is not None:
            base_shape = list(input.shape)
            input = input.reshape([base_shape[0] * base_shape[1]] + base_shape[2:])
            input = self._conv(input)
            input = input.reshape((base_shape[0], base_shape[1], -1))
        
        hidden, cell = state
        features, hidden = self._gru(input, hidden)
        return self._output(features), (hidden, cell)

    @torch.jit.export
    def initial_state(self, batch_size: int=1, device: str="cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        shape = [self._hidden_layers, batch_size, self._hidden_size]  # NOTE: Shape must be a list for TorchScript serialization to work
        hidden = torch.zeros(shape, dtype=torch.float32, device=device)
        cell = torch.zeros((1,), dtype=torch.float32, device=device)
        
        return hidden, cell
