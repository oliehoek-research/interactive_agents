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
            state: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if self._conv is not None:
            base_shape = list(input.shape)
            input = input.reshape([base_shape[0] * base_shape[1]] + base_shape[2:])
            input = self._conv(input)
            input = input.reshape((base_shape[0], base_shape[1], -1))

        return self._features(input), state

    @torch.jit.export
    def initial_state(self, batch_size: int=1, device: str="cpu") -> None:
        return None


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
            state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if self._conv is not None:
            base_shape = list(input.shape)
            input = input.reshape([base_shape[0] * base_shape[1]] + base_shape[2:])
            input = self._conv(input)
            input = input.reshape((base_shape[0], base_shape[1], -1))
        
        hidden, cell = torch.split(state, self._hidden_size, dim=-1)
        features, (hidden, cell) = self._lstm(input, (hidden, cell))
        return self._output(features), torch.cat((hidden, cell), dim=-1)

    @torch.jit.export
    def initial_state(self, batch_size: int=1, device: str="cpu") -> torch.Tensor:
        shape = [self._hidden_layers, batch_size, self._hidden_size * 2]  # NOTE: Shape must be a list for TorchScript serialization to work
        state = torch.zeros(shape, dtype=torch.float32)
        
        return state.to(device)


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
            state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        if self._conv is not None:
            base_shape = list(input.shape)
            input = input.reshape([base_shape[0] * base_shape[1]] + base_shape[2:])
            input = self._conv(input)
            input = input.reshape((base_shape[0], base_shape[1], -1))
        
        features, state = self._gru(input, state)
        return self._output(features), state

    @torch.jit.export
    def initial_state(self, batch_size: int=1, device: str="cpu") -> torch.Tensor:
        shape = [self._hidden_layers, batch_size, self._hidden_size]  # NOTE: Shape must be a list for TorchScript serialization to work
        cell = torch.zeros(shape, dtype=torch.float32)
        
        return cell.to(device)


# UNIT TESTS
def serialization_cycle(model, data, output_shape, tmp_path):
    path = os.path.join(tmp_path, "model.pt")
    model = torch.jit.script(model)
    hidden = model.initial_state(batch_size=data.shape[1])

    target, _ = model(data, hidden)
    assert tuple(target.shape) == tuple(output_shape)

    torch.jit.save(model, path)
    model = torch.jit.load(path)
    hidden = model.initial_state(batch_size=data.shape[1])

    output, _ = model(data, hidden)
    assert torch.equal(target, output)


def serialization_vector(model_name, tmp_path):
    VECTOR_SIZE = 64
    HIDDEN_LAYERS = 2
    NUM_FEATURES = 64
    SEQ_LEN = 50
    BATCH_SIZE = 25

    vectors = torch.ones((SEQ_LEN, BATCH_SIZE, VECTOR_SIZE))
    output_shape = (SEQ_LEN, BATCH_SIZE, NUM_FEATURES)

    model = build_model([VECTOR_SIZE], NUM_FEATURES, {
        "model": model_name,
        "hidden_layers": HIDDEN_LAYERS,
        "hidden_size": NUM_FEATURES
    })
    serialization_cycle(model, vectors, output_shape, tmp_path)


def serialization_image(model_name, tmp_path):
    IMAGE_SHAPE = [3, 100, 100]
    CONV_FILTERS = [
            (8, (5, 5), (2, 2)), 
            (16, (4, 4), 2), 
            (16, (5, 5), 2)]
    HIDDEN_LAYERS = 2
    NUM_FEATURES = 64
    SEQ_LEN = 50
    BATCH_SIZE = 25

    images = torch.ones([SEQ_LEN, BATCH_SIZE] + IMAGE_SHAPE)
    output_shape = (SEQ_LEN, BATCH_SIZE, NUM_FEATURES)

    model = build_model(IMAGE_SHAPE, NUM_FEATURES, {
        "model": model_name,
        "hidden_layers": HIDDEN_LAYERS,
        "hidden_size": NUM_FEATURES,
        "conv_filters": CONV_FILTERS
    })
    serialization_cycle(model, images, output_shape, tmp_path)


def test_dense(tmp_path):
    serialization_vector("dense", tmp_path)


def test_dense_cnn(tmp_path):
    serialization_image("dense", tmp_path)


def test_lstm(tmp_path):
    serialization_vector("lstm", tmp_path)


def test_lstm_cnn(tmp_path):
    serialization_image("lstm", tmp_path)


def test_gru(tmp_path):
    serialization_vector("gru", tmp_path)


def test_gru_cnn(tmp_path):
    serialization_image("gru", tmp_path)
