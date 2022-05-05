"""
Test of integrating torch Conv2d and LSTM modules
"""
from numbers import Number
import numpy as np
import torch
import torch.nn as nn

if __name__ == "__main__":
    MAX_LENGTH = 100
    MIN_LENGTH = 10
    NUM_SAMPLES = 45
    CHANNELS = 3
    WIDTH = 128
    HEIGHT = 128

    HIDDEN_SIZE = 32
    HIDDEN_LAYERS = 2

    CONV_FILTERS =  [
        (8, (5, 5), 2), 
        (8, (5, 5), 2), 
        (8, (5, 5), 2)
    ]

    # Generate random sequence data
    samples = []

    for _ in range(NUM_SAMPLES):
        seq_len = np.random.randint(MIN_LENGTH, MAX_LENGTH)
        seq = np.random.uniform(0.0, 0.1, (seq_len, CHANNELS, WIDTH, HEIGHT))
        samples.append(seq)

    # Construct model
    input_channels = CHANNELS
    image_shape = [WIDTH, HEIGHT]

    layers = []
    for l, (channels, kernel, stride) in enumerate(CONV_FILTERS):
        kernel = list(kernel)
        padding = []
        for d in range(len(kernel)):
            kernel[d] += (kernel[d] + 1) % 2
            padding.append(kernel[d] // 2)

        layers.append(nn.Conv2d(
            input_channels,
            channels,
            kernel,
            stride=stride,
            padding=padding
        ))
        input_channels = channels
        
        if isinstance(stride, Number):
            stride = [stride] * len(image_shape)

        for d, s in enumerate(stride):
            image_shape[d] = -(-image_shape[d] // s)
        
        print(f"Conv2D: predicted output shape: {image_shape}")

        if l < len(CONV_FILTERS) - 1:
            layers.append(nn.ReLU())
    
    conv = nn.Sequential(*layers)
    num_features = int(input_channels * np.prod(image_shape))

    lstm = nn.LSTM(num_features, HIDDEN_SIZE, HIDDEN_LAYERS)

    # Process sequences
    seq_lens = [sample.shape[0] for sample in samples]
    samples = [torch.as_tensor(sample, dtype=torch.float32) for sample in samples]
    
    samples = nn.utils.rnn.pad_sequence(samples)
    print(f"\nPadded sequence shape: {samples.shape}")

    samples = nn.utils.rnn.pack_padded_sequence(samples, seq_lens, enforce_sorted=False)
    print(f"\nPacked sequence type: {samples.data.shape}")
    print(f"packed type: {type(samples)}")

    data = conv(samples.data)
    print(f"\nConv output shape: {data.shape}")
    data = torch.flatten(data, start_dim=1)
    samples = nn.utils.rnn.PackedSequence(data, samples.batch_sizes, samples.sorted_indices, samples.unsorted_indices)
    print(f"flattened shape: {samples.data.shape}")

    hidden_shape = (HIDDEN_LAYERS, NUM_SAMPLES, HIDDEN_SIZE)
    state = (torch.zeros(hidden_shape, dtype=torch.float32), torch.zeros(hidden_shape, dtype=torch.float32))
    output, (hidden, cell) = lstm(samples, state)
    print(f"\nLSTM output shape: {output.data.shape}")
    print(f"hidden shape: {hidden.shape}")
    print(f"cell shape: {cell.shape}")

    output, _ = nn.utils.rnn.pad_packed_sequence(output)
    print(f"\nFinal output shape: {output.shape}")
