"""
Script to compare different approaches to selecting Q-values according to integer action indices.

Compares two approaches:
1. Create a new one-hot encoding, apply the element-wise product, and reduce
2. Use torch.gather()
"""

import time
from numpy import dtype
import torch
import torch.nn as nn

class Stopwatch:

    def __init__(self):
        self._started = None
        self._elapsed = 0

    def start(self):
        if self._started is None:
            self._started = time.time()

    def stop(self):
        stopped = time.time()
        if self._started is not None:
            self._elapsed += stopped - self._started
            self._started = None

    def elapsed(self):
        return self._elapsed


if __name__ == "__main__":

    # Parameters
    DEVICE = 'cpu'
    NUM_SEEDS = 10
    NUM_ITERATIONS = 1000
    NUM_ACTIONS = 4
    BATCH_SIZE = 1024

    # Print config
    print("\nRunning Q-value benchmark")
    print(f"device: {DEVICE}")
    print(f"num seeds: {NUM_SEEDS}")
    print(f"num iterations: {NUM_ITERATIONS}")
    print(f"num actions: {NUM_ACTIONS}")
    print(f"batch size: {BATCH_SIZE}")

    # Timers
    gather_timer = Stopwatch()
    mask_timer = Stopwatch()

    # Accumulator - to ensure values are computed
    acc = torch.zeros((BATCH_SIZE,), dtype=torch.float32)

    # Iterate over seeds
    for seed in range(NUM_SEEDS):
        generator = torch.manual_seed(seed)
        q_values = torch.rand(BATCH_SIZE, NUM_ACTIONS, 
                            dtype=torch.float32, 
                            device="cpu", 
                            generator=generator)
        actions = torch.randint(0, NUM_ACTIONS, (BATCH_SIZE,), 
                            dtype=torch.int64, 
                            device="cpu", 
                            generator=generator)
        
        q_values.to(DEVICE)
        actions.to(DEVICE)

        mask = nn.functional.one_hot(actions, NUM_ACTIONS)
        
        # Test gather
        gather_timer.start()
        for _ in range(NUM_ITERATIONS):
            action_values = torch.gather(q_values, -1, actions.unsqueeze(-1)).squeeze(-1)
            acc += action_values
        gather_timer.stop()

        # Test mask
        mask_timer.start()
        for _ in range(NUM_ITERATIONS):
            # mask = nn.functional.one_hot(actions, NUM_ACTIONS)
            action_values = (mask * q_values).sum(-1)
            acc += action_values
        mask_timer.stop()

    # Print results
    print(f"\naccumulated values:\n{acc}")

    print(f"\ngather - total time: {gather_timer.elapsed()}s")
    print(f"mask    - total time: {mask_timer.elapsed()}s")
