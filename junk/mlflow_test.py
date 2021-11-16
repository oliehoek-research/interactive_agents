"""Test of multi-process training with mlflow"""
import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam


def generate_data(fn, num_features, feature_min=-1, feature_max=1, num_instances=1000):
    instances = []
    values = []

    for _ in range(num_instances):
        instance = np.random.random(num_features)
        instance = (feature_max - feature_min) * instance + feature_min
        instances.append(instance)
        values.append(fn(instance))

    instances = np.stack(instances)
    values = np.asarray(values)

    return instances, values


def train(seed, instances, values, num_iterations=1000, config={}):

    # Seed random number generators
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Build model
    layers = []
    last_size = instances[0].shape[0]
    for size in config.get("hidden", [32]):
        layers.append(nn.Linear(last_size, size))
        layers.append(nn.ReLU())
        last_size = size

    layers.append(nn.Linear(last_size, 1))
    model = nn.Sequential(*layers)

    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=config.get("lr", 0.001))

    # Train model
    batch_size = config.get("batch_size", 32)
    for _ in range(num_iterations):
        batch_indices = np.random.randint(0, len(instances), batch_size)
        instance_batch = torch.tensor(instances[batch_indices], dtype=torch.float32)
        value_batch = torch.tensor(values[batch_indices], dtype=torch.float32)

        optimizer.zero_grad()
        loss = (value_batch - model(instance_batch)) ** 2
        loss = loss.mean()

        mlflow.log_metric("error", loss.item())

        loss.backward()
        optimizer.step()

    # Evaluate model
    outputs = model(torch.tensor(instances, dtype=torch.float32))
    errors = (torch.tensor(values, dtype=torch.float32) - outputs) ** 2

    return errors.mean().sqrt().item()


if __name__ == "__main__":
    num_runs = 4
    config = {
        "hidden": [64, 64],
        "batch_size": 128,
        "lr": 0.01,
    }

    instances, values = generate_data(lambda x: np.sin(3 * x), 1)

    for seed in range(num_runs):
        with mlflow.start_run():
            mlflow.log_param("seed", seed)
            error = train(seed, instances, values, num_iterations=200, config=config)
            print(f"seed {seed}, error: {error}")
