"""Test of Ray-tune without RLLib"""
from ray import tune


def objective(step, alpha, beta):
    return (0.1 + alpha * step / 100)**(-1) + beta * 0.1


def train(config):
    alpha, beta = config["alpha"], config["beta"]
    for step in range(10):
        score = objective(step, alpha, beta)
        tune.report(mean_loss=score)


analysis = tune.run(
    train,
    config={
        "alpha": tune.grid_search([0.001, 0.01]),
        "beta": tune.choice([1, 2])
    })

print("Best config: ", analysis.get_best_config(metric="mean_loss", mode="min"))
