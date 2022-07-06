import argparse
import math
import os
import wandb

class WandBSession:

    def __init__(self, project, entity, key, config={}):
        self._project = project
        self._entity = entity

        os.environ["WANDB_API_KEY"] = key

        wandb.init(project=project, entity=entity)
        wandb.config = config

    def log(self, scalars):
        wandb.log(scalars)


def parse_args():
    parser = argparse.ArgumentParser("Test script for Weights and Biases Logging")

    parser.add_argument("-k", "--api-key", default=None, type=str,
                        help="API Key for WandB project")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    wandb_session = WandBSession("alpha_test", 
                                 "ad_hoc_cooperation", 
                                 args.api_key, {
                                    "hyperparameter_1": 1.0,
                                    "hyperparameter_2": 2.0
                                 })

    for idx in range(1000):
        wandb_session.log({
            "x": idx,
            "sin": math.sin(idx),
            "cos": math.cos(idx)
        })
    
    print(f"test complete")
