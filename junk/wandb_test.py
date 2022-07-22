import argparse
import math
from multiprocessing import Pool
import os
from re import X
import time
import traceback
import wandb

def parse_args():
    parser = argparse.ArgumentParser("Test script for Weights and Biases Logging")

    parser.add_argument("-p", "--num-processes", default=20, type=int,
                        help="number of simultaneous experimental processes to launch (default: 20)")
    
    parser.add_argument("-k", "--api-key", default=None, type=str,
                        help="API Key for WandB project")
    parser.add_argument("--project", default="wandb_throttle_test", type=str,
                        help="the WandB project name to log results to")
    parser.add_argument("--entity", default="ad_hoc_cooperation", type=str,
                        help="the WandB team name (entity) to log results to")
    parser.add_argument("--group", default=None, type=str,
                        help="the WandB group name for results (optional)")
    
    parser.add_argument("-t", "--time", default=300, type=float,
                        help="the time limit for each dummy experiment in seconds (default: 300)")
    parser.add_argument("-r", "--rate", default=1, type=float,
                        help="the number of dummy training iterations to run per second (default: 1)")
    parser.add_argument("--scale", default=0.01, type=float,
                        help="increment along the x-axis per iteration (default: 0.01)")

    parser.add_argument("--sync-iters", default=1, type=float,
                        help="number of iterations between server syncs")

    return parser.parse_args()


class WandBSession:

    def __init__(self, project, group=None, entity=None, api_key=None):
        self._project = project
        self._group = group
        self._entity = entity
        self._api_key = api_key  # A little confusing, entities refer to teams, we can create multiple teams

    def run(self, name, config={}):
        if self._api_key is not None:
            os.environ["WANDB_API_KEY"] = self._api_key

        return wandb.init(entity=self._entity,
                          project=self._project,
                          group=self._group,
                          name=name,
                          config=config,
                          reinit=True)


def experiment(name, session, iter_per_s, time_limit, scale, sync_iters):
    start_time = time.time()
    sleep_interval = 1.0 / iter_per_s 
    config = {
        "iter_per_s": iter_per_s,
        "time_limit": time_limit,
        "scale": scale,
        "sync_iters": sync_iters,
    }

    with session.run(name, config):
        x = 0.0
        iteration = 0

        while time.time() - start_time <= time_limit:
            if (iteration + 1) % sync_iters == 0:
                commit = True
            else:
                commit = False

            wandb.log({
                "x": x,
                "sin": math.sin(x),
                "cos": math.cos(x)
            }, step=iteration, commit=commit)

            x += scale
            iteration += 1

            time.sleep(sleep_interval)


def print_error(error):
    traceback.print_exception(type(error), error, error.__traceback__, limit=5)


if __name__ == "__main__":
    args = parse_args()

    # Enable the WandB service to allow distributed logging
    wandb.require("service")

    # Initialize process pool
    pool = Pool(args.num_processes)

    # Configure WandB session
    session = WandBSession(project=args.project, 
        group=args.group, entity=args.entity, api_key=args.api_key)

    print(f"Launching {args.num_processes} processes")
    print(f"time limit: {args.time} seconds")

    runs = []
    for idx in range(args.num_processes):
        name = f"run_{idx}"
        runs.append(pool.apply_async(experiment, 
            (name, session, args.rate, args.time, args.scale, args.sync_iters), error_callback=print_error))
    
    for run in runs:
        run.wait()

    print("\nFinished!")
