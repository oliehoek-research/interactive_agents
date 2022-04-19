#!/usr/bin/env python3

'''
Generic training script using Stable Baselines 3.
'''

# TODO: Launch SB3 training through API
# TODO: Test TorchScript Export
# TODO: Examine running multiple experiments concurrently - potentially using Ray

import argparse
import gym
import yaml

from stable_baselines3 import DQN


def parse_args():
    parser = argparse.ArgumentParser("Generic training script for any registered single agent environment.")

    parser.add_argument("-f", "--config-file", default=None, type=str, action="append",
                        help="If specified, use config options from these files.")
    parser.add_argument("--local-dir", type=str, default="../../results/debug",
                        help="path to save training results")
    parser.add_argument("--num-cpus", type=int, default=4,
                        help="the maximum number of CPUs ray is allowed to us, useful for running locally")
    parser.add_argument("--num-gpus", type=float, default=0,
                        help="the number of GPUs to allocate for each trainer")
    parser.add_argument("--total-gpus", type=int, default=0,
                        help="the maximum number of GPUs ray is allowed to use")
    parser.add_argument("--num-workers", type=int,
                        help="the number of parallel workers per experiment")
    
    return parser.parse_args()


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')

    model = DQN('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=int(2e5))

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()
