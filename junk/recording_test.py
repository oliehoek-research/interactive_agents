import json
import gym
from gym.wrappers import RecordVideo

def main():
    env = gym.make("MountainCar-v0")
    env = RecordVideo(env, "../results/debug/video")

    NUM_EPISODES = 1
    MAX_STEPS = 5

    for episode in range(NUM_EPISODES):
        obs = env.reset()
        done = False
        steps = 0

        while not done and steps < MAX_STEPS:
            action = env.action_space.sample()
            print(f"\nstep {steps}")
            obs, reward, done, info = env.step(action)
            steps += 1


if __name__ == "__main__":
    main()  # No clue why, but wrapping this in a function fixes the meta-data write issue
