import json
import gym
from gym.wrappers import RecordVideo, TimeLimit
import time

def main():

    NUM_EPISODES = 5
    MAX_STEPS = 500
    TIMESTEP = 0.02

    env = gym.make("MountainCar-v0")
    env = TimeLimit(env, max_episode_steps=MAX_STEPS)
    env = RecordVideo(env, "../results/debug/video",episode_trigger=lambda id: True)

    for episode in range(NUM_EPISODES):
        print(f"episode {episode}")
        obs = env.reset()
        done = False
        steps = 0

        env.render()
        time.sleep(TIMESTEP)

        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            steps += 1

            env.render()
            time.sleep(TIMESTEP)


if __name__ == "__main__":
    main()  # No clue why, but wrapping this in a function fixes the meta-data write issue
