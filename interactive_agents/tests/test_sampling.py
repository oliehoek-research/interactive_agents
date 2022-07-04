from gym.spaces import Discrete, Box
import numpy as np

from interactive_agents.envs import MultiagentEnv
from interactive_agents.sampling import Batch, sample

class DummyEnv(MultiagentEnv):

    def __init__(self, obs_spaces, action_spaces, rewards, num_steps):
        self.observation_spaces = obs_spaces
        self.action_spaces = action_spaces

        self._rewards = rewards

        self._max_steps = num_steps
        self._current_step = 0
    
    def _obs(self):
        obs = {}
        for pid, obs_space in self.observation_spaces.items():
            obs[pid] = obs_space.sample()

        return obs

    def reset(self):
        self._current_step = 0
        return self._obs()
    
    def step(self, actions):
        for pid, space in self.action_spaces.items():
            assert pid in actions, f"no action specified for agent '{pid}'"
            assert space.contains(actions[pid]), f"invalid action for agent '{pid}'"
        
        self._current_step += 1

        done = self._current_step >= self._max_steps
        dones = {pid:done for pid in self.observation_spaces.keys()}

        return self._obs(), self._rewards, dones, None


class DummyAgent:

    def __init__(self, policy):
        self._policy = policy

    def act(self, obs):
        return self._policy.act(obs)


class DummyPolicy:

    def __init__(self, obs_space, action_space, fetches):
        self.obs_space = obs_space
        self.action_space = action_space
        self.fetches = fetches

    def make_agent(self):
        return DummyAgent(self)
    
    def act(self, obs):
        assert self.obs_space.contains(obs), "agent received invalid observation"
        return self.action_space.sample(), self.fetches


def validate_batch(batch,
                   obs_space, 
                   action_space, 
                   fetches, 
                   num_episodes, 
                   num_steps):
    assert len(batch) == num_episodes, "unexpected number of episodes in batch"
    
    for episode in batch:
        assert len(episode[Batch.OBS]) == num_steps, "unexpected number of obs in batch"
        assert len(episode[Batch.ACTION]) == num_steps, "unexpected number of actions in batch"
        assert len(episode[Batch.REWARD]) == num_steps, "unexpected number of rewards in batch"
        assert len(episode[Batch.DONE]) == num_steps, "unexpected number of dones in batch"
        assert len(episode[Batch.NEXT_OBS]) == num_steps, "unexpected number of next_obs in batch"

        assert all([obs_space.contains(obs) for obs in episode[Batch.OBS]]), "invalid obs in batch"
        assert all([obs_space.contains(obs) for obs in episode[Batch.NEXT_OBS]]), "invalid next_obs in batch"
        assert all([action_space.contains(action) for action in episode[Batch.ACTION]]), "invalid action in batch"

        for key, value in fetches.items():
            assert key in episode, f"fetch item '{key}' not found in batch"
            assert all([(value == fetch).all() for fetch in episode[key]]), f"invalid value for '{key}' found in batch"


def test_sampling():
    NUM_EPISODES = 30
    NUM_STEPS = 10

    agent_obs_space = Box(0, 1, (2,))
    agent_action_space = Discrete(2)

    adversary_obs_space = Box(0, 1, (3,))
    adversary_action_space = Discrete(3)

    obs_spaces = {
        "agent": agent_obs_space,
        "adversary": adversary_obs_space
    }

    action_spaces = {
        "agent": agent_action_space,
        "adversary": adversary_action_space
    }

    rewards = {
        "agent": 1,
        "adversary": -1
    }

    env = DummyEnv(obs_spaces, action_spaces, rewards, NUM_STEPS)

    alice_fetches = {"alice_fetch": np.zeros((5,))}
    bob_fetches = {"bob_fetch": np.zeros((6,))}
    eve_fetches = {"eve_fetch": np.zeros((7,))}

    policies = {
        "alice": DummyPolicy(agent_obs_space, agent_action_space, alice_fetches),
        "bob": DummyPolicy(agent_obs_space, agent_action_space, bob_fetches),
        "eve": DummyPolicy(adversary_obs_space, adversary_action_space, eve_fetches),
    }
    
    alice_batch = sample(env, policies, NUM_EPISODES, 
        policy_fn=lambda id: "alice" if "agent" == id else "eve")

    bob_batch = sample(env, policies, NUM_EPISODES, 
        policy_fn=lambda id: "bob" if "agent" == id else "eve")
    
    batch = Batch()
    batch.extend(alice_batch)
    batch.extend(bob_batch)

    assert "alice" in batch, "no data for alice in batch"
    assert "bob" in batch, "no data for bob in batch"
    assert "eve" in batch, "no data for eve in batch"

    validate_batch(batch["alice"], agent_obs_space, 
        agent_action_space, alice_fetches, NUM_EPISODES, NUM_STEPS)
    
    validate_batch(batch["bob"], agent_obs_space, 
        agent_action_space, bob_fetches, NUM_EPISODES, NUM_STEPS)
    
    validate_batch(batch["eve"], adversary_obs_space, 
        adversary_action_space, eve_fetches, NUM_EPISODES * 2, NUM_STEPS)

    stats = batch.statistics({
        "alice": "alice_t",
        "bob": "bob_t",
        "eve": "eve_t"
    })

    assert stats["alice_t/reward_mean"] == NUM_STEPS, "incorrect reward mean"
    assert stats["alice_t/reward_max"] == NUM_STEPS, "incorrect reward max"
    assert stats["alice_t/reward_min"] == NUM_STEPS, "incorrect reward min"

    assert stats["bob_t/reward_mean"] == NUM_STEPS, "incorrect reward mean"
    assert stats["bob_t/reward_max"] == NUM_STEPS, "incorrect reward max"
    assert stats["bob_t/reward_min"] == NUM_STEPS, "incorrect reward min"

    assert stats["eve_t/reward_mean"] == -NUM_STEPS, "incorrect reward mean"
    assert stats["eve_t/reward_max"] == -NUM_STEPS, "incorrect reward max"
    assert stats["eve_t/reward_min"] == -NUM_STEPS, "incorrect reward min"

    assert stats["reward_mean"] == NUM_STEPS, "incorrect global reward mean"
    assert stats["reward_max"] == NUM_STEPS, "incorrect global reward max"
    assert stats["reward_min"] == -NUM_STEPS, "incorrect global reward min"
