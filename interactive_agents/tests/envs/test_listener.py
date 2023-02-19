from gymnasium.spaces import Discrete, Box
import numpy as np

from interactive_agents.envs import get_env_class

def get_cue(obs, num_cues):
    cue = -1
    for idx in range(num_cues):
        if obs[idx] > 0.5:
            cue = idx
    
    return cue


def get_previous_action(obs, num_cues):
    cue = -1
    for idx in range(num_cues):
        if obs[idx + num_cues] > 0.5:
            cue = idx
    
    return cue


def test_listener():
    NUM_CUES = [2, 5, 9]

    env_cls = get_env_class("listener")
    rng = np.random.default_rng(seed=0)

    for cues in NUM_CUES:
        stages = cues * 5
        agent_id = f"agent_{stages}_{cues}"
        env = env_cls({
            "stages": stages,
            "cues": cues,
            "agent_id": agent_id
        })

        assert agent_id in env.observation_spaces, "correct agent ID not given for obs space"
        assert agent_id in env.action_spaces, "correct agent ID not given for action space"
        assert isinstance(env.observation_spaces[agent_id], Box), "obs space is not Box"
        assert isinstance(env.action_spaces[agent_id], Discrete), "action space is not Discrete"
        assert env.observation_spaces[agent_id].shape == (cues * 2,), "incorrect obs size"
        assert env.action_spaces[agent_id].n == cues, "incorrect number of actions"

        obs = env.reset(seed=rng.integers(1000))
        terminated = False

        cue = get_cue(obs[agent_id], cues)
        prev_action = get_previous_action(obs[agent_id], cues)

        assert cue != -1, "no cue provided in the first obs"
        assert prev_action == -1, "previous action provided in the first obs"

        mapping = np.full(cues, -1, dtype=np.int64)

        for _ in range(stages):
            assert not terminated, "episode terminated early"

            action = 0 if mapping[cue] == -1 else mapping[cue]
            action = {agent_id: action}

            obs, reward, terminated, truncated, _ = env.step(action)
            obs = obs[agent_id]
            reward = reward[agent_id]
            terminated = terminated[agent_id]
            truncated = truncated[agent_id]

            assert not truncated, "episode was truncated"

            if mapping[cue] != -1:
                assert reward == 1, "corrrect action did not receive a reward of 1"

            prev_action = get_previous_action(obs, cues)
            assert prev_action != -1, "no previous action provided"

            mapping[cue] = prev_action
            cue = get_cue(obs, cues)

        assert terminated, "episode did not terminate on time"


def test_identity():
    NUM_CUES = [2, 5, 9]

    env_cls = get_env_class("listener")
    rng = np.random.default_rng(seed=0)

    for cues in NUM_CUES:
        stages = cues * 5
        agent_id = f"agent_{stages}_{cues}"
        env = env_cls({
            "stages": stages,
            "cues": cues,
            "agent_id": agent_id,
            "identity": True,
        })

        obs = env.reset(seed=rng.integers(1000))
        cue = get_cue(obs[agent_id], cues)

        for _ in range(stages):
            action = {agent_id: cue}
            obs, reward, _, _, _ = env.step(action)

            assert reward[agent_id] == 1, "corrrect action did not receive a reward of 1"
            cue = get_cue(obs[agent_id], cues)
