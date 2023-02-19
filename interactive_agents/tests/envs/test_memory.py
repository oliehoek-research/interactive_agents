from gymnasium.spaces import Discrete, Box
import numpy as np

from interactive_agents.envs import get_env_class

def test_memory():
    LENGTHS = [1, 2, 10]
    NUM_CUES = [2, 5]

    env_cls = get_env_class("memory")
    rng = np.random.default_rng(seed=0)

    def get_cue(obs, num_cues):
        cue = -1
        for idx in range(num_cues):
            if obs[idx] > 0.5:
                cue = idx
        
        return cue

    for length in LENGTHS:
        for num_cues in NUM_CUES:
            agent_id = f"agent_{length}_{num_cues}"
            env = env_cls({
                "length": length,
                "num_cues": num_cues,
                "agent_id": agent_id
            })

            assert agent_id in env.observation_spaces, "correct agent ID not given for obs space"
            assert agent_id in env.action_spaces, "correct agent ID not given for action space"
            assert isinstance(env.observation_spaces[agent_id], Box), "obs space is not Box"
            assert isinstance(env.action_spaces[agent_id], Discrete), "action space is not Discrete"
            assert env.observation_spaces[agent_id].shape == (num_cues + 2,), "incorrect obs size"
            assert env.action_spaces[agent_id].n == num_cues, "incorrect number of actions"

            # Run one episode where we take the correct action at the end
            obs = env.reset(seed=rng.integers(1000))
            cue = get_cue(obs[agent_id], num_cues)

            assert obs[agent_id][num_cues] > 0.5, "failed to indicate initial state"
            assert cue >= 0, "no cue provided"

            for step in range(length):
                obs, reward, terminated, truncated, _ = env.step({agent_id: 0})
                assert get_cue(obs[agent_id], num_cues) == -1, "new cue provided after first step"
                assert reward[agent_id] == 0, "non-zero reward at intermediate step"
                assert not terminated[agent_id], "episode terminated early"
                assert not truncated[agent_id], "episode truncated"

                if step < length -1:
                    assert all([value < 0.5 for value in obs[agent_id]]), "non-zero obs at intermediate step"
                else:
                    assert obs[agent_id][num_cues + 1] > 0.5, "failed to indicate final step"
                
            obs, reward, terminated, truncated, _ = env.step({agent_id: cue})
            assert reward[agent_id] > 0, "failed to give reward for correct cue"
            assert terminated[agent_id], "failed to terminate after last action"
            assert not truncated[agent_id], "episode truncated"

            # Run another episode where we take the incorrect action at the end
            obs = env.reset()
            cue = get_cue(obs[agent_id], num_cues)
            assert cue >= 0, "no cue provided"

            for step in range(length):
                obs, reward, terminated, _, _ = env.step({agent_id: 0})
                
            action = (cue + 1) % num_cues
            obs, reward, terminated, _, _ = env.step({agent_id: action})
            assert reward[agent_id] == 0, "gave reward for incorrect cue"
            assert terminated[agent_id], "failed to terminate after last action"
