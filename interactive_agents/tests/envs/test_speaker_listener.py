from gymnasium.spaces import Discrete, Box
import numpy as np

from interactive_agents.envs import get_env_class

def get_first_feature(obs, num_cues):
    cue = -1
    for idx in range(num_cues):
        if obs[idx] > 0.5:
            cue = idx
    
    return cue


def get_second_feature(obs, num_cues):
    cue = -1
    for idx in range(num_cues):
        if obs[idx + num_cues] > 0.5:
            cue = idx
    
    return cue


def test_speaker_listener():
    NUM_CUES = [2, 5, 9]

    env_cls = get_env_class("speaker_listener")
    rng = np.random.default_rng(seed=0)

    for cues in NUM_CUES:
        stages = cues + 1
        env = env_cls({
            "stages": stages,
            "cues": cues
        })

        assert isinstance(env.observation_spaces["speaker"], Box), "speaker obs space is not Box"
        assert isinstance(env.action_spaces["speaker"], Discrete), "speaker action space is not Discrete"
        assert isinstance(env.observation_spaces["listener"], Box), "listener obs space is not Box"
        assert isinstance(env.action_spaces["listener"], Discrete), "listener action space is not Discrete"

        assert env.observation_spaces["speaker"].shape == (cues * 2,), "incorrect speaker obs size"
        assert env.action_spaces["speaker"].n == cues, "incorrect number of speaker actions"
        assert env.observation_spaces["listener"].shape == (cues * 2,), "incorrect listener obs size"
        assert env.action_spaces["listener"].n == cues, "incorrect number of listener actions"

        obs = env.reset(seed=rng.integers(1000))

        current_cue = get_first_feature(obs["speaker"], cues)
        current_signal = get_first_feature(obs["listener"], cues)
        last_action = get_second_feature(obs["speaker"], cues)
        last_cue = get_second_feature(obs["listener"], cues)

        assert current_cue != -1, "no cue provided in the first obs"
        assert current_signal == -1, "signal provided in the first obs"
        assert last_action == -1, "previous action provided in the first obs"
        assert last_cue == -1, "previous cue provided in the first obs"

        prev_cue = -1
        terminated = {}
        truncated = {}

        for _ in range(stages):
            assert not any([t for t in terminated.values()]), "episode terminated early"
            assert not any([t for t in truncated.values()]), "episode truncated"

            actions = { 
                "speaker": current_cue,
                "listener": prev_cue if prev_cue != -1 else 0
            }

            obs, rewards, terminated, truncated, _ = env.step(actions)

            if last_cue != -1:
                assert 1 == rewards["speaker"] and 1 == rewards["listener"], "correct action did not receive reward of 1"

            last_action = get_second_feature(obs["speaker"], cues)
            last_cue = get_second_feature(obs["listener"], cues)

            assert last_action == actions["listener"], "listener action not properly encoded"
            assert last_cue == prev_cue, "previous cue not properly encoded"

            prev_cue = current_cue

            current_cue = get_first_feature(obs["speaker"], cues)
            current_signal = get_first_feature(obs["listener"], cues)

            assert current_cue != -1, "no cue provided"
            assert current_signal == actions["speaker"], "signal not properly encoded"
        
        assert all([t for t in terminated.values()]), "episode failed to terminate"
        assert not any([t for t in truncated.values()]), "episode truncated"


def test_meta_learning():
    NUM_CUES = [2, 5, 9]

    env_cls = get_env_class("speaker_listener")
    rng = np.random.default_rng(seed=0)

    for cues in NUM_CUES:
        stages = cues * 5
        env = env_cls({
            "stages": stages,
            "cues": cues,
            "other_play": True,
            "meta_learning": True, 
        })

        assert "listener" not in env.observation_spaces, "listener obs included for meta-learning"
        assert "listener" not in env.action_spaces, "listener actions included for meta-learning"

        mapping = np.full(cues, -1, dtype=np.int64)

        obs = env.reset(seed=rng.integers(1000))
        cue = get_first_feature(obs["speaker"], cues)
        prev_cue = -1

        for _ in range(stages):
            obs, _, _, _, _ = env.step({"speaker": cue})

            last_action = get_second_feature(obs["speaker"], cues)

            if -1 != prev_cue:
                if -1 != mapping[prev_cue]:
                    assert mapping[prev_cue] == last_action, "listener took a different actions for the same cue"
                else:
                    mapping[prev_cue] = last_action

            prev_cue = cue
            cue = get_first_feature(obs["speaker"], cues)
