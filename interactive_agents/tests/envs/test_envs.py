from interactive_agents.envs import get_env_class, BatchedEnv

def test_mpe():
    env_cls = get_env_class("mpe")
    env = env_cls({"scenario": "simple_push_v2"})

    env.reset(seed=0)
    assert "agent_0" in env.agents, "'agent_0' not present in environment"
    assert "adversary_0" in env.agents, "'adversary_0' not present in environment"


def test_gym():
    env_cls = get_env_class("gym")
    env = env_cls({
        "name": "CartPole-v1",
        "agent_id": "agent_0"
    })

    env.reset(seed=0)
    assert "agent_0" in env.agents


def test_batched():
    NUM_STAGES = 10
    NUM_PLAYERS = 5
    NUM_ENVS = 10

    env_cls = get_env_class("coordination")
    env_config = {
        "stages": NUM_STAGES,
        "players": NUM_PLAYERS
    }

    agent_ids = [f"agent_{id}" for id in range(NUM_PLAYERS)]

    env = BatchedEnv(env_cls, env_config, NUM_ENVS)
    obs = env.reset(seed=0)

    for id in agent_ids:
        assert id in obs, "did not find all expected agents in initial obs"
        assert len(obs[id]) == NUM_ENVS, "incorrect number of entries in initial obs"
        assert all([(ob is not None) for ob in obs[id]]), "'None' found in initial obs"

    action = [0] * NUM_ENVS
    actions = { id:action for id in agent_ids}

    for stage in range(NUM_STAGES):
        obs, rewards, terminated, truncated, _ = env.step(actions)

        for id in agent_ids:
            for item in (obs, rewards, truncated, terminated):
                assert id in item, "did not find all expected agents"
                assert len(item[id]) == NUM_ENVS, "incorrect number of entries"
                assert all([(value is not None) for value in item[id]]), "'None' found in initial obs"
        
            if stage < NUM_STAGES - 1:
                assert not any(terminated[id]), "episode terminated early"
            else:
                print(f"stage: {stage}")
                assert all(terminated[id]), "episode failed to terminate properly"

            assert not any(truncated[id]), "episode truncated"
