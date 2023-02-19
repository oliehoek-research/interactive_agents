from interactive_agents.envs import get_env_class

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
