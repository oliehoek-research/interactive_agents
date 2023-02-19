import numpy as np

from interactive_agents.envs import get_env_class

def common_actions(action, num_players):
    actions = {}
    for pid in range(num_players):
        actions[f"agent_{pid}"] = action
    
    return actions
    

def random_actions(num_actions, num_players, rng):
    actions = {}
    for pid in range(num_players):
        actions[f"agent_{pid}"] = rng.integers(num_actions)
    
    if actions["agent_0"] == actions["agent_1"]:
        actions["agent_0"] = (actions["agent_0"] + 1) % num_actions
    
    return actions


def check_encoding(obs, actions, num_players, num_actions):
    pids = [f"agent_{pid}" for pid in range(num_players)]
    for pid in pids:
        index = 0
        for other_pid in pids:
            if pid != other_pid:
                encoding = obs[pid][index:index + num_actions]
                if encoding.argmax() != actions[other_pid]:
                    return False
                
                index += num_actions
    
    return True


def test_coordination():
    NUM_ACTIONS = [5, 10]
    NUM_PLAYERS = [2, 5]

    env_cls = get_env_class("coordination")
    rng = np.random.default_rng(seed=0)

    for num_actions in NUM_ACTIONS:
        for num_players in NUM_PLAYERS:

            # Test standard game
            num_stages = num_actions * 2
            env = env_cls({
                "stages": num_stages,
                "actions": num_actions,
                "players": num_players
            }) 

            obs = env.reset(seed=rng.integers(1000))
            for ob in obs.values():
                assert ob.shape == (num_actions * (num_players - 1),), "incorrect observation shape"
                assert np.allclose(ob, np.zeros(ob.shape)), "non-zero observation in first stage"
            
            actions = [common_actions(action, num_players) for action in range(num_actions)]
            actions += [random_actions(num_actions, num_players, rng) for _ in range(num_actions)]

            for stage, action in enumerate(actions):
                obs, rewards, terminated, truncated, _ = env.step(action)
                assert check_encoding(obs, action, num_players, num_actions), "actions not properly encoded"

                if stage < num_actions:
                    assert all([1 == r for r in rewards.values()]), "identical actions did not receive a payoff of 1"
                else:
                    assert all([0 == r for r in rewards.values()]), f"distinct actions did not receive a payoff of 0"

                
                if stage < num_stages - 1:
                    assert not any([t for t in terminated.values()]), "episode terminated early"
                else:
                    assert all([t for t in terminated.values()]), "episode failed to terminate"

                assert not any([t for t in truncated.values()]), "episode truncated"


def test_other_play():
    NUM_ACTIONS = [5, 10]
    NUM_PLAYERS = [2, 5]

    env_cls = get_env_class("coordination")
    rng = np.random.default_rng(seed=0)

    for num_actions in NUM_ACTIONS:
        for num_players in NUM_PLAYERS:
            env = env_cls({
                "stages": 2,
                "actions": num_actions,
                "players": num_players,
                "other_play": True,
            })
            env.reset(seed=rng.integers(1000))

            actions = random_actions(num_actions, num_players, rng)
            obs, _, _, _, _ = env.step(actions)

            new_actions = {"agent_0": actions["agent_0"]}
            for pid, ob in obs.items():
                if "agent_0" != pid:
                    new_actions[pid] = ob[:num_actions].argmax()

            _, rewards, _, _, _ = env.step(new_actions)
            assert all([1 == r for r in rewards.values()]), "coordination did not yield payoff of 1"


def test_focal_point():
    NUM_ACTIONS = [5, 10]
    NUM_PLAYERS = [2, 5]

    env_cls = get_env_class("coordination")
    rng = np.random.default_rng(seed=0)

    for num_actions in NUM_ACTIONS:
        for num_players in NUM_PLAYERS:
            env = env_cls({
                "stages": 1,
                "actions": num_actions,
                "players": num_players,
                "focal_point": True,
                "focal_payoff": .8,
                "other_play": True,
            })
            env.reset(seed=rng.integers(1000))

            actions = common_actions(0, num_players)
            _, rewards, _, _, _ = env.step(actions)

            assert all([.8 == r for r in rewards.values()]), f"focal point did not receive correct payoff"


def test_meta_learning():
    NUM_ACTIONS = [5, 10]
    NUM_PLAYERS = [2, 5]

    env_cls = get_env_class("coordination")
    rng = np.random.default_rng(seed=0)

    for num_actions in NUM_ACTIONS:
        for num_players in NUM_PLAYERS:
            env = env_cls({
                "stages": 2,
                "actions": num_actions,
                "players": num_players,
                "meta_learning": True,
            })
            env.reset(seed=rng.integers(1000))

            assert len(env.observation_spaces.keys()) == 1, "Too many agents in the observation space"
            assert len(env.action_spaces.keys()) == 1, "Too many agents in the action space"

            first_obs, _, _, _, _ = env.step({"agent_0": 0})
            first_obs = first_obs["agent_0"]

            first_actions = []
            for index in range(0, len(first_obs), num_actions):
                first_actions.append(first_obs[index:index + num_actions].argmax())

            assert all([first_actions[0] == a for a in first_actions[1:]]), "fixed agents did not play the same action"

            second_obs, rewards, _, _, _ = env.step({"agent_0": first_actions[0]})
            second_obs = second_obs["agent_0"]

            second_actions = []
            for index in range(0, len(second_obs), num_actions):
                second_actions.append(second_obs[index:index + num_actions].argmax())

            assert rewards["agent_0"] == 1, "coordination did not yield payoff of 1"
            assert np.allclose(first_actions, second_actions), "fixed agents did not play the same actions in the second round"
