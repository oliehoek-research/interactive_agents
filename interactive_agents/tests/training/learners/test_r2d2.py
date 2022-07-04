import os.path
from gym.spaces import Box, Discrete
import torch

from interactive_agents.training.learners.priority_tree import PriorityTree
from interactive_agents.training.learners.r2d2 import QNetwork, QPolicy


def serialization_cycle(model, data, output_shape, tmp_path):
    path = os.path.join(tmp_path, "model.pt")
    model = torch.jit.script(model)
    hidden = model.initial_state(batch_size=data.shape[1])

    target, _ = model(data, hidden)
    assert tuple(target.shape) == tuple(output_shape)

    torch.jit.save(model, path)
    model = torch.jit.load(path)
    hidden = model.initial_state(batch_size=data.shape[1])

    output, _ = model(data, hidden)
    assert torch.equal(target, output)


def test_q_net(tmp_path):
    obs_space = Box(0,1,(10,))
    action_space = Discrete(5)

    data = torch.ones([20, 2] + list(obs_space.shape))
    q_shape = (20, 2, action_space.n)
    action_shape = (20, 2)

    q_network = QNetwork(obs_space, action_space, {
        "model": "dense",
        "hidden_layers": 1,
        "hidden_size": 32,
    }, dueling=True)
    serialization_cycle(q_network, data, q_shape, tmp_path)
    policy = QPolicy(q_network)
    serialization_cycle(policy, data, action_shape, tmp_path)


    q_network = QNetwork(obs_space, action_space, {
        "model": "lstm",
        "hidden_layers": 1,
        "hidden_size": 32,
    }, dueling=True)
    serialization_cycle(q_network, data, q_shape, tmp_path)
    policy = QPolicy(q_network)
    serialization_cycle(policy, data, action_shape, tmp_path)

    q_network = QNetwork(obs_space, action_space, {
        "model": "gru",
        "hidden_layers": 1,
        "hidden_size": 32,
    }, dueling=True)
    serialization_cycle(q_network, data, q_shape, tmp_path)
    policy = QPolicy(q_network)
    serialization_cycle(policy, data, action_shape, tmp_path)


def test_priority_tree():
    tree = PriorityTree(7)
    assert tree._capacity == 8

    tree.set([0], [1])
    assert tree.min() == 1
    assert tree.sum() == 1
    assert tree.prefix_index(0.5) == 0

    tree.set([0, 2, 3, 4], [2, 1, 4, 3])
    assert tree.min() == 1
    assert tree.sum() == 10
    assert tree.prefix_index(1) == 0
    assert tree.prefix_index(2) == 2
    assert tree.prefix_index(6) == 3
    assert tree.prefix_index(11) == 7

    tree.set([1, 5, 6, 7], [.5, 2, 1, 3])
    assert tree.min() == .5
    assert tree.sum() == 16.5
    assert tree.prefix_index(1) == 0
    assert tree.prefix_index(7) == 3
    assert tree.prefix_index(13) == 6
    assert tree.prefix_index(20) == 7
