import os.path
import torch

from interactive_agents.training.learners.models import build_model

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


def serialization_vector(model_name, tmp_path):
    VECTOR_SIZE = 64
    HIDDEN_LAYERS = 2
    NUM_FEATURES = 64
    SEQ_LEN = 50
    BATCH_SIZE = 25

    vectors = torch.ones((SEQ_LEN, BATCH_SIZE, VECTOR_SIZE))
    output_shape = (SEQ_LEN, BATCH_SIZE, NUM_FEATURES)

    model = build_model([VECTOR_SIZE], NUM_FEATURES, {
        "model": model_name,
        "hidden_layers": HIDDEN_LAYERS,
        "hidden_size": NUM_FEATURES
    })
    serialization_cycle(model, vectors, output_shape, tmp_path)


def serialization_image(model_name, tmp_path):
    IMAGE_SHAPE = [3, 100, 100]
    CONV_FILTERS = [
            (8, (5, 5), (2, 2)), 
            (16, (4, 4), 2), 
            (16, (5, 5), 2)]
    HIDDEN_LAYERS = 2
    NUM_FEATURES = 64
    SEQ_LEN = 50
    BATCH_SIZE = 25

    images = torch.ones([SEQ_LEN, BATCH_SIZE] + IMAGE_SHAPE)
    output_shape = (SEQ_LEN, BATCH_SIZE, NUM_FEATURES)

    model = build_model(IMAGE_SHAPE, NUM_FEATURES, {
        "model": model_name,
        "hidden_layers": HIDDEN_LAYERS,
        "hidden_size": NUM_FEATURES,
        "conv_filters": CONV_FILTERS
    })
    serialization_cycle(model, images, output_shape, tmp_path)


def test_dense(tmp_path):
    serialization_vector("dense", tmp_path)


def test_dense_cnn(tmp_path):
    serialization_image("dense", tmp_path)


def test_lstm(tmp_path):
    serialization_vector("lstm", tmp_path)


def test_lstm_cnn(tmp_path):
    serialization_image("lstm", tmp_path)


def test_gru(tmp_path):
    serialization_vector("gru", tmp_path)


def test_gru_cnn(tmp_path):
    serialization_image("gru", tmp_path)
