import torch

from models.operation_classifier_net import (
    NUM_CLASSES,
    INPUT_DIM,
    OperationClassifierNet,
    load_model,
    save_model,
)


def test_forward_pass_shape() -> None:
    model = OperationClassifierNet()
    x = torch.rand(4, INPUT_DIM)
    out = model(x)
    assert out.shape == (4, NUM_CLASSES)


def test_predict_returns_class_indices() -> None:
    model = OperationClassifierNet()
    x = torch.rand(3, INPUT_DIM)
    preds = model.predict(x)
    assert preds.shape == (3,)
    assert all(0 <= int(p) < NUM_CLASSES for p in preds)


def test_save_and_load_roundtrip(tmp_path) -> None:
    model = OperationClassifierNet()
    model.eval()  # disable dropout so outputs are directly comparable
    path = str(tmp_path / "checkpoint.pth")
    save_model(model, path)

    loaded = OperationClassifierNet()
    loaded = load_model(loaded, path)

    x = torch.rand(2, INPUT_DIM)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x))
