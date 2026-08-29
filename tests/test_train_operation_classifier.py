from src.train_operation_classifier import train_operation_classifier


def test_training_produces_a_checkpoint_file(tmp_path) -> None:
    output_path = str(tmp_path / "op_classifier.pth")
    train_operation_classifier(epochs=2, batch_size=16, output_path=output_path)
    assert (tmp_path / "op_classifier.pth").exists()
