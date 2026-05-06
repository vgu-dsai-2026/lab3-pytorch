from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

sys.path.append(str(Path(__file__).resolve().parents[1]))


# -----------------------------------------------------------------------------
# MODULE BOOTSTRAP
# -----------------------------------------------------------------------------

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebook.py"
SPEC = importlib.util.spec_from_file_location("student_notebook", NOTEBOOK_PATH)
NOTEBOOK_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(NOTEBOOK_MODULE)

SEED = NOTEBOOK_MODULE.SEED
LABELS = NOTEBOOK_MODULE.LABELS

build_label_mapping = NOTEBOOK_MODULE.build_label_mapping
image_to_tensor = NOTEBOOK_MODULE.image_to_tensor
CatsDogsDataset = NOTEBOOK_MODULE.CatsDogsDataset
build_dataloaders = NOTEBOOK_MODULE.build_dataloaders
inspect_first_batch = NOTEBOOK_MODULE.inspect_first_batch
CatsDogsSimpleCNN = NOTEBOOK_MODULE.CatsDogsSimpleCNN
setup_training = NOTEBOOK_MODULE.setup_training
train_one_epoch = NOTEBOOK_MODULE.train_one_epoch
evaluate = NOTEBOOK_MODULE.evaluate
run_training_experiment = NOTEBOOK_MODULE.run_training_experiment


# -----------------------------------------------------------------------------
# SHARED TEST HELPERS
# -----------------------------------------------------------------------------


def write_rgb_image(path: Path, color: tuple[int, int, int], size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    image[..., 0] = color[0]
    image[..., 1] = color[1]
    image[..., 2] = color[2]
    Image.fromarray(image, mode="RGB").save(path)


class IndexedDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, data_root: Path):
        self.frame = frame.reset_index(drop=True)
        self.data_root = data_root

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        image = torch.full((3, 4, 4), float(row["sample_id"]), dtype=torch.float32)
        label = torch.tensor(int(row["label_id"]), dtype=torch.long)
        return image, label


def make_tiny_loader(
    num_samples: int = 6,
    batch_size: int = 2,
    image_shape: tuple[int, int, int] = (3, 8, 8),
) -> DataLoader:
    generator = torch.Generator().manual_seed(SEED)
    images = torch.randn((num_samples, *image_shape), generator=generator)
    labels = torch.tensor([idx % 2 for idx in range(num_samples)], dtype=torch.long)
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=False)


def make_tiny_model(input_shape: tuple[int, int, int] = (3, 8, 8)) -> nn.Module:
    channels, height, width = input_shape
    return nn.Sequential(nn.Flatten(), nn.Linear(channels * height * width, 2))


@pytest.fixture()
def split_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "filepath": [
                "train/cat/cat_0.png",
                "train/dog/dog_0.png",
                "val/cat/cat_1.png",
                "test/dog/dog_1.png",
            ],
            "label": ["cat", "dog", "cat", "dog"],
            "split": ["train", "train", "val", "test"],
        }
    )


@pytest.fixture()
def image_dataset_root(tmp_path: Path) -> Path:
    write_rgb_image(tmp_path / "train" / "cat" / "cat_0.png", color=(255, 0, 0), size=(20, 10))
    write_rgb_image(tmp_path / "train" / "dog" / "dog_0.png", color=(0, 255, 0), size=(12, 18))
    write_rgb_image(tmp_path / "val" / "cat" / "cat_1.png", color=(0, 0, 255), size=(16, 16))
    write_rgb_image(tmp_path / "test" / "dog" / "dog_1.png", color=(128, 64, 32), size=(14, 11))
    return tmp_path


# -----------------------------------------------------------------------------
# Question 1: build_label_mapping
# -----------------------------------------------------------------------------


def test_build_label_mapping_creates_expected_mapping_and_label_ids(split_frame: pd.DataFrame) -> None:
    label_to_index, labeled, *_ = build_label_mapping(split_frame)

    assert label_to_index == {"cat": 0, "dog": 1}
    assert "label_id" in labeled.columns
    assert labeled["label_id"].tolist() == [0, 1, 0, 1]


def test_build_label_mapping_returns_expected_split_frames(split_frame: pd.DataFrame) -> None:
    _, labeled, train_df, val_df, test_df = build_label_mapping(split_frame)

    assert len(train_df) == 2
    assert len(val_df) == 1
    assert len(test_df) == 1
    assert labeled is not split_frame


# -----------------------------------------------------------------------------
# Question 2: image_to_tensor / CatsDogsDataset
# -----------------------------------------------------------------------------


def test_image_to_tensor_returns_normalized_channel_first_tensor(image_dataset_root: Path) -> None:
    tensor = image_to_tensor(image_dataset_root / "train" / "cat" / "cat_0.png")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == (3, 64, 64)
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


def test_catsdogs_dataset_returns_tensor_and_long_label(
    split_frame: pd.DataFrame, image_dataset_root: Path
) -> None:
    _, labeled, *_ = build_label_mapping(split_frame)
    dataset = CatsDogsDataset(labeled[labeled["split"] == "train"], image_dataset_root)

    image_tensor, label_tensor = dataset[0]

    assert len(dataset) == 2
    assert image_tensor.shape == (3, 64, 64)
    assert image_tensor.dtype == torch.float32
    assert label_tensor.dtype == torch.long
    assert int(label_tensor.item()) == 0


# -----------------------------------------------------------------------------
# Question 3: build_dataloaders
# -----------------------------------------------------------------------------


@pytest.fixture()
def indexed_split_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(
        {
            "filepath": [f"unused_{idx}.png" for idx in range(8)],
            "label": ["cat", "dog", "cat", "dog", "cat", "dog", "cat", "dog"],
            "split": ["train", "train", "train", "train", "val", "val", "test", "test"],
            "label_id": [0, 1, 0, 1, 0, 1, 0, 1],
            "sample_id": list(range(8)),
        }
    )
    return (
        frame[frame["split"] == "train"].copy(),
        frame[frame["split"] == "val"].copy(),
        frame[frame["split"] == "test"].copy(),
    )


def test_build_dataloaders_returns_three_loaders_with_expected_batch_size(
    indexed_split_frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    train_df, val_df, test_df = indexed_split_frames
    train_loader, val_loader, test_loader = build_dataloaders(
        train_df,
        val_df,
        test_df,
        tmp_path,
        batch_size=2,
        seed=SEED,
        dataset_cls=IndexedDataset,
    )

    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    assert train_loader.batch_size == 2
    assert len(train_loader.dataset) == 4
    assert len(val_loader.dataset) == 2
    assert len(test_loader.dataset) == 2


def test_build_dataloaders_training_loader_is_reproducible(
    indexed_split_frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    train_df, val_df, test_df = indexed_split_frames
    loaders_a = build_dataloaders(
        train_df, val_df, test_df, tmp_path, batch_size=2, seed=SEED, dataset_cls=IndexedDataset
    )
    loaders_b = build_dataloaders(
        train_df, val_df, test_df, tmp_path, batch_size=2, seed=SEED, dataset_cls=IndexedDataset
    )

    train_batch_a = next(iter(loaders_a[0]))
    train_batch_b = next(iter(loaders_b[0]))

    assert torch.equal(train_batch_a[0], train_batch_b[0])
    assert torch.equal(train_batch_a[1], train_batch_b[1])


# -----------------------------------------------------------------------------
# Question 4: inspect_first_batch
# -----------------------------------------------------------------------------


def test_inspect_first_batch_returns_first_batch_from_loader() -> None:
    loader = make_tiny_loader(num_samples=4, batch_size=2, image_shape=(3, 8, 8))
    expected_images, expected_labels = next(iter(loader))

    batch_images, batch_labels = inspect_first_batch(loader)

    assert torch.equal(batch_images, expected_images)
    assert torch.equal(batch_labels, expected_labels)


def test_inspect_first_batch_rejects_none_loader() -> None:
    with pytest.raises(ValueError):
        inspect_first_batch(None)  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# Question 5: CatsDogsSimpleCNN
# -----------------------------------------------------------------------------


def test_simple_cnn_outputs_two_logits_per_example() -> None:
    model = CatsDogsSimpleCNN()
    batch = torch.randn(4, 3, 64, 64)
    logits = model(batch)

    assert logits.shape == (4, 2)
    assert hasattr(model, "stage1")
    assert hasattr(model, "stage2")
    assert hasattr(model, "classifier")


def test_simple_cnn_supports_backward_pass() -> None:
    model = CatsDogsSimpleCNN()
    batch = torch.randn(2, 3, 64, 64)
    labels = torch.tensor([0, 1], dtype=torch.long)

    loss = nn.CrossEntropyLoss()(model(batch), labels)
    loss.backward()

    assert any(param.grad is not None for param in model.parameters())


# -----------------------------------------------------------------------------
# Question 6: setup_training
# -----------------------------------------------------------------------------


def test_setup_training_returns_expected_components() -> None:
    model = make_tiny_model()
    device, moved_model, criterion, optimizer = setup_training(
        model, device=torch.device("cpu"), learning_rate=5e-4
    )

    assert device.type == "cpu"
    assert moved_model is model
    assert isinstance(criterion, nn.CrossEntropyLoss)
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-4)


def test_setup_training_respects_explicit_device() -> None:
    model = make_tiny_model()
    explicit_device = torch.device("cpu")

    device, moved_model, _, _ = setup_training(model, device=explicit_device)

    assert device == explicit_device
    assert next(moved_model.parameters()).device.type == explicit_device.type


# -----------------------------------------------------------------------------
# Question 7: train_one_epoch
# -----------------------------------------------------------------------------


def test_train_one_epoch_returns_finite_metrics() -> None:
    loader = make_tiny_loader()
    model = make_tiny_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    loss, accuracy = train_one_epoch(model, loader, optimizer, criterion, torch.device("cpu"))

    assert np.isfinite(loss)
    assert 0.0 <= accuracy <= 1.0


def test_train_one_epoch_updates_model_parameters() -> None:
    loader = make_tiny_loader()
    model = make_tiny_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = [param.detach().clone() for param in model.parameters()]

    train_one_epoch(model, loader, optimizer, criterion, torch.device("cpu"))

    after = list(model.parameters())
    assert any(not torch.allclose(old, new.detach()) for old, new in zip(before, after))


# -----------------------------------------------------------------------------
# Question 8: evaluate
# -----------------------------------------------------------------------------


def test_evaluate_returns_finite_metrics() -> None:
    loader = make_tiny_loader()
    model = make_tiny_model()
    criterion = nn.CrossEntropyLoss()

    loss, accuracy = evaluate(model, loader, criterion, torch.device("cpu"))

    assert np.isfinite(loss)
    assert 0.0 <= accuracy <= 1.0


def test_evaluate_does_not_update_model_parameters() -> None:
    loader = make_tiny_loader()
    model = make_tiny_model()
    criterion = nn.CrossEntropyLoss()
    before = [param.detach().clone() for param in model.parameters()]

    evaluate(model, loader, criterion, torch.device("cpu"))

    after = list(model.parameters())
    assert all(torch.allclose(old, new.detach()) for old, new in zip(before, after))


# -----------------------------------------------------------------------------
# Question 9: run_training_experiment
# -----------------------------------------------------------------------------


def test_run_training_experiment_returns_history_and_metrics_when_baseline_missing(tmp_path: Path) -> None:
    train_loader = make_tiny_loader()
    val_loader = make_tiny_loader()
    test_loader = make_tiny_loader()
    model = make_tiny_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    history, test_loss, test_acc = run_training_experiment(
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        torch.device("cpu"),
        epochs=1,
        plot=False,
    )

    assert len(history) == 1
    assert set(history[0]) == {"epoch", "train_loss", "train_acc", "val_loss", "val_acc"}
    assert np.isfinite(test_loss)
    assert 0.0 <= test_acc <= 1.0


def test_run_training_experiment_reads_baseline_csv(tmp_path: Path) -> None:
    train_loader = make_tiny_loader()
    val_loader = make_tiny_loader()
    test_loader = make_tiny_loader()
    model = make_tiny_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    history, _, _ = run_training_experiment(
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        torch.device("cpu"),
        epochs=1,
        plot=False,
    )

    assert len(history) == 1
