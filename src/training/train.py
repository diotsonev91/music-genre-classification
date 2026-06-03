from pathlib import Path

import joblib
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.early_stopping import create_early_stopping


def train_model(
    model,
    X_train,
    y_train
):
    model.fit(
        X_train,
        y_train
    )

    return model


def save_model(
    model,
    path
):
    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        path
    )

    return path


def load_model(
    path
):
    return joblib.load(
        path
    )


def create_tensor_loader(
    X,
    y,
    batch_size,
    shuffle=False
):
    dataset = TensorDataset(
        torch.tensor(
            X,
            dtype=torch.float32
        ),
        torch.tensor(
            y,
            dtype=torch.long
        )
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )


def train_torch_model(
    model,
    X_train,
    y_train,
    X_validation,
    y_validation,
    batch_size=32,
    epochs=30,
    learning_rate=0.001,
    patience=5,
    device=None
):
    if device is None:

        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(
        device
    )

    train_loader = create_tensor_loader(
        X_train,
        y_train,
        batch_size=batch_size,
        shuffle=True
    )

    validation_loader = create_tensor_loader(
        X_validation,
        y_validation,
        batch_size=batch_size,
        shuffle=False
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    early_stopping = create_early_stopping(
        patience=patience,
        mode="min",
        restore_best_weights=True
    )

    history = []

    for epoch in range(
        1,
        epochs + 1
    ):

        train_loss, train_accuracy = run_torch_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer
        )

        validation_loss, validation_accuracy = run_torch_epoch(
            model,
            validation_loader,
            criterion,
            device,
            optimizer=None
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_accuracy": train_accuracy,
            "validation_accuracy": validation_accuracy
        })

        should_stop = early_stopping.step(
            validation_loss,
            model=model
        )

        if should_stop:

            break

    model = early_stopping.restore(
        model
    )

    return (
        model,
        history
    )


def run_torch_epoch(
    model,
    data_loader,
    criterion,
    device,
    optimizer=None
):
    is_training = optimizer is not None

    if is_training:

        model.train()

    else:

        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for X_batch, y_batch in data_loader:

        X_batch = X_batch.to(
            device
        )

        y_batch = y_batch.to(
            device
        )

        if is_training:

            optimizer.zero_grad()

        with torch.set_grad_enabled(
            is_training
        ):

            logits = model(
                X_batch
            )

            loss = criterion(
                logits,
                y_batch
            )

            if is_training:

                loss.backward()
                optimizer.step()

        predictions = torch.argmax(
            logits,
            dim=1
        )

        batch_size = y_batch.size(
            0
        )

        total_loss += loss.item() * batch_size
        total_correct += (
            predictions == y_batch
        ).sum().item()
        total_samples += batch_size

    return (
        total_loss / total_samples,
        total_correct / total_samples
    )


def save_torch_model(
    model,
    path
):
    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        path
    )

    return path


def load_torch_model(
    model,
    path,
    device=None
):
    if device is None:

        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.load_state_dict(
        torch.load(
            path,
            map_location=device
        )
    )

    model.to(
        device
    )

    model.eval()

    return model
