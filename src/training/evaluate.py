import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from src.training.metrics import (
    classification_report_dict,
    compute_accuracy,
    compute_macro_f1,
    compute_weighted_f1
)


def generate_confusion_matrix(
    y_true,
    y_pred,
    labels=None
):
    return confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )


def evaluate_model(
    model,
    X,
    y_true,
    class_names,
    model_name=None,
    split=None
):
    y_pred = model.predict(
        X
    )

    labels = np.arange(
        len(class_names)
    )

    summary = {
        "model": model_name,
        "split": split,
        "accuracy": compute_accuracy(
            y_true,
            y_pred
        ),
        "macro_f1": compute_macro_f1(
            y_true,
            y_pred
        ),
        "weighted_f1": compute_weighted_f1(
            y_true,
            y_pred
        )
    }

    report = classification_report_dict(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names
    )

    matrix = generate_confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    return {
        "summary": summary,
        "classification_report": report,
        "confusion_matrix": matrix,
        "predictions": y_pred
    }


def predict_torch_model(
    model,
    X,
    batch_size=32,
    device=None
):
    if device is None:

        device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(
        X,
        np.ndarray
    ):
        X_tensor = torch.from_numpy(
            X
        ).to(
            dtype=torch.float32
        )

    else:
        X_tensor = torch.as_tensor(
            X,
            dtype=torch.float32
        )

    dataset = TensorDataset(
        X_tensor
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model = model.to(
        device
    )

    model.eval()

    predictions = []

    with torch.no_grad():

        for (X_batch,) in data_loader:

            X_batch = X_batch.to(
                device
            )

            logits = model(
                X_batch
            )

            batch_predictions = torch.argmax(
                logits,
                dim=1
            )

            predictions.extend(
                batch_predictions
                .cpu()
                .numpy()
                .tolist()
            )

    return np.array(
        predictions
    )


def evaluate_torch_model(
    model,
    X,
    y_true,
    class_names,
    model_name=None,
    split=None,
    batch_size=32,
    device=None
):
    y_pred = predict_torch_model(
        model,
        X,
        batch_size=batch_size,
        device=device
    )

    labels = np.arange(
        len(class_names)
    )

    summary = {
        "model": model_name,
        "split": split,
        "accuracy": compute_accuracy(
            y_true,
            y_pred
        ),
        "macro_f1": compute_macro_f1(
            y_true,
            y_pred
        ),
        "weighted_f1": compute_weighted_f1(
            y_true,
            y_pred
        )
    }

    report = classification_report_dict(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names
    )

    matrix = generate_confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    return {
        "summary": summary,
        "classification_report": report,
        "confusion_matrix": matrix,
        "predictions": y_pred
    }
