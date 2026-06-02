import numpy as np
from sklearn.metrics import confusion_matrix

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
