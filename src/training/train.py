from pathlib import Path

import joblib


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